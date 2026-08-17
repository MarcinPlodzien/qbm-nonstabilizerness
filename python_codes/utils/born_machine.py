"""
born_machine.py -- scrambled Born machine, statevector, JAX (jit + vmap).

ARCHITECTURE

    |0...0>  --[ Rx(th) Rz(th) ]--[ U_S ]--[ Ry(th) ]--  x L layers  -->  Born rule

    * U_S is a FIXED brickwork circuit of depth K, reused in every layer.
      Its two-qubit gates are drawn either Haar (entangle + create magic) or
      Clifford (entangle, create NO magic).  That is the control variable.
    * Only the 3*N*L rotation angles are trained.

MATRIX-FREE SIMULATION
    The state is a (2,)*N tensor.  A 1-qubit gate is a tensordot on one axis; a
    2-qubit gate is a reshape + 4x4 matmul on two axes.  No 2^N x 2^N matrix is
    ever formed.  Cost per gate: O(2^N).

PERFORMANCE
    * `make_forward` closes over N, L and the (static) brickwork pairs, so the
      whole circuit unrolls once and XLA fuses it.
    * Training is a `lax.scan` over epochs -- one compiled step, no Python loop.
    * `train_batch` vmaps over seeds: every seed has its own scrambler instance
      AND its own initial angles, all trained in parallel in one kernel.

EXACT PROBABILITIES
    Exact Born probabilities, not shot samples.  A finite-shot estimate carries a
    sampling floor  E[KL] ~ (M-1) / (2 N_shots)  (~0.026 for M = 256 bins and
    5000 shots), which is the same order as the Clifford/Haar gap of interest.
    Exact probabilities remove that floor.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

__all__ = ["brickwork_pairs", "plus_state", "zero_state", "apply_1q", "apply_2q",
           "apply_scrambler", "make_forward", "layer_states", "make_loss",
           "train_batch", "kld_from_probs"]


# ------------------------------------------------------------------------------
# Single-qubit rotation generators
# ------------------------------------------------------------------------------
@jax.jit
def _rx(t):
    c, s = jnp.cos(t / 2), jnp.sin(t / 2)
    return jnp.array([[c + 0j, -1j * s], [-1j * s, c + 0j]])


@jax.jit
def _ry(t):
    c, s = jnp.cos(t / 2), jnp.sin(t / 2)
    return jnp.array([[c + 0j, -s + 0j], [s + 0j, c + 0j]])


@jax.jit
def _rz(t):
    e = jnp.exp(-1j * t / 2)
    return jnp.array([[e, 0j], [0j, jnp.conj(e)]])


# ------------------------------------------------------------------------------
# Matrix-free gate application
# ------------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(2, 3))
def apply_1q(psi, gate, q, N):
    """Apply a 2x2 gate to qubit q via a single einsum contraction (matrix-free).

    The integer-subscript einsum contracts the gate's input leg with axis q of the
    (2,)*N state tensor and writes the output leg back to axis q, with no explicit
    transpose (moveaxis); benchmarked ~1.5x faster than the tensordot form at N<=10.
    """
    st = psi.reshape((2,) * N)                       # state as an N-leg tensor
    out_idx = list(range(N))
    out_idx[q] = N                                   # gate output leg -> new label N
    st = jnp.einsum(gate, [N, q], st, list(range(N)), out_idx)
    return st.reshape(-1)


@partial(jax.jit, static_argnums=(2, 3, 4))
def apply_2q(psi, gate, q0, q1, N):
    """Apply a 4x4 gate to qubits (q0,q1) via one einsum contraction (matrix-free).

    The 4x4 gate is viewed as a (2,2,2,2) tensor and its two input legs are contracted
    with axes q0,q1 of the state; the two output legs replace them in place. No dense
    2^N x 2^N operator and no moveaxis.
    """
    st = psi.reshape((2,) * N)                       # state as an N-leg tensor
    g = gate.reshape(2, 2, 2, 2)
    out_idx = list(range(N))
    out_idx[q0] = N
    out_idx[q1] = N + 1                              # gate output legs -> labels N, N+1
    st = jnp.einsum(g, [N, N + 1, q0, q1], st, list(range(N)), out_idx)
    return st.reshape(-1)


# ------------------------------------------------------------------------------
# States and topology
# ------------------------------------------------------------------------------
def brickwork_pairs(N, depth):
    """Brickwork layout: even bonds on even layers, odd bonds on odd layers."""
    return tuple((q, q + 1) for d in range(depth) for q in range(d % 2, N - 1, 2))


def zero_state(N):
    psi = jnp.zeros(2 ** N, dtype=jnp.complex128)
    return psi.at[0].set(1.0)


def plus_state(N):
    return jnp.ones(2 ** N, dtype=jnp.complex128) / jnp.sqrt(2.0 ** N)


@partial(jax.jit, static_argnums=(2, 3))
def apply_scrambler(psi, gates, pairs, N):
    """Apply the fixed brickwork U_S. `gates[i]` acts on `pairs[i]`."""
    for i, (a, b) in enumerate(pairs):
        psi = apply_2q(psi, gates[i], a, b, N)
    return psi


# ------------------------------------------------------------------------------
# Forward map:  params (L, 3, N) -> Born probabilities
# ------------------------------------------------------------------------------
def make_forward(N, L, pairs):
    """Return a jitted forward(params (L,3,N), gates) -> (probs of length 2^N, psi).

    The L layers are a `lax.scan` over `params` (the entangler U_S is identical in every
    layer, so only the per-layer angles vary). This keeps the compiled graph O(1) in L --
    one layer body plus a scan -- instead of unrolling into an L-deep graph whose XLA
    compilation time grows with L (and, deep, dominates the run). Numerically identical to
    the unrolled loop: same gate order (Rx,Rz per qubit; U_S; Ry per qubit)."""

    @jax.jit
    def forward(params, gates):
        def layer(psi, p_l):                         # p_l: (3, N) -- one layer's angles
            for q in range(N):                       # pre-rotations Rx, Rz
                psi = apply_1q(psi, _rx(p_l[0, q]), q, N)
                psi = apply_1q(psi, _rz(p_l[1, q]), q, N)
            psi = apply_scrambler(psi, gates, pairs, N)   # fixed entangler, reused every layer
            for q in range(N):                       # post-rotation Ry
                psi = apply_1q(psi, _ry(p_l[2, q]), q, N)
            return psi, None

        psi, _ = jax.lax.scan(layer, zero_state(N), params)
        return jnp.abs(psi) ** 2, psi

    return forward


def layer_states(params, gates, pairs, N, L):
    """Replay a TRAINED circuit and yield the statevector after each layer, l=0..L.

    Reconstructs the exact ansatz of `make_forward` (same gate order: Rx,Rz per qubit,
    then U_S, then Ry) from STORED weights `params` (L,3,N) and the STORED entangler
    `gates` (n_gates,4,4) acting on `pairs`. The first yield is |0...0> (l=0, before any
    layer); the l-th subsequent yield is the state after layer l. Any per-layer quantity
    (half-chain SvN, stabilizer Renyi entropy, ...) is then a map over this generator, so
    the SvN/SRE-vs-layer trace of any trained model is recomputable from the saved run.
    """
    pairs = tuple((int(a), int(b)) for a, b in pairs)    # accept stored ndarray; apply_scrambler needs it hashable
    psi = zero_state(N)
    yield psi                                            # l = 0: input state
    for l in range(L):
        for q in range(N):
            psi = apply_1q(psi, _rx(params[l, 0, q]), q, N)
            psi = apply_1q(psi, _rz(params[l, 1, q]), q, N)
        psi = apply_scrambler(psi, gates, pairs, N)
        for q in range(N):
            psi = apply_1q(psi, _ry(params[l, 2, q]), q, N)
        yield psi                                        # state after layer l


def make_forward_dense(N, L):
    """forward(params, U) with a DENSE entangler U (2^N x 2^N), reused every layer.

    Same ansatz as make_forward but U_S is a full unitary applied as a matmul U @ psi
    (used for the Haar-unitary entangler). The single-qubit rotations stay matrix-free.
    """

    @jax.jit
    def forward(params, U):
        psi = zero_state(N)
        for l in range(L):
            for q in range(N):
                psi = apply_1q(psi, _rx(params[l, 0, q]), q, N)
                psi = apply_1q(psi, _rz(params[l, 1, q]), q, N)
            psi = U @ psi                                 # dense entangler
            for q in range(N):
                psi = apply_1q(psi, _ry(params[l, 2, q]), q, N)
        return jnp.abs(psi) ** 2, psi

    return forward


# ------------------------------------------------------------------------------
# Loss / metrics
# ------------------------------------------------------------------------------
_EPS = 1e-12


def make_loss(forward, target):
    """Negative log-likelihood; minimising it minimises KLD(target || model)."""

    @jax.jit
    def loss(params, gates):
        probs, _ = forward(params, gates)
        return -jnp.sum(target * jnp.log(probs + _EPS))

    return loss


@jax.jit
def kld_from_probs(p_target, q_model):
    """KL(target || model): the reported figure of merit."""
    p = jnp.clip(p_target, _EPS, None)
    q = jnp.clip(q_model, _EPS, None)
    return jnp.sum(p * jnp.log(p / q))


# ------------------------------------------------------------------------------
# Training: one compiled scan over epochs, vmapped over seeds
# ------------------------------------------------------------------------------
def train_batch(forward, target, gates_batch, params_init_batch, n_epochs, lr,
                clip_norm=1.0):
    """Train a batch of independent runs in parallel.

    gates_batch       (S, n_gates, 4, 4)  -- one scrambler instance per seed
    params_init_batch (S, L, 3, N)        -- one initial angle set per seed

    Returns (final_params (S,...), loss_history (S, n_epochs)).
    """
    loss_fn = make_loss(forward, target)
    opt = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(lr))

    def train_one(gates, params0):
        state0 = opt.init(params0)

        def step(carry, _):
            params, state = carry
            val, grads = jax.value_and_grad(loss_fn)(params, gates)
            updates, state = opt.update(grads, state, params)
            params = optax.apply_updates(params, updates)
            return (params, state), val

        (params, _), losses = jax.lax.scan(step, (params0, state0), None, length=n_epochs)
        return params, losses

    return jax.jit(jax.vmap(train_one))(gates_batch, params_init_batch)


