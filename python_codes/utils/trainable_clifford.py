"""
================================================================================
 trainable_clifford.py -- a *trainable but exactly-Clifford* entangler
================================================================================
The reused fixed entangler is replaced by a per-layer trainable CZ graph. Each
candidate edge (i,j) carries a logit; a straight-through estimator turns it into a
HARD 0/1 selection in the forward pass, so the entangler applied is always a product
of CZ gates -- a graph-state generator, exactly Clifford, M_2(U_S|+>)=0 at every step
-- while gradients flow through the soft (sigmoid) relaxation. The entangler therefore
adapts its entanglement *geometry* to the target while carrying no nonstabilizerness;
all magic still comes from the trainable single-qubit rotations.

Why only the graph (not a single-qubit Clifford frame): the rotation gates Rx,Rz,Ry are
already fully general single-qubit unitaries, so a trainable single-qubit *Clifford*
frame is subsumed by them and adds nothing. The multi-qubit CZ connectivity is the part
the rotations cannot supply, so it is the meaningful trainable-Clifford degree of freedom.

A CZ layer is diagonal: on basis state x it multiplies the amplitude by
    prod_e (-1)^{s_e x_{e0} x_{e1}} = exp(i*pi * sum_e s_e m_e[x]),   m_e[x]=x_{e0} x_{e1},
so the whole layer is a single diagonal multiply -- cheap and differentiable.
================================================================================
"""
import jax, jax.numpy as jnp, numpy as np
from functools import partial
jax.config.update("jax_enable_x64", True)
from utils.born_machine import apply_1q, _rx, _ry, _rz

__all__ = ["grid_edges", "edge_masks", "make_forward_tclifford", "hard_edges",
           "clifford_entangler_gates"]


def grid_edges(N):
    """Candidate edges for an N=2n register split into two n-qubit coordinate blocks:
    the linear chain 0-1-...-(N-1) plus the n cross-register rungs (i, i+n). Gives the
    entangler both intra- and inter-coordinate couplings on the 2D grid."""
    n = N // 2
    chain = [(q, q + 1) for q in range(N - 1)]
    rungs = [(i, i + n) for i in range(n)]
    return tuple(chain + rungs)


def edge_masks(N, edges):
    """(n_edges, 2^N) array m_e[x] = bit(x,e0)*bit(x,e1) over all basis states x."""
    x = np.arange(2 ** N)
    bit = lambda q: (x >> (N - 1 - q)) & 1        # qubit q is bit (N-1-q) (big-endian, matches reshape)
    M = np.stack([bit(a) * bit(b) for (a, b) in edges]).astype(np.float64)
    return jnp.asarray(M)


def _straight_through(logits):
    """Sigmoid soft prob with a hard {0,1} forward value (straight-through gradient)."""
    p = jax.nn.sigmoid(logits)
    hard = (p > 0.5).astype(p.dtype)
    return p + jax.lax.stop_gradient(hard - p)      # value=hard, grad=dp


def make_forward_tclifford(N, L, edges):
    """forward(rot (L,3,N), edge_logits (L, n_edges)) -> (probs 2^N, psi).
    Per layer: Rx,Rz pre-rotations; trainable CZ graph (diagonal, exactly Clifford); Ry post."""
    M = edge_masks(N, edges)                        # (n_edges, 2^N)

    @jax.jit
    def forward(rot, edge_logits):
        psi = jnp.ones(2 ** N, dtype=jnp.complex128) / jnp.sqrt(2.0 ** N)   # |+>^N start
        def layer(psi, carry):
            p_l, el = carry                         # (3,N), (n_edges,)
            for q in range(N):
                psi = apply_1q(psi, _rx(p_l[0, q]), q, N)
                psi = apply_1q(psi, _rz(p_l[1, q]), q, N)
            s = _straight_through(el)               # (n_edges,) hard 0/1
            phase = jnp.exp(1j * np.pi * (s @ M))   # (2^N,) diagonal CZ-graph phase
            psi = psi * phase
            for q in range(N):
                psi = apply_1q(psi, _ry(p_l[2, q]), q, N)
            return psi, None
        psi, _ = jax.lax.scan(layer, psi, (rot, edge_logits))
        return jnp.abs(psi) ** 2, psi
    return forward


def hard_edges(edge_logits):
    """Discrete edge selection actually used in the forward (n_layers, n_edges) in {0,1}."""
    return (jax.nn.sigmoid(edge_logits) > 0.5).astype(jnp.int32)
