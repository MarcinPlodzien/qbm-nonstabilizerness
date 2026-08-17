"""
sre.py -- Stabilizer Renyi Entropy (magic) and entanglement, in JAX.

The n-th stabilizer Renyi entropy of a pure state |psi> on N qubits is

    M_n(psi) = 1/(1-n) * log2 [ (1/d) * sum_P |<psi|P|psi>|^{2n} ],    d = 2^N

where P runs over all 4^N Pauli strings.  M_n = 0 iff |psi> is a stabilizer
state, and M_n > 0 measures "magic" / nonstabilizerness.

COST OF THE NAIVE SUM
    4^N Pauli strings x O(2^N) per expectation value = O(8^N).  N=12 already
    costs ~90 s.

FAST WALSH-HADAMARD METHOD   (Sierant, Valles-Muns & Garcia-Saez, 2601.07824)
    Split a Pauli string into its X-part x and Z-part z.  For fixed x define

        C_x[j] = conj(psi[j]) * psi[j XOR x]

    Then the Walsh-Hadamard transform of C_x over j yields, in one shot, the
    expectation values <psi| P_{x,z} |psi> for *all* 2^N choices of z (up to an
    irrelevant phase that drops out under |.|^{2n}).  Cost O(N 2^N) per x, and
    there are 2^N values of x:  O(N 4^N) total.

    N=8:   4^8 = 65536 Paulis, evaluated in ~5e5 flops instead of ~1.7e7.

TWO PROPERTIES USED THROUGHOUT
    * M_n is invariant under Clifford unitaries.  A Clifford entangler adds
      exactly zero magic; all magic then comes from the local rotations.
    * Stabilizer states have flat Born distributions (uniform on an affine
      subspace), so magic is necessary for a Born machine to be expressive.
"""
from functools import partial

import jax
import jax.numpy as jnp

# Magic is a sum of 4^N small numbers; float32 loses it. Enable x64 once, here.
jax.config.update("jax_enable_x64", True)

__all__ = ["fwht", "stabilizer_renyi_entropy", "half_chain_entropy", "batched_sre"]


# ------------------------------------------------------------------------------
# Fast Walsh-Hadamard transform
# ------------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(1,))
def fwht(v: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
    """Unnormalised H^{otimes N} applied to a length-2^N vector.  O(N 2^N).

    Implemented as N butterfly stages on the (2,)*N tensor view: no 2^N x 2^N
    matrix is ever built.  `n_qubits` is static so the loop unrolls at trace
    time and XLA fuses the stages.
    """
    v = v.reshape((2,) * n_qubits)
    for q in range(n_qubits):
        v = jnp.moveaxis(v, q, 0)
        a, b = v[0], v[1]
        v = jnp.stack([a + b, a - b], axis=0)
        v = jnp.moveaxis(v, 0, q)
    return v.reshape(-1)


# ------------------------------------------------------------------------------
# Stabilizer Renyi entropy
# ------------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(1, 2))
def stabilizer_renyi_entropy(psi: jnp.ndarray, n_qubits: int, n: int = 2) -> jnp.ndarray:
    """M_n(psi) for a normalised pure state.  Default n=2 (the usual M_2).

    The outer `scan` runs over the 2^N X-parts; each step FWHTs one C_x vector.
    Using `scan` rather than a python loop keeps the jaxpr small (one compiled
    body reused 2^N times) so compile time does not blow up with N.
    """
    d = 2 ** n_qubits
    j = jnp.arange(d)

    def body(acc, x):
        C = jnp.conj(psi) * psi[jnp.bitwise_xor(j, x)]   # C_x[j]
        F = fwht(C, n_qubits)                            # all <P_{x,z}> at once
        return acc + jnp.sum(jnp.abs(F) ** (2 * n)), None

    total, _ = jax.lax.scan(body, jnp.asarray(0.0, jnp.float64), j)
    # sum_P |<P>|^{2n} = total ; the 1/d normalisation is the standard one
    return jnp.log2(total / d) / (1 - n)


# vmap over a batch of states (e.g. one per training seed)
batched_sre = jax.jit(
    jax.vmap(stabilizer_renyi_entropy, in_axes=(0, None, None)),
    static_argnums=(1, 2),
)


# ------------------------------------------------------------------------------
# Entanglement, to certify the matched-scrambling condition
# ------------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(1,))
def half_chain_entropy(psi: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
    """von Neumann entropy (nats) of the left ceil(N/2) qubits.

    Compared against the Page value to certify that the Clifford and Haar
    scramblers are compared at *matched* entanglement -- otherwise the experiment
    varies entanglement and magic together.
    """
    n_left = n_qubits // 2
    m = psi.reshape(2 ** n_left, -1)
    s = jnp.linalg.svd(m, compute_uv=False)
    p = jnp.clip(s ** 2, 1e-300, None)
    return -jnp.sum(p * jnp.log(p))


def page_entropy(n_qubits: int) -> float:
    """Average entropy of a Haar-random pure state across an equal bipartition."""
    dA = 2 ** (n_qubits // 2)
    dB = 2 ** (n_qubits - n_qubits // 2)
    return float(sum(1.0 / k for k in range(dB + 1, dA * dB + 1)) - (dA - 1) / (2.0 * dB))
