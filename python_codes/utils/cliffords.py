"""
cliffords.py -- the two-qubit Clifford group, and gate samplers.

THE CONTROL VARIABLE OF THIS PROJECT

    Haar 2-qubit gates      entangle  and  create magic
    Clifford 2-qubit gates  entangle  but  create no magic   (M_n is Clifford-invariant)

    Random Clifford circuits are 3-designs: at sufficient depth they reproduce
    Haar-random entanglement (Page value), subsystem purities and Renyi-2
    entropies.  Every entanglement-based scrambling diagnostic is therefore blind
    to the difference, which makes them the control for isolating
    nonstabilizerness.

THE GROUP
    |C_2| = 11520 elements modulo global phase.  Enumerated once by breadth-first
    search over the standard generators {H_1, H_2, S_1, S_2, CNOT_12, CNOT_21}.
    Elements are canonicalised by dividing out the phase of the first non-zero
    entry, so U and e^{i phi} U hash to the same key.

    The enumeration takes ~2 s and is cached on disk (see `load_clifford_group`).

T-DOPING
    `doped_clifford_gates` right-multiplies m of the Clifford gates by a T gate on
    one qubit.  This dials the magic the entangler can create continuously from
    zero (m=0) upward, at essentially fixed entangling power: fix entanglement,
    vary magic.
"""
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["two_qubit_clifford_group", "load_clifford_group", "haar_unitary",
           "haar_2q_gates", "clifford_2q_gates", "doped_clifford_2q_gates",
           "matchgate_2q_gates", "deterministic_gates", "sample_scrambler_gates"]

# ------------------------------------------------------------------------------
# Elementary matrices
# ------------------------------------------------------------------------------
_I = np.eye(2, dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
_CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
_CZ = np.diag([1, 1, 1, -1]).astype(complex)

# Fixed 2-qubit gates for DETERMINISTIC (seed-free) entanglers.
#   cluster    CZ*(H(x)H) on every brickwork bond -> cluster/graph-state generation.
#              Clifford => exactly zero magic; reaches near-maximal entanglement.
#              A deterministic, hardware-native zero-magic scrambler.
#   fixed_magic CNOT*(T(x)T): deterministic NON-Clifford entangler (scrambles AND makes
#              magic) -- a fixed analogue of the Haar/doped arm.
_G_CLUSTER = _CZ @ np.kron(_H, _H)
_G_FIXED_MAGIC = _CNOT @ np.kron(_T, _T)

#   ising      self-dual kicked transverse-field Ising at its Clifford point,
#              U_bond = (Rx(pi/2) x Rx(pi/2)) . exp(-i pi/4 Z x Z).  Rx(pi/2) and the
#              pi/4 Ising phase are both Clifford, so U is Clifford => EXACTLY zero magic,
#              yet the non-commuting transverse kick makes it scramble (a pure Ising ZZ
#              term commutes with itself and never entangles).  The analog, hardware-native
#              zero-magic scrambler; being Clifford it shows entanglement revivals in K.
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_RX90 = np.cos(np.pi / 4) * _I - 1j * np.sin(np.pi / 4) * _X          # Rx(pi/2), Clifford
_ZZ_Q = np.diag([np.exp(-1j * np.pi / 4), np.exp(1j * np.pi / 4),
                 np.exp(1j * np.pi / 4), np.exp(-1j * np.pi / 4)])    # exp(-i pi/4 Z x Z)
_G_ISING = np.kron(_RX90, _RX90) @ _ZZ_Q


def _canonical_key(U: np.ndarray, decimals: int = 8):
    """Hashable representative of U modulo a global phase."""
    flat = U.reshape(-1)
    k = int(np.argmax(np.abs(flat) > 1e-9))          # first non-zero entry
    U = U * np.exp(-1j * np.angle(flat[k]))          # fix its phase to be real +
    return tuple(np.round(U.reshape(-1), decimals))


# ------------------------------------------------------------------------------
# Group enumeration
# ------------------------------------------------------------------------------
def two_qubit_clifford_group(verbose: bool = False) -> np.ndarray:
    """All 11520 two-qubit Cliffords (mod phase) as a (11520, 4, 4) complex array."""
    generators = [
        np.kron(_H, _I), np.kron(_I, _H),
        np.kron(_S, _I), np.kron(_I, _S),
        _CNOT, _SWAP @ _CNOT @ _SWAP,                 # CNOT_12 and CNOT_21
    ]
    identity = np.eye(4, dtype=complex)
    seen = {_canonical_key(identity): identity}
    frontier = [identity]
    while frontier:
        new_frontier = []
        for U in frontier:
            for g in generators:
                V = g @ U
                key = _canonical_key(V)
                if key not in seen:
                    seen[key] = V
                    new_frontier.append(V)
        frontier = new_frontier
        if verbose:
            print(f"    clifford BFS: {len(seen)} elements")
    return np.stack(list(seen.values()))


def load_clifford_group(cache_path: str = None, verbose: bool = False) -> np.ndarray:
    """Enumerate once, then reuse.  Cache lives next to this file by default."""
    if cache_path is None:
        cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "clifford_group_2q.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    G = two_qubit_clifford_group(verbose=verbose)
    assert G.shape == (11520, 4, 4), f"expected 11520 Cliffords, got {G.shape[0]}"
    with open(cache_path, "wb") as f:
        pickle.dump(G, f)
    return G


# ------------------------------------------------------------------------------
# Gate samplers.  All return (n_gates, 4, 4) complex arrays.
# ------------------------------------------------------------------------------
def haar_unitary(key, dim: int) -> jnp.ndarray:
    """A single Haar-random unitary of size dim = 2^N (dense).

    Used as the HAAR entangler: a genuine Haar unitary on the whole register, applied as
    a matmul U @ psi. Exact and simplest at small N (dim = 1024 at N=10); the Clifford
    entangler stays matrix-free (brickwork). QR of a complex Ginibre matrix with the
    diagonal-phase fix so the result is Haar-distributed.
    """
    k1, k2 = jax.random.split(key)
    z = (jax.random.normal(k1, (dim, dim)) + 1j * jax.random.normal(k2, (dim, dim))) / jnp.sqrt(2.0)
    q, r = jnp.linalg.qr(z)
    d = jnp.diagonal(r)
    return q * (d / jnp.abs(d))[None, :]


def haar_2q_gates(key, n_gates: int) -> jnp.ndarray:
    """n Haar-random U(4) gates: QR of a complex Ginibre matrix, phases fixed.

    The `q * (d/|d|)` step removes the sign ambiguity of QR, without which the
    result is not Haar-distributed.
    """
    k1, k2 = jax.random.split(key)
    z = (jax.random.normal(k1, (n_gates, 4, 4))
         + 1j * jax.random.normal(k2, (n_gates, 4, 4))) / jnp.sqrt(2.0)
    q, r = jnp.linalg.qr(z)
    d = jnp.diagonal(r, axis1=-2, axis2=-1)
    return q * (d / jnp.abs(d))[:, None, :]


def clifford_2q_gates(key, n_gates: int, group: np.ndarray) -> jnp.ndarray:
    """n gates drawn uniformly from the 11520-element two-qubit Clifford group."""
    idx = jax.random.randint(key, (n_gates,), 0, group.shape[0])
    return jnp.asarray(group)[idx]


def doped_clifford_2q_gates(key, n_gates: int, group: np.ndarray, n_t: int) -> jnp.ndarray:
    """Clifford gates, of which `n_t` are right-multiplied by  T (x) I.

    Interpolates the entangler between zero magic (n_t = 0) and non-Clifford,
    at essentially unchanged entangling power.
    """
    k_gate, k_pick = jax.random.split(key)
    gates = clifford_2q_gates(k_gate, n_gates, group)
    if n_t <= 0:
        return gates
    n_t = min(n_t, n_gates)
    T2 = jnp.asarray(np.kron(_T, _I))
    which = jax.random.choice(k_pick, n_gates, (n_t,), replace=False)
    return gates.at[which].set(gates[which] @ T2)


def _haar_u2(key, n_gates: int) -> jnp.ndarray:
    """n Haar-random U(2) gates (QR of 2x2 Ginibre, phases fixed)."""
    k1, k2 = jax.random.split(key)
    z = (jax.random.normal(k1, (n_gates, 2, 2))
         + 1j * jax.random.normal(k2, (n_gates, 2, 2))) / jnp.sqrt(2.0)
    q, r = jnp.linalg.qr(z)
    d = jnp.diagonal(r, axis1=-2, axis2=-1)
    return q * (d / jnp.abs(d))[:, None, :]


def matchgate_2q_gates(key, n_gates: int) -> jnp.ndarray:
    """n random nearest-neighbour MATCHGATES G(A,B) -- the free-fermion (Gaussian) gates.

    G(A,B) acts as A in U(2) on the even-parity subspace {|00>,|11>} and B in U(2) on the
    odd-parity subspace {|01>,|10>}, subject to the matchgate condition det A = det B:

            [ A00  0    0    A01 ]
        G = [ 0    B00  B01  0   ]
            [ 0    B10  B11  0   ]
            [ A10  0    0    A11 ]     (basis |00>,|01>,|10>,|11>).

    Nearest-neighbour matchgate circuits are exactly the classically-simulable free-fermion
    dynamics (Jordan-Wigner-quadratic). They generate extensive entanglement but NEVER form
    a unitary 2-design -- so this is the SUB-2-design lower-bound probe: an entangler that
    entangles yet lacks the 2nd-moment expressibility we claim is the resource.
    """
    kA, kB = jax.random.split(key)
    A = _haar_u2(kA, n_gates)
    B = _haar_u2(kB, n_gates)
    detA = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
    detB = B[:, 0, 0] * B[:, 1, 1] - B[:, 0, 1] * B[:, 1, 0]
    B = B * jnp.sqrt(detA / detB)[:, None, None]         # enforce det B = det A
    G = jnp.zeros((n_gates, 4, 4), dtype=jnp.complex128)
    G = G.at[:, 0, 0].set(A[:, 0, 0]).at[:, 0, 3].set(A[:, 0, 1])
    G = G.at[:, 3, 0].set(A[:, 1, 0]).at[:, 3, 3].set(A[:, 1, 1])
    G = G.at[:, 1, 1].set(B[:, 0, 0]).at[:, 1, 2].set(B[:, 0, 1])
    G = G.at[:, 2, 1].set(B[:, 1, 0]).at[:, 2, 2].set(B[:, 1, 1])
    return G


def deterministic_gates(n_gates: int, gate: np.ndarray) -> jnp.ndarray:
    """Tile a single fixed 2-qubit gate over every brickwork bond (no seed)."""
    return jnp.broadcast_to(jnp.asarray(gate), (n_gates, 4, 4))


def sample_scrambler_gates(key, kind: str, n_gates: int,
                           group: np.ndarray = None, n_t: int = 0) -> jnp.ndarray:
    """Dispatch on the scrambler kind used throughout the sweep.

    Random families use `key`; deterministic families ('cluster','fixed_magic') ignore it.
    """
    if kind == "haar" or kind == "haar2":        # haar2 = same gates, used at the 2-design depth K2
        return haar_2q_gates(key, n_gates)
    if kind == "matchgate":                      # free-fermion; sub-2-design lower bound
        return matchgate_2q_gates(key, n_gates)
    if kind == "clifford":
        return clifford_2q_gates(key, n_gates, group)
    if kind == "doped_clifford":
        return doped_clifford_2q_gates(key, n_gates, group, n_t)
    if kind == "cluster":                        # deterministic, Clifford, zero magic
        return deterministic_gates(n_gates, _G_CLUSTER)
    if kind == "ising":                          # analog kicked-Ising, Clifford point, zero magic
        return deterministic_gates(n_gates, _G_ISING)
    if kind == "fixed_magic":                    # deterministic, non-Clifford, has magic
        return deterministic_gates(n_gates, _G_FIXED_MAGIC)
    raise ValueError(f"unknown scrambler kind: {kind!r}")
