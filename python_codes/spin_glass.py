"""
spin_glass.py -- Canonical random spin-glass Hamiltonians and Gibbs targets.

A small, self-contained library of textbook disordered Ising models, used to
build classically-hard, physically-motivated generative-modeling targets:
the Gibbs distribution  p(x) = exp(-beta H(x)) / Z  over the 2^N spin configs.

Only canonical, well-established ensembles are used:

  "SK"    Sherrington-Kirkpatrick: fully connected, Gaussian couplings
          J_ij ~ N(0,1)/sqrt(N). The paradigmatic mean-field spin glass
          (Parisi's replica-symmetry-breaking solution, 2021 Physics Nobel).

  "SKpm"  Sherrington-Kirkpatrick, bimodal +-J: fully connected,
          J_ij ~ +-1/sqrt(N) with equal probability. Same universality class,
          discrete disorder.

  "EA2D"  Edwards-Anderson: nearest-neighbor Gaussian couplings on a periodic
          sqrt(N) x sqrt(N) square lattice -- the canonical *short-range* spin
          glass (requires N a perfect square).

Each Hamiltonian is  H(s) = sum_{<ij>} J_ij s_i s_j  with s_i = +-1, and the
disorder is fixed by an integer seed so every realization is reproducible.
"""
import numpy as np

MODELS = ("SK", "SKpm", "EA2D")


def couplings(model, N, seed, J_mean=0.0, J_std=1.0):
    """Symmetric coupling matrix J (N x N, zero diagonal). The coupling distribution has
    mean J_mean (J_0, ferromagnetic bias) and width J_std (sigma_0, disorder strength),
    both O(1); overall 1/sqrt(N) SK normalization. J_mean=0, J_std=1 is the pure spin glass.

      SK   : J_ij ~ Normal(J_mean, J_std) / sqrt(N)              (fully connected, Gaussian)
      SKpm : J_ij = (J_mean +- J_std) / sqrt(N)                 (fully connected, bimodal)
      EA2D : nearest-neighbor Normal(J_mean, J_std)/sqrt(z) on a periodic sqrt(N)xsqrt(N) lattice

    For J_mean=0 (the case used throughout) this matches the textbook SK model exactly:
    std(J_ij)=J_std/sqrt(N) is the canonical width J/sqrt(N), and the spin-glass freezing
    temperature is T_c=J_std (beta_c=1 for J_std=1).
    CAVEAT: the ferromagnetic mean here scales as J_mean/sqrt(N), whereas the original SK model
    scales the mean as J_0/N (a *different* N-power from the width). The two agree only at
    J_mean=0; if you ever use a ferromagnetic bias, rescale J_mean by 1/sqrt(N) to recover
    the standard SK (de Almeida-Thouless) phase diagram.
    """
    rng = np.random.default_rng(seed)
    if model == "SK":
        # Mirror the upper triangle (each off-diagonal is ONE N(0,1) draw) so the
        # coupling width is preserved: std(J_ij*sqrt(N)) = J_std exactly (true
        # standard SK, sigma_0 = J_std). NB: averaging (z+z.T)/2 would instead halve
        # the variance to J_std/sqrt(2); the mirror matches the SKpm branch below.
        z = rng.standard_normal((N, N)); z = np.triu(z, 1); z = z + z.T
        J = (J_mean + J_std * z) / np.sqrt(N)
        np.fill_diagonal(J, 0.0)
    elif model == "SKpm":
        s = rng.integers(0, 2, (N, N)) * 2 - 1          # +-1
        s = np.triu(s, 1); s = s + s.T
        J = (J_mean + J_std * s) / np.sqrt(N)
        np.fill_diagonal(J, 0.0)
    elif model == "EA2D":
        Ld = int(round(np.sqrt(N)))
        assert Ld * Ld == N, f"EA2D needs N a perfect square, got N={N}"
        J = np.zeros((N, N))
        idx = lambda a, b: (a % Ld) * Ld + (b % Ld)
        for a in range(Ld):
            for b in range(Ld):
                i = idx(a, b)
                for di, dj in ((1, 0), (0, 1)):         # right and down bonds (periodic)
                    j = idx(a + di, b + dj)
                    w = (J_mean + J_std * rng.standard_normal()) / np.sqrt(4)   # coordination 4
                    J[i, j] += w; J[j, i] += w
    else:
        raise ValueError(f"unknown spin-glass model {model!r}")
    return J


def energies(model, N, seed, J_mean=0.0, J_std=1.0):
    """Energy H(x) = sum_{<ij>} J_ij s_i s_j for every bit string x (s = 1 - 2*bit)."""
    d = 2 ** N
    J = couplings(model, N, seed, J_mean, J_std)
    bits = ((np.arange(d)[:, None] >> np.arange(N)) & 1)
    s = (1 - 2 * bits).astype(float)                    # (d, N) spins in {+1,-1}
    return 0.5 * np.einsum("ai,ij,aj->a", s, J, s)      # 0.5 undoes the double count (J symmetric)


def gibbs(model, N, beta, seed, J_mean=0.0, J_std=1.0):
    """Gibbs distribution p(x) ~ exp(-beta H(x)); min-energy shift for numerical stability."""
    E = energies(model, N, seed, J_mean, J_std)
    w = np.exp(-beta * (E - E.min()))
    return w / w.sum()
