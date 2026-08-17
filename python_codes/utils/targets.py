"""
targets.py -- target probability distributions.

All targets are returned as a normalised jnp vector of length 2^N, indexed by the
computational basis state read as a bitstring.  For 2D targets the N qubits are
split into two registers of n_x = n_y = N/2 qubits and the flat index is

        idx = x * 2^{n_y} + y          (x = row register, y = column register)

(row-major, so a 2D density reshapes as p.reshape(2**n_x, 2**n_y)).

REGISTRY
    Add a new target by writing a builder and decorating it with @register.
    `make_target(name, N, **kwargs)` dispatches; `list_targets()` enumerates.
    Nothing else in the codebase needs to change.

AVAILABLE TARGETS
    'multimodal_1d'   five Gaussian peaks, unequal weights, on 2^N bins
    'gaussian_2d'     bivariate Gaussian with correlation rho
    'multimodal_2d'   mixture of four isotropic Gaussians
    'uniform'         flat -- the only kind of distribution a stabilizer state reaches

The 1D multimodal target drives the scrambler sweeps: its sharp inter-peak valleys
make it a non-trivial test of expressivity at small parameter count.
"""
import jax.numpy as jnp
import numpy as np
import optax

__all__ = ["register", "make_target", "list_targets", "target_entropy",
           "magic_cost", "magic_cost_upper"]

_REGISTRY = {}


def register(name):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def list_targets():
    return sorted(_REGISTRY)


def make_target(name: str, n_qubits: int, **kwargs) -> jnp.ndarray:
    """Build a normalised target of length 2^n_qubits."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown target {name!r}; available: {list_targets()}")
    p = _REGISTRY[name](n_qubits, **kwargs)
    p = np.asarray(p, dtype=float)
    assert p.shape == (2 ** n_qubits,), f"{name}: got shape {p.shape}, want {(2**n_qubits,)}"
    s = p.sum()
    assert s > 0 and np.isfinite(s), f"{name}: target does not normalise"
    return jnp.asarray(p / s)


def target_entropy(p) -> float:
    """Shannon entropy H(p) using natural log.  KLD = NLL - H(p), so this converts between them."""
    p = np.asarray(p, dtype=float)
    q = p[p > 0]
    return float(-np.sum(q * np.log(q)))


# ------------------------------------------------------------------------------
# 1D: five Gaussian peaks with unequal weights
# ------------------------------------------------------------------------------
@register("multimodal_1d")
def _multimodal_1d(n_qubits, n_peaks=5, sigma_frac=1 / 20, seed=42, w_lo=0.2, w_hi=2.0, **_):
    """p(x) ~ sum_j w_j exp[-(x - mu_j)^2 / 2 sigma^2] on 2^N bins.

    mu_j    = (j - 0.5) * 2^N / n_peaks          uniformly spaced
    sigma   = 2^N * sigma_frac                   (default 2^N / 20)
    w_j     ~ U(w_lo, w_hi), fixed seed          strongly unequal heights (default 0.2..2.0,
                                                 wider than the QSBM 0.5..1.5 -> harder,
                                                 stresses mode-collapse; standalone-paper choice)
    """
    n_bins = 2 ** n_qubits
    x = np.arange(n_bins)
    mu = np.array([(j - 0.5) * n_bins / n_peaks for j in range(1, n_peaks + 1)])
    w = np.random.default_rng(seed).uniform(w_lo, w_hi, size=n_peaks)
    sigma = n_bins * sigma_frac
    return sum(wj * np.exp(-((x - mj) ** 2) / (2 * sigma ** 2)) for wj, mj in zip(w, mu))


# ------------------------------------------------------------------------------
# 1D skewed: log-normal (asymmetric unimodal) and exponential decay
# ------------------------------------------------------------------------------
@register("lognormal_1d")
def _lognormal_1d(n_qubits, mu_frac=0.25, sigma=0.6, **_):
    """Skewed unimodal p(x) ~ (1/x) exp[-(ln x - mu)^2 / 2 sigma^2] on 2^N bins.

    A right-skewed alternative to the symmetric Gaussian peaks: heavy right tail,
    sharp left edge.  `mu_frac` places the (log-)location; larger sigma = more skew.
    """
    n_bins = 2 ** n_qubits
    x = np.arange(1, n_bins + 1)                     # 1..2^N, avoids log(0)
    mu = np.log(n_bins * mu_frac)
    return np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma ** 2)) / x


@register("exponential_1d")
def _exponential_1d(n_qubits, tau_frac=1 / 6, **_):
    """Monotone right-skewed decay p(x) ~ exp(-x / tau) on 2^N bins."""
    n_bins = 2 ** n_qubits
    x = np.arange(n_bins)
    return np.exp(-x / (n_bins * tau_frac))


# ------------------------------------------------------------------------------
# 2D helpers
# ------------------------------------------------------------------------------
def _grid_2d(n_qubits, lo=-3.0, hi=3.0):
    """Two equal registers mapped to [lo, hi]; returns meshgrid and register size."""
    assert n_qubits % 2 == 0, "2D targets need an even number of qubits"
    nx = n_qubits // 2
    axis = np.linspace(lo, hi, 2 ** nx)
    X, Y = np.meshgrid(axis, axis, indexing="ij")   # idx = x * 2^ny + y
    return X, Y, nx


# ------------------------------------------------------------------------------
# 2D: correlated bivariate Gaussian
# ------------------------------------------------------------------------------
@register("gaussian_2d")
def _gaussian_2d(n_qubits, rho=0.9, **_):
    """p(x,y) ~ exp[-(x^2 - 2 rho x y + y^2) / (2 (1 - rho^2))]"""
    assert -1.0 < rho < 1.0, "rho must lie strictly between -1 and 1"
    X, Y, _nx = _grid_2d(n_qubits)
    p = np.exp(-(X ** 2 - 2 * rho * X * Y + Y ** 2) / (2 * (1 - rho ** 2)))
    return p.reshape(-1)


# ------------------------------------------------------------------------------
# 2D: mixture of four isotropic Gaussians
# ------------------------------------------------------------------------------
@register("multimodal_2d")
def _multimodal_2d(n_qubits, sigma=0.5, centre=1.5, **_):
    """Four peaks at (+-centre, +-centre); the mode-collapse benchmark."""
    X, Y, _nx = _grid_2d(n_qubits)
    p = np.zeros_like(X)
    for mx in (-centre, centre):
        for my in (-centre, centre):
            p += np.exp(-((X - mx) ** 2 + (Y - my) ** 2) / (2 * sigma ** 2))
    return p.reshape(-1)


# ------------------------------------------------------------------------------
# 2D: two interleaving crescents ("two moons") -- a curved, non-Gaussian benchmark
# ------------------------------------------------------------------------------
def _ridge(X, Y, cx, cy, width):
    """Gaussian density ridge: min squared distance from each grid point to a curve (cx,cy)."""
    P = np.stack([X.ravel(), Y.ravel()], 1)
    C = np.stack([cx, cy], 1)
    d2 = ((P[:, None, :] - C[None, :, :]) ** 2).sum(-1).min(1)
    return np.exp(-d2 / (2 * width ** 2)).reshape(-1)


@register("two_moons")
def _two_moons(n_qubits, width=0.30, R=2.0, n_curve=400, **_):
    """Density ridge along the sklearn make_moons geometry: two interleaving crescents,
    centered on the grid. Higher nonstabilizerness cost than the Gaussian-lobe target."""
    X, Y, _nx = _grid_2d(n_qubits)
    t = np.linspace(0, np.pi, n_curve)
    ax, ay = R * np.cos(t), R * np.sin(t) - 0.5
    bx, by = R - R * np.cos(t), -R * np.sin(t) + 0.5
    cx = np.concatenate([ax, bx]); cy = np.concatenate([ay, by])
    cx = cx - cx.mean(); cy = cy - cy.mean()                 # center the two crescents on the grid
    return _ridge(X, Y, cx, cy, width)


@register("two_spirals")
def _two_spirals(n_qubits, width=0.18, turns=2.5, a=2.7, n_curve=700, **_):
    """Two interleaving Archimedean spirals (the classic hard 2D benchmark), centered.
    Higher nonstabilizerness cost and finer structure than the two moons."""
    X, Y, _nx = _grid_2d(n_qubits)
    th = np.linspace(0.35, turns * 2 * np.pi, n_curve)
    r = a * th / (turns * 2 * np.pi)                          # radius grows to ~a
    ax, ay = r * np.cos(th), r * np.sin(th)                   # spiral 1
    bx, by = -ax, -ay                                          # spiral 2 (rotated by pi)
    cx = np.concatenate([ax, bx]); cy = np.concatenate([ay, by])
    return _ridge(X, Y, cx, cy, width)


# ------------------------------------------------------------------------------
# 2D: concentric circular ridges ("rings") -- sklearn make_circles geometry
# ------------------------------------------------------------------------------
@register("three_stripes")
def _three_stripes(n_qubits, offsets=(-2.5, 0.0, 2.5), width=0.5, **_):
    """Three parallel diagonal Gaussian ridges along x - y = const; a clean, well-trainable
    2D benchmark with structure distinct from the rings/moons/lobes."""
    X, Y, _nx = _grid_2d(n_qubits)
    u = X - Y
    p = np.zeros_like(X)
    for c in offsets:
        p += np.exp(-((u - c) ** 2) / (2 * width ** 2))
    return p.reshape(-1)


@register("cross")
def _cross(n_qubits, width=0.4, **_):
    """Diagonal cross: two crossing Gaussian ridges along x-y=0 and x+y=0 (an ``X''), with a
    central intersection; thin diagonal support in two directions at once."""
    X, Y, _nx = _grid_2d(n_qubits)
    p = np.exp(-((X - Y) ** 2) / (2 * width ** 2)) + np.exp(-((X + Y) ** 2) / (2 * width ** 2))
    return p.reshape(-1)


@register("rings")
def _rings(n_qubits, radii=(1.0, 2.3), width=0.26, **_):
    """Sum of Gaussian ridges at fixed radii from the origin; two (or more) concentric rings."""
    X, Y, _nx = _grid_2d(n_qubits)
    r = np.sqrt(X ** 2 + Y ** 2)
    p = np.zeros_like(X)
    for R in radii:
        p += np.exp(-((r - R) ** 2) / (2 * width ** 2))
    return p.reshape(-1)


# ------------------------------------------------------------------------------
# Diagnostic: flat distribution (what a purely Clifford circuit can produce)
# ------------------------------------------------------------------------------
@register("uniform")
def _uniform(n_qubits, **_):
    """Uniform over all 2^N outcomes.

    A stabilizer state's Born distribution is uniform on an affine subspace, so a
    Clifford circuit with Clifford rotation angles can only reach targets of this
    type.  Useful as an end-to-end sanity check of the whole stack.
    """
    return np.ones(2 ** n_qubits)


# ------------------------------------------------------------------------------
# Magic cost of a distribution
# ------------------------------------------------------------------------------
# A stabilizer state's Born distribution is uniform on an affine subspace, so ANY
# non-flat target needs magic.  How much?  The states reproducing a distribution p
# are  |psi_phi> = sum_x sqrt(p_x) e^{i phi_x} |x>  over all phase choices, so
#
#       M(p) = min_{phi} M_2(|psi_phi>)
#
# is the minimum magic any Born machine must carry to represent p exactly.  It is
# a property of the DISTRIBUTION, not of any ansatz -- a lower bound against which
# the trained model's M_2 can be compared.
#
#   magic_cost_upper(p)   M_2 of the real, positive-amplitude representative
#                         (an upper bound on M(p); cheap, deterministic)
#   magic_cost(p)         gradient descent over the 2^N phases (tighter)
# ------------------------------------------------------------------------------
def _amplitudes(p, phases=None):
    import jax.numpy as _jnp
    a = _jnp.sqrt(_jnp.asarray(p))
    return a if phases is None else a * _jnp.exp(1j * phases)


def magic_cost_upper(p, n_qubits: int, n: int = 2) -> float:
    """M_2 of the positive-amplitude state sqrt(p).  Upper bound on the magic cost."""
    from utils.sre import stabilizer_renyi_entropy
    import jax.numpy as _jnp
    psi = _amplitudes(p).astype(_jnp.complex128)
    return float(stabilizer_renyi_entropy(psi, n_qubits, n))


def magic_cost(p, n_qubits: int, n: int = 2, steps: int = 400, lr: float = 0.05,
               n_restarts: int = 4, seed: int = 0) -> float:
    """Upper bound on  M(p) = min_phi M_2( sum_x sqrt(p_x) e^{i phi_x} |x> ).

    The minimisation is non-convex, so this returns an UPPER BOUND, never the
    exact value.  Two safeguards make the bound honest:

      * one restart always starts at phi = 0 (the real, positive-amplitude
        representative), so the result can never exceed `magic_cost_upper`;
      * the running minimum over the whole trajectory is kept, not the endpoint.

    M(p) = 0  iff  p is uniform on an affine subspace of F_2^N, i.e. iff p is the
    Born distribution of some stabilizer state.  A delta and the uniform
    distribution are the two extreme cases, both with M(p) = 0.
    """
    import jax
    import jax.numpy as _jnp
    from utils.sre import stabilizer_renyi_entropy

    p = _jnp.asarray(p)
    d = 2 ** n_qubits

    def obj(phi):
        psi = _amplitudes(p, phi).astype(_jnp.complex128)
        return stabilizer_renyi_entropy(psi, n_qubits, n)

    def descend(phi0):
        opt = optax.adam(lr)
        state = opt.init(phi0)

        def step(carry, _):
            phi, st = carry
            val, g = jax.value_and_grad(obj)(phi)
            upd, st = opt.update(g, st, phi)
            return (optax.apply_updates(phi, upd), st), val

        (_, _), vals = jax.lax.scan(step, (phi0, state), None, length=steps)
        return _jnp.min(vals)

    # restart 0: phi = 0  ->  guarantees best <= magic_cost_upper
    starts = [_jnp.zeros(d)]
    for r in range(max(0, n_restarts - 1)):
        starts.append(jax.random.uniform(jax.random.PRNGKey(seed + r), (d,),
                                         minval=0.0, maxval=2 * np.pi))

    best = float(obj(starts[0]))                 # value at phi = 0, before descending
    for phi0 in starts:
        best = min(best, float(descend(phi0)))
    return best
