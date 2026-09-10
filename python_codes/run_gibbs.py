"""
================================================================================
 run_gibbs.py -- Learning spin-glass Gibbs distributions across temperature
================================================================================

PURPOSE
    Test whether the converged loss of a fixed-entangler Born machine depends on
    the entangler's nonstabilizerness (SRE), on a physically motivated,
    magic-demanding target: the Gibbs distribution of a random spin glass.
    Temperature is a physical knob on the target's intrinsic magic M2,min(beta),
    which peaks at intermediate T and vanishes at both T->0 and T->infinity.

MODEL (see spin_glass.py)
    H(s) = sum_{<ij>} J_ij s_i s_j, s_i = +-1. The coupling distribution has mean
    J_mean (ferromagnetic bias) and width J_std (disorder strength). Gibbs target
    p(x) = exp(-beta H(x)) / Z on 2^N bit strings. Disorder is averaged over
    N_DIS independent coupling realizations, each a reproducible integer seed.

ANSATZE (env ANSATZ)
    "main" : QSBM ansatz -- per layer Rx.Rz (pre) . U_S . Ry (post), 3NL angles.
    "A"    : alternating single-axis Rx/Ry per layer . U_S, ending in Rx.  N(L+1) angles.
    "B"    : interior Rz then alternating Rx/Ry per layer . U_S, ending in Rx. N(2L+1) angles.

PROTOCOL
    Fixed depth L, four entangler families, PAIRED initialization (identical
    rotation-angle init reused across every realization, family, temperature;
    only the fixed entangler U_S differs).

STORAGE  (one file PER CONFIG; FULLY SELF-CONTAINED -- reproduce the datapoint
          from the file alone, no re-derivation from seed rules)
    data/runs/gibbs__N{N}__{model}__Jmean{J_mean}__Jstd{J_std}__{ansatz}__L{L}__
              {entangler}__realization_idx{r}__beta{beta}.npz
      -> kld (N_REAL,)               converged KLD per training seed
         loss_history (N_REAL,E)     full per-epoch NLL trajectory (convergence auditable)
         params (N_REAL,*shape)      ALL trained angles (every seed, not just the best)
         init_angles (N_REAL,*shape) shared paired initialization
         entangler_gates (N_REAL,ng,4,4)  the fixed U_S actually used (per seed)
         pairs (ng,2)                brickwork bonds
         target_p (2^N,)             the exact Gibbs target distribution
         best_idx, m2_trained, m2min_target = M2(sqrt p), meta (JSON, all fields+seed rules)
    data/runs/gibbs_couplings__N{N}__{model}__Jmean{J_mean}__Jstd{J_std}.npz
      -> couplings J for every realization_idx (explicit reproducibility)

PARALLELISM
    Small statevectors do not spread across CPU cores, so shard the
    (realization_idx, entangler) chains across worker processes: launch N_WORKERS
    copies with WORKER_ID = 0..N_WORKERS-1.

USAGE
    JAX_PLATFORMS=cpu GRID_N=10 ANSATZ=main N_DIS=10 \
        N_WORKERS=12 WORKER_ID=0 python3 run_gibbs.py
    (env: SG_MODEL, DISORDER_J_MEAN, DISORDER_J_STD, BETA_STEP, BETA_MAX, GIBBS_L)
================================================================================
"""
import os, sys, json, time
os.environ.setdefault("N_REAL", "10")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
import run_sweep as R
from utils.born_machine import (make_forward, kld_from_probs, apply_1q,
                                 apply_scrambler, zero_state, _rx, _ry, _rz)
from utils.sre import stabilizer_renyi_entropy
from utils.train import make_trainer
from utils.store import runs_dir
import spin_glass as SG

# ------------------------------------------------------------------ parameters
N          = int(os.environ.get("GRID_N", "10"))
K          = 2 * N
N_REAL     = int(os.environ.get("N_REAL", "10"))          # training realizations (rotation-init seeds)
N_EPOCHS   = int(os.environ.get("N_EPOCHS", "1000"))
L          = int(os.environ.get("GIBBS_L", "12"))
ANSATZ     = os.environ.get("ANSATZ", "main")             # "main" | "A" | "B"
MODEL      = os.environ.get("SG_MODEL", "SK")             # canonical spin glass (spin_glass.py)
J_MEAN     = float(os.environ.get("DISORDER_J_MEAN", "0.0"))   # J_0, ferromagnetic bias
J_STD      = float(os.environ.get("DISORDER_J_STD", "1.0"))    # sigma_0, disorder strength
N_DIS      = int(os.environ.get("N_DIS", "10"))           # disorder realizations, realization_idx 0..N_DIS-1
FAMS       = os.environ.get("GIBBS_FAMS", "clifford,doped,haar,matchgate").split(",")
NT_DOPED   = 4                                            # T-gates for the doped-Clifford family
BETA_STEP  = float(os.environ.get("BETA_STEP", "0.2"))
BETA_MAX   = float(os.environ.get("BETA_MAX", "4.0"))
# BETA_LIST (comma-separated) overrides the uniform grid, so a non-uniform grid
# (fine near the magic peak, coarse in the tail) is explicit and reproducible.
_BLIST     = os.environ.get("BETA_LIST", "").strip()
BETAS      = ([round(float(b), 2) for b in _BLIST.split(",")] if _BLIST
              else [round(b, 2) for b in np.arange(0.0, BETA_MAX + 1e-4, BETA_STEP)])


def _fmt(x):
    """Format a float for a filename field: 1.60 -> 1p60, -0.50 -> m0p50."""
    return f"{x:.2f}".replace("-", "m").replace(".", "p")

def realization_seed(realization_idx):
    """Reproducible RNG seed for a disorder realization."""
    return 200 + realization_idx

def energies_of(realization_idx):
    """Spin-glass energies H(x) for a disorder realization."""
    return SG.energies(MODEL, N, realization_seed(realization_idx), J_MEAN, J_STD)

def gibbs_dist(E, beta):
    """Boltzmann distribution p(x) ~ exp(-beta H(x)); min-energy shift for stability."""
    w = np.exp(-beta * (E - E.min()))
    return w / w.sum()

def gibbs_filename(entangler, realization_idx, beta):
    """Per-config filename; every field spelled out and consistent with meta."""
    return (f"gibbs__N{N}__{MODEL}__Jmean{_fmt(J_MEAN)}__Jstd{_fmt(J_STD)}"
            f"__{ANSATZ}__L{L}__{entangler}"
            f"__realization_idx{realization_idx}__beta{_fmt(beta)}.npz")

def couplings_filename():
    return f"gibbs_couplings__N{N}__{MODEL}__Jmean{_fmt(J_MEAN)}__Jstd{_fmt(J_STD)}.npz"


# --------------------------------------------------------- ansatz forward maps
def make_fwd_main(N, L, pairs):
    return make_forward(N, L, pairs)                      # QSBM: Rx.Rz | U_S | Ry, params (L,3,N)

def make_fwd_A(N, L, pairs):
    """A: alternating single-axis Rx/Ry per layer + U_S, ending in Rx. params (L+1, N)."""
    @jax.jit
    def fwd(params, gates):
        psi = zero_state(N)
        for l in range(L):
            rot = _rx if l % 2 == 0 else _ry
            for q in range(N):
                psi = apply_1q(psi, rot(params[l, q]), q, N)
            psi = apply_scrambler(psi, gates, pairs, N)
        for q in range(N):
            psi = apply_1q(psi, _rx(params[L, q]), q, N)  # final Rx (acts on Born output)
        return jnp.abs(psi) ** 2, psi
    return fwd

def make_fwd_B(N, L, pairs):
    """B: interior Rz then alternating Rx/Ry per layer + U_S, ending in Rx. params (2L+1, N)."""
    @jax.jit
    def fwd(params, gates):
        psi = zero_state(N)
        for l in range(L):
            for q in range(N):
                psi = apply_1q(psi, _rz(params[2 * l, q]), q, N)        # interior Rz
            rot = _rx if l % 2 == 0 else _ry
            for q in range(N):
                psi = apply_1q(psi, rot(params[2 * l + 1, q]), q, N)
            psi = apply_scrambler(psi, gates, pairs, N)
        for q in range(N):
            psi = apply_1q(psi, _rx(params[2 * L, q]), q, N)            # final Rx
        return jnp.abs(psi) ** 2, psi
    return fwd

_ANSATZ = {"main": (make_fwd_main, lambda L: (L, 3, N)),
           "A":    (make_fwd_A,    lambda L: (L + 1, N)),
           "B":    (make_fwd_B,    lambda L: (2 * L + 1, N))}


# ---------------------------------------------------------------------- driver
def m2_state(psi):
    return float(stabilizer_renyi_entropy(jnp.asarray(psi), N))

def m2_sqrt(p):
    return float(stabilizer_renyi_entropy(jnp.asarray(np.sqrt(p)).astype(jnp.complex128), N))

def main():
    make_fwd, param_shape = _ANSATZ[ANSATZ]
    group = R.load_clifford_group(verbose=False)
    print(f"Gibbs  N={N} L={L} model={MODEL} J_mean={J_MEAN} J_std={J_STD} ansatz={ANSATZ} "
          f"disorder={N_DIS} beta={BETAS[0]}..{BETAS[-1]} ({len(BETAS)} pts)", flush=True)

    # explicit reproducibility: store J for every realization (idempotent under many workers)
    cpath = os.path.join(runs_dir(), couplings_filename())
    if not os.path.exists(cpath):
        try:
            Js = np.stack([SG.couplings(MODEL, N, realization_seed(r), J_MEAN, J_STD) for r in range(N_DIS)])
            np.savez(cpath, couplings=Js, seeds=np.array([realization_seed(r) for r in range(N_DIS)]),
                     N=N, model=MODEL, J_mean=J_MEAN, J_std=J_STD, seed_rule="200+realization_idx")
        except Exception:
            pass

    # PAIRED init: identical rotation-angle init reused across every realization/family/temperature
    init = jnp.stack([jax.random.uniform(jax.random.PRNGKey(900 + r), param_shape(L),
                                         minval=0.0, maxval=2 * np.pi) for r in range(N_REAL)])

    # shard the (realization_idx, entangler) chains across worker processes
    chains = [(realization_idx, fam) for realization_idx in range(N_DIS) for fam in FAMS]
    if os.environ.get("N_WORKERS"):
        wid, nw = int(os.environ.get("WORKER_ID", "0")), int(os.environ["N_WORKERS"])
        chains = [c for i, c in enumerate(chains) if i % nw == wid]
        print(f"[worker {wid}/{nw}] {len(chains)} chains", flush=True)

    for realization_idx, fam in chains:
        E = energies_of(realization_idx)
        n_t = NT_DOPED if fam == "doped" else 0
        ent, pairs = R.build_entangler(fam, N, K, n_t, group)   # fixed U_S; pairs identical across families
        fwd = make_fwd(N, L, pairs)
        trainer = make_trainer(fwd, N_EPOCHS, R.LR, R.CLIP, alpha=R.ALPHA)
        for beta in BETAS:
            path = os.path.join(runs_dir(), gibbs_filename(fam, realization_idx, beta))
            if os.path.exists(path):
                continue
            pv = gibbs_dist(E, beta); p = jnp.asarray(pv); t0 = time.time()
            params, losses = trainer(ent, init, p)           # losses: (N_REAL, N_EPOCHS) full trajectory
            klds = np.array([float(kld_from_probs(p, fwd(params[r], ent[r])[0])) for r in range(N_REAL)])
            best = int(klds.argmin())
            m2tr = m2_state(fwd(params[best], ent[best])[1])
            m2min = m2_sqrt(pv)
            meta = dict(sweep="gibbs", N=N, K=K, L=L, model=MODEL, J_mean=J_MEAN, J_std=J_STD,
                        ansatz=ANSATZ, entangler=fam, n_t=n_t, realization_idx=realization_idx,
                        beta=float(beta), n_real=N_REAL, n_epochs=N_EPOCHS,
                        lr=R.LR, clip=R.CLIP, alpha=R.ALPHA, param_shape=list(param_shape(L)),
                        n_gates=int(len(pairs)), best_idx=best,
                        realization_seed=realization_seed(realization_idx),
                        init_seed_rule="uniform(PRNGKey(900+r), param_shape) for r in 0..N_REAL-1",
                        entangler_seed_base=int(R.BASE),
                        entangler_seed_rule="split(PRNGKey(BASE+991*fam_id+100*K+n_t), N_REAL); "
                                            "fam_id={haar:0,clifford:1,doped:2,matchgate:3}")
            # FULLY SELF-CONTAINED FILE: everything needed to reproduce this datapoint without
            # re-deriving anything from seed rules -- target, entangler, all angles, loss trajectory.
            np.savez(
                path,
                kld=klds,                                          # (N_REAL,) converged KLD per training seed
                loss_history=np.asarray(losses, dtype=np.float32), # (N_REAL, N_EPOCHS) NLL trajectory
                params=np.asarray(params),                         # (N_REAL, *param_shape) ALL trained angles
                best_idx=best,                                     # argmin KLD training seed
                init_angles=np.asarray(init),                      # (N_REAL, *param_shape) shared paired init
                entangler_gates=np.asarray(ent),                   # (N_REAL, n_gates, 4, 4) fixed U_S per seed
                pairs=np.asarray(pairs, dtype=np.int32),           # (n_gates, 2) brickwork bonds
                target_p=np.asarray(pv, dtype=np.float64),         # (2^N,) the exact Gibbs target
                m2_trained=m2tr,                                   # SRE of best-seed trained state
                m2min_target=m2min,                                # M2(sqrt p): target nonstabilizerness
                meta=json.dumps(meta),
            )
            print(f"  realization_idx={realization_idx} {fam:>10} beta={beta:<4} "
                  f"KLD med={np.median(klds):.4f} best={klds[best]:.4f} m2tr={m2tr:.2f} "
                  f"m2min={m2min:.2f} [{time.time()-t0:.0f}s]", flush=True)
    print(f"GIBBS DONE (N={N} {MODEL} {ANSATZ})", flush=True)


if __name__ == "__main__":
    main()
