"""
run_sweep.py -- unified sweep engine; every scan runs through it.

Each scan is a config grid fed to one engine. For each config it trains, then
writes one self-describing file (utils/store.save_run) with:
  * full metadata + hyperparameters + seeds,
  * the complete per-epoch KLD history (n_real, n_epochs),
  * per-realization finals (kld, trained magic m2_final, trained entanglement
    s_half_final, convergence) and per-config scalars (target magic cost M(p),
    target entanglement s_half_target, entangler magic/entanglement of U_S|+>).
Crash-safe (saved per config), resumable (skips configs whose file exists).

SWEEPS (env SWEEP):
  decoupling : entangler ladder x L, headline 1D 5-peak target.   -> KLD vs L
  kscan      : Clifford x K x L, headline target.                 -> KLD vs K (per L)
  richness   : ladder x (n_peaks, sigma) x L, 1D targets.         -> difficulty vs modes/sharpness
  targetmagic: {Clifford,Haar} x T-doped targets (n_t).           -> M(p) floor, decoupling@high magic
  nt         : doped-Clifford x n_t, fixed K,L, headline target.  -> magic doping curve

Env: SWEEP, GRID_N (comma N list). e.g.  JAX_PLATFORMS=cpu SWEEP=decoupling GRID_N=10 python3 run_sweep.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax

# Persistent, cross-process compilation cache: identical (N,L) circuit shapes (the 4
# entanglers at each depth, and every worker) reuse XLA executables from disk instead of
# each recompiling. Combined with the scan-over-layers forward this makes compilation cheap.
_CACHE = os.environ.get("JAX_COMPILE_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                          ".jax_cache"))
jax.config.update("jax_compilation_cache_dir", _CACHE)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

import jax.numpy as jnp
import numpy as np

from utils.cliffords import (load_clifford_group, sample_scrambler_gates, doped_clifford_2q_gates)
from utils.born_machine import (brickwork_pairs, make_forward, plus_state, zero_state,
                                apply_scrambler, kld_from_probs, layer_states)
from utils.targets import make_target, magic_cost_upper
from utils.sre import stabilizer_renyi_entropy, half_chain_entropy, page_entropy
from utils.train import make_trainer
from utils.store import save_run, run_filename, runs_dir

# ---- fixed protocol / hyperparameters (stored in every file's meta) ----
N_REAL   = int(os.environ.get("N_REAL", "20"))       # warm-start curriculum uses fewer (converges fast)
N_EPOCHS = int(os.environ.get("N_EPOCHS", "1000"))   # 2D sweeps use more epochs (deeper/harder targets)
LR, CLIP = 0.02, 1.0
ALPHA    = 0.1           # cosine-decay floor: LR ends at ALPHA*LR = 2e-3 (A/B: floor barely matters,
#                          plateau is expressivity not LR-starvation; 0.1 keeps the tail mildly active)
SEED     = 42            # target weight seed
BASE     = 12000         # base PRNG seed for entanglers/inits
_LN2 = float(np.log(2.0))
HEADLINE = dict(target="mm", n_peaks=5, sigma_frac=0.05)   # 1D 5-peak, wider weights (targets.py default)


def build_target(cfg, N, group, pairs, ng):
    """Return (p vector, meta-extra). 'mm' = 1D multimodal; 'tdoped' = doped-Clifford Born dist."""
    if cfg["target"] == "mm":
        p = jnp.asarray(make_target("multimodal_1d", N, n_peaks=cfg["n_peaks"],
                                    sigma_frac=cfg["sigma_frac"], seed=SEED))
        return p
    if cfg["target"] == "mm2d":
        # 2D four-lobe Gaussian mixture (mode-collapse benchmark); N split into two n=N/2
        # registers. Analogue of the QSBM 2D multimodal target, now through the SRE lens:
        # a genuinely 2D target learned by a zero-magic (Clifford) entangler + local rotations.
        p = jnp.asarray(make_target("multimodal_2d", N))
        return p
    if cfg["target"] == "moons":                          # two interleaving crescents, M(p)~2.2
        return jnp.asarray(make_target("two_moons", N))
    if cfg["target"] == "rings":                          # concentric rings, M(p)~1.9
        return jnp.asarray(make_target("rings", N))
    if cfg["target"] == "spiral":                         # two interleaving spirals (hardest)
        return jnp.asarray(make_target("two_spirals", N))
    if cfg["target"] == "stripes":                        # three diagonal ridges (clean, trainable)
        return jnp.asarray(make_target("three_stripes", N))
    if cfg["target"] == "cross":                          # two crossing diagonals (an X)
        return jnp.asarray(make_target("cross", N))
    if cfg["target"] == "tdoped":
        gen = doped_clifford_2q_gates(jax.random.PRNGKey(BASE + 900 + cfg["n_t_target"]), ng, group,
                                      cfg["n_t_target"])
        psi = apply_scrambler(zero_state(N), gen, pairs, N)
        return jnp.abs(psi) ** 2
    raise ValueError(cfg["target"])


def build_entangler(kind, N, K, n_t, group):
    pairs = brickwork_pairs(N, K)
    ng = len(pairs)
    keys = jax.random.split(jax.random.PRNGKey(BASE + 991 * {"matchgate": 3, "clifford": 1, "doped": 2,
                            "haar": 0, "cluster": 4, "ising": 5}[kind] + 100 * K + n_t), N_REAL)
    if kind == "doped":
        ent = jnp.stack([doped_clifford_2q_gates(keys[r], ng, group, n_t) for r in range(N_REAL)])
    else:
        ent = jnp.stack([sample_scrambler_gates(keys[r], kind, ng, group, 0) for r in range(N_REAL)])
    return ent, pairs


# SRE is O(4^N), so compute it only where needed.
_MP_CACHE = {}


def target_magic(p, key, N):
    """M(p)=M2(sqrt p) depends only on the TARGET; compute once, cache."""
    if key not in _MP_CACHE:
        _MP_CACHE[key] = float(magic_cost_upper(np.asarray(p), N))
    return _MP_CACHE[key]


def sre_budget(N):
    """Number of realizations on which to evaluate SRE (the expensive part)."""
    return N_REAL if N <= 12 else (8 if N == 13 else 4)


def layer_budget(N):
    """How many TRAINED realizations to profile layer-by-layer (SvN+SRE after every layer).
    Each costs L+1 SRE calls, so keep it modest; enough seeds to show the spread."""
    return 8 if N <= 12 else (4 if N == 13 else 2)


def layer_profile(params, gates, pairs, N, L):
    """SRE (magic) and s_half (SvN, mid-cut) of the state AFTER each layer l=0..L, for ONE
    trained realization -> L+1 SRE calls. The per-layer resource build-up of a trained model.
    Uses born_machine.layer_states so the layer order matches the forward map exactly and the
    same trace is reproducible offline from the stored weights."""
    m2, sh = [], []
    for psi in layer_states(params, gates, pairs, N, L):
        m2.append(float(stabilizer_renyi_entropy(psi, N)))       # magic M_2: log2, by definition
        sh.append(float(half_chain_entropy(psi, N)))             # von Neumann SvN: natural log (raw value)
    return np.array(m2), np.array(sh)                             # each (L+1,)


# fields that identify a config; the filename must be injective on these (else -> silent overwrite)
_ID_FIELDS = ("sweep", "N", "K", "L", "entangler", "n_t", "target", "n_peaks", "sigma_frac", "n_t_target")


def _assert_existing_matches(path, meta):
    """Before skipping an existing file as already done, verify it holds this config. A different
    config mapping to the same filename is a name collision -> abort rather than treat it as done."""
    import json
    try:
        existing = json.loads(str(np.load(path, allow_pickle=True)["meta"]))
    except Exception:
        return                                          # legacy/unreadable file: don't block
    for k in _ID_FIELDS:
        if str(existing.get(k)) != str(meta.get(k)):
            raise SystemExit(f"FATAL: {os.path.basename(path)} already holds a DIFFERENT config "
                             f"({k}={existing.get(k)!r}, wanted {meta.get(k)!r}) -> filename collision. "
                             "run_filename is missing a field this sweep varies.")


def run_one(cfg, group, init_override=None):
    """Train+diagnose one config. init_override (n_real,L,3,N) supplies a warm-start init
    (depth curriculum); when None the rotation angles start from random, as before."""
    N, K, L = cfg["N"], cfg["K"], cfg["L"]
    meta = dict(sweep=cfg["sweep"], N=N, K=K, L=L, entangler=cfg["entangler"], n_t=cfg.get("n_t", 0),
                target=cfg["target"], n_peaks=cfg.get("n_peaks", "na"), sigma_frac=cfg.get("sigma_frac", "na"),
                n_t_target=cfg.get("n_t_target", 0), n_real=N_REAL, n_epochs=N_EPOCHS, lr=LR, lr_alpha=ALPHA,
                clip=CLIP, weight_seed=SEED, base_seed=BASE, n_params=int(3 * N * L),
                under_param=bool(3 * N * L < 2 ** N))
    path = os.path.join(runs_dir(), run_filename(meta))
    if os.path.exists(path):
        _assert_existing_matches(path, meta)   # existing file MUST be this same config, not a name clash
        print(f"  skip (exists) {os.path.basename(path)}", flush=True)
        return
    ent, pairs = build_entangler(cfg["entangler"], N, K, cfg.get("n_t", 0), group)
    ng = len(pairs)
    p = build_target(cfg, N, group, pairs, ng)
    H = float(-np.sum(np.asarray(p) * np.log(np.clip(np.asarray(p), 1e-12, None))))
    # per-config scalars (SRE only where needed; entanglement is cheap)
    tkey = (cfg["target"], cfg.get("n_peaks", 0), cfg.get("sigma_frac", 0), cfg.get("n_t_target", 0), N)
    Mp = target_magic(p, tkey, N)                                  # cached per target
    sT = float(half_chain_entropy(jnp.sqrt(p).astype(jnp.complex128), N))   # SvN, natural log (raw)
    scr = [apply_scrambler(plus_state(N), ent[r], pairs, N) for r in range(N_REAL)]
    sh_scr = float(np.mean([half_chain_entropy(s, N) for s in scr]))        # SvN, natural log (raw)
    if cfg["entangler"] in ("clifford", "cluster", "ising"):
        m2_scr = 0.0                                               # Clifford invariance: no SRE call
    else:
        nb = min(4, N_REAL)                                        # few realizations suffice for U_S magic
        m2_scr = float(np.mean([stabilizer_renyi_entropy(scr[i], N) for i in range(nb)]))

    forward = make_forward(N, L, pairs)
    trainer = make_trainer(forward, N_EPOCHS, LR, CLIP, alpha=ALPHA)
    if init_override is not None:
        init = jnp.asarray(init_override)                         # depth-curriculum warm start
        assert init.shape == (N_REAL, L, 3, N), f"warm init {init.shape} != {(N_REAL,L,3,N)}"
    else:
        init = jnp.stack([jax.random.uniform(jax.random.PRNGKey(BASE + 7 + 13 * L + r), (L, 3, N),
                          minval=0.0, maxval=2 * np.pi) for r in range(N_REAL)])
    params, losses = trainer(ent, init, p)
    kld_hist = np.asarray(losses) - H                              # (n_real, n_epochs)
    sb = sre_budget(N)
    kld_f, sh_f, conv, q_all, psis = [], [], [], [], []
    for r in range(N_REAL):
        probs, psi = forward(params[r], ent[r])
        kld_f.append(float(kld_from_probs(p, probs)))
        sh_f.append(float(half_chain_entropy(psi, N)))            # SvN (natural log, raw); cheap, all seeds
        c = kld_hist[r]
        conv.append(float(abs(c[-1] - c[int(0.9 * N_EPOCHS)]) / (abs(c[-1]) + 1e-12)))
        q_all.append(np.asarray(probs, dtype=np.float64))         # learned output distribution q_theta
        if r < sb:
            psis.append(psi)
    m2_f = [float(stabilizer_renyi_entropy(psis[r], N)) for r in range(len(psis))]   # magic of state psi_theta (gauge)
    mq_f = [float(magic_cost_upper(q_all[r], N)) for r in range(sb)]                 # magic COST of output q_theta (sampling-relevant)
    b = int(np.argmin(kld_f))
    # per-layer resource build-up (SvN + SRE after every layer) for several TRAINED realizations,
    # so the magic/entanglement-vs-layer trace of trained models is plottable with its seed spread.
    lr_n = min(N_REAL, layer_budget(N))
    prof = [layer_profile(np.asarray(params[r]), ent[r], pairs, N, L) for r in range(lr_n)]
    m2_lay = np.stack([pr[0] for pr in prof])                     # (lr_n, L+1)
    sh_lay = np.stack([pr[1] for pr in prof])                     # (lr_n, L+1)
    finals = dict(
        kld=np.array(kld_f), m2_final=np.array(m2_f), mq_final=np.array(mq_f),
        s_half_final=np.array(sh_f), conv_rel10=np.array(conv),
        magic_cost_Mp=Mp, s_half_target=sT, m2_scrambler=m2_scr, s_half_scrambler=sh_scr,
        page_svn=page_entropy(N), n_real_sre=len(m2_f), n_layer_real=lr_n, best_real=b,
        m2_by_layer=m2_lay, s_half_by_layer=sh_lay,
        # --- everything needed to reconstruct/reuse the trained circuits exactly ---
        params=np.asarray(params, dtype=np.float64),             # (n_real, L, 3, N) trained weights, layer-structured
        scrambler_gates=np.asarray(ent, dtype=np.complex128),    # (n_real, n_gates, 4, 4) U_S actually used, per realization
        pairs=np.asarray(pairs, dtype=np.int32),                 # (n_gates, 2) brickwork bonds gates[i] acts on
        target_p=np.asarray(p, dtype=np.float64),                # (2^N,) the target distribution (self-contained)
        probs_final=np.asarray(q_all, dtype=np.float64),         # (n_real, 2^N) learned output distributions q_theta
    )
    save_run(meta, kld_hist, finals)
    print(f"  saved {cfg['sweep']} N={N} K={K} L={L} {cfg['entangler']} nt={cfg.get('n_t',0)} "
          f"tgt={cfg['target']}(m{cfg.get('n_peaks','')},s{cfg.get('sigma_frac','')}) "
          f"KLD(med)={np.median(kld_f):.4f} M(p)={Mp:.2f}", flush=True)


# ---- config grids ----
LADDER = ("matchgate", "clifford", "doped", "haar")
MAX_L = 10                                                       # maximum circuit depth in the sweeps


def _l_grid(Lcap):
    """Systematic depth grid L = 1,2,4,6,8,...,MAX_L (regular, evenly spaced -> clean curves),
    capped at MAX_L and the under-param ceiling."""
    return tuple(L for L in [1, 2] + list(range(4, MAX_L + 1, 2)) if L <= Lcap)


def _k_grid(N):
    """Systematic entangler-depth grid K = 1,2,4,6,8,...,2N (regular -> clean curves)."""
    return tuple([1, 2] + list(range(4, 2 * N + 1, 2)))


def grid(sweep, N):
    K = 2 * N
    Lcap = int(2 ** N / (3 * N))                                 # under-param ceiling: 3NL < 2^N
    if sweep == "decoupling":
        Ls = _l_grid(Lcap)                                       # KLD vs L, systematic points
        return [dict(sweep=sweep, N=N, K=K, L=L, entangler=e, n_t=(4 if e == "doped" else 0), **HEADLINE)
                for L in Ls for e in LADDER]
    if sweep == "kscan":
        # scan entangler depth K across the FULL ladder: below saturation (K<~12 at N=10) Haar
        # entangles more than Clifford, so this is where entangler TYPE could still matter.
        Ks = _k_grid(N)                                          # K = 1,2,4,...,2N, systematic points
        Ls = tuple(L for L in (4, MAX_L) if L <= Lcap)           # one shallow, one deeper (under-param)
        return [dict(sweep=sweep, N=N, K=k, L=L, entangler=e, n_t=(4 if e == "doped" else 0), **HEADLINE)
                for k in Ks for L in Ls for e in LADDER]
    if sweep == "richness":
        cfgs = [(5, s) for s in (0.02, 0.03, 0.05, 0.07, 0.10)] + [(m, 0.05) for m in (1, 2, 3, 4, 6, 8)]
        cfgs = list(dict.fromkeys(cfgs))
        Ls = tuple(L for L in (4, MAX_L) if L <= Lcap)           # shallow / deeper (under-param)
        return [dict(sweep=sweep, N=N, K=K, L=L, entangler=e, n_t=(4 if e == "doped" else 0),
                     target="mm", n_peaks=m, sigma_frac=s)
                for (m, s) in cfgs for L in Ls for e in LADDER]
    if sweep == "targetmagic":
        L = min(Lcap, MAX_L)
        return [dict(sweep=sweep, N=N, K=K, L=L, entangler=e, n_t=0, target="tdoped", n_t_target=nt)
                for nt in (0, 1, 2, 3, 4, 6, 8, 12, 16) for e in ("clifford", "haar")]
    if sweep == "nt":
        L = min(Lcap, MAX_L)
        ng = len(brickwork_pairs(N, K))
        return [dict(sweep=sweep, N=N, K=K, L=L, entangler="doped", n_t=nt, **HEADLINE)
                for nt in (0, 1, 2, 4, 8, 16, 32, ng)]
    if sweep == "lk":
        # FULL 2D L x K grid at fixed N -> heatmaps of KLD / SvN / M2_final per entangler.
        # L extended to deep values so the KLD heatmap reaches convergence (~1e-2), not just the plateau.
        Ks = _k_grid(N)                                          # 1,2,4,...,2N
        Ls = tuple(l for l in [1, 2, 4, 6, 8, 10, 14, 18, 22, 26] if l <= Lcap)
        return [dict(sweep=sweep, N=N, K=k, L=l, entangler=e, n_t=(4 if e == "doped" else 0), **HEADLINE)
                for k in Ks for l in Ls for e in LADDER]
    if sweep == "deepl":
        # deep L at K=2N to CONVERGE KLD to ~1e-2..1e-3 (the L<=10 cap plateaus at ~1e-1).
        Ls = tuple(l for l in [12, 14, 16, 18, 20, 22, 24, 26] if l <= Lcap)
        return [dict(sweep=sweep, N=N, K=2 * N, L=l, entangler=e, n_t=(4 if e == "doped" else 0), **HEADLINE)
                for l in Ls for e in LADDER]
    if sweep == "twod":
        # Analog route: three 2D targets of increasing nonstabilizerness cost
        #   mm2d (4-lobe, M(p)~1.25), rings (M(p)~1.9), moons (M(p)~2.2).
        # Headline: the analog kicked-Ising entangler (zero SRE) matches Haar (max SRE) at
        # converged depth, with all nonstabilizerness supplied by the local rotations.
        Ls = tuple(l for l in [1, 2, 4, 6, 8, 10, 12] if l <= Lcap)
        ROSTER = ("ising", "cluster", "haar", "matchgate")
        TARGETS = ("mm2d", "rings", "moons")
        return [dict(sweep=sweep, N=N, K=2 * N, L=l, entangler=e, n_t=0, target=t)
                for t in TARGETS for l in Ls for e in ROSTER]
    if sweep == "sigdecoup":
        # decoupling (KLD vs L) at several target sharpnesses -> collapse holds at each sigma.
        Ls = tuple(l for l in [1, 2, 4, 6, 8, 10] if l <= Lcap)
        return [dict(sweep=sweep, N=N, K=2 * N, L=l, entangler=e, n_t=(4 if e == "doped" else 0),
                     target="mm", n_peaks=5, sigma_frac=s)
                for s in (0.02, 0.03, 0.07, 0.10) for l in Ls for e in LADDER]
    raise ValueError(sweep)


# ---- overnight plan: full ordered set of (sweep, N); headline decoupling first ----
OVERNIGHT_PLAN = [("decoupling", 10), ("decoupling", 12), ("decoupling", 8),
                  ("targetmagic", 8), ("targetmagic", 10), ("kscan", 10),
                  ("richness", 10), ("nt", 10)]

# phase-2: N=10 ONLY (N=12 too slow). deep-L convergence + full L x K grid for heatmaps.
PHASE2_PLAN = [("deepl", 10), ("lk", 10)]
# phase-3: N=12 deep-L convergence + sigma-resolved decoupling at N=10 (collapse across sharpness).
PHASE3_PLAN = [("deepl", 12), ("sigdecoup", 10)]
# N=14 decoupling point (one larger size for the collapse's N-scaling; exact-SRE ceiling).
N14_PLAN = [("decoupling", 14)]
# deep-L L-K heatmap at N=10 (extended lk grid; skip-existing runs only the new deep-L configs).
LKDEEP_PLAN = [("lk", 10)]
# 2D four-lobe target study (SRE-lens analogue of the QSBM 2D multimodal experiment).
TWOD_PLAN = [("twod", 10)]
PLANS = {"overnight": OVERNIGHT_PLAN, "phase2": PHASE2_PLAN, "phase3": PHASE3_PLAN, "n14": N14_PLAN,
         "lkdeep": LKDEEP_PLAN, "twod": TWOD_PLAN}


def plan_configs(plan):
    """Flatten a list of (sweep, N) into the full ordered config list."""
    out = []
    for sweep, N in plan:
        out.extend(grid(sweep, N))
    return out


def _cfg_filename(cfg):
    """The .npz filename a cfg will write to (same fields run_one puts in meta)."""
    return run_filename(dict(
        sweep=cfg["sweep"], N=cfg["N"], K=cfg["K"], L=cfg["L"], entangler=cfg["entangler"],
        n_t=cfg.get("n_t", 0), target=cfg["target"], n_peaks=cfg.get("n_peaks", "na"),
        sigma_frac=cfg.get("sigma_frac", "na"), n_t_target=cfg.get("n_t_target", 0)))


def assert_unique_filenames(cfgs):
    """Abort before any compute if two distinct configs map to the same file. A collision means
    run_filename is missing a field the sweep varies (this once corrupted the targetmagic sweep
    when n_t_target was omitted)."""
    seen = {}
    for c in cfgs:
        fn = _cfg_filename(c)
        if fn in seen and seen[fn] != c:
            raise SystemExit(f"FATAL: filename collision on {fn}\n  A: {seen[fn]}\n  B: {c}\n"
                             "run_filename is missing a varying field -> fix it before running.")
        seen[fn] = c


def main():
    group = load_clifford_group(verbose=False)
    # PARALLEL MODE: WORKER_ID/N_WORKERS shard the full OVERNIGHT_PLAN across processes.
    # Configs are interleaved (i % N_WORKERS == WORKER_ID) so every worker starts on the
    # headline decoupling configs and cheap/expensive work is balanced. Skip-existing makes
    # it fully resumable and race-safe (a config is only ever claimed by one worker).
    if os.environ.get("N_WORKERS"):
        wid = int(os.environ.get("WORKER_ID", "0"))
        nw = int(os.environ["N_WORKERS"])
        plan = PLANS.get(os.environ.get("PLAN_NAME", "overnight"), OVERNIGHT_PLAN)
        cfgs = plan_configs(plan)
        assert_unique_filenames(cfgs)                   # fail fast on any collision before compute
        # deepl first (deepest L first -> lowest KLD ASAP); everything else cheap (small N,L) first
        cfgs.sort(key=lambda c: (0, -c["L"]) if c["sweep"] == "deepl" else (1, c["N"], c["L"]))
        mine = [c for i, c in enumerate(cfgs) if i % nw == wid]
        print(f"[worker {wid}/{nw}] {len(mine)}/{len(cfgs)} configs -> {runs_dir()}", flush=True)
        for cfg in mine:
            run_one(cfg, group)
        print(f"DONE worker {wid}/{nw}", flush=True)
        return
    # LEGACY single-sweep mode (SWEEP + GRID_N).
    sweep = os.environ.get("SWEEP", "decoupling")
    Ns = [int(x) for x in os.environ.get("GRID_N", "10").split(",")]
    for N in Ns:
        cfgs = grid(sweep, N)
        assert_unique_filenames(cfgs)                   # fail fast on any collision before compute
        print(f"[SWEEP={sweep} N={N}] {len(cfgs)} configs -> {runs_dir()}", flush=True)
        for cfg in cfgs:
            run_one(cfg, group)
    print(f"DONE SWEEP={sweep} N={Ns}", flush=True)


if __name__ == "__main__":
    main()
