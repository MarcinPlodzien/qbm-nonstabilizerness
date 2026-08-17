# Role of nonstabilizerness for quantum Born-machine generative modeling

Code and figure-generation scripts for the paper

> **Role of nonstabilizerness for quantum Born-machine generative modeling**
> M. Płodzień

A quantum circuit Born machine (the *quantum scrambling Born machine*, QSBM) fixes the entangling
unitary `U_S` and reuses it in every layer, training only the single-qubit rotations. This code separates the
entangler's two candidate resources, the **entanglement** it generates and its **nonstabilizerness**
(measured by the stabilizer Rényi entropy `M_2`), and finds the nonstabilizerness irrelevant. Four entangler
families span zero to near-maximal nonstabilizerness: Clifford (3-design), doped Clifford (tunable), local Haar
(near-maximal), and a classically simulable free-fermion matchgate (sub-two-design). The converged loss is the
same across all four, and for a Clifford entangler all of the trained state's nonstabilizerness comes from the
rotations. What the entangler must supply is volume-law entanglement, which every family reaches; a slower
entangler simply needs more depth. The code also computes a target's intrinsic nonstabilizerness through the
identity `M_2,min(p) = M_2(|sqrt p>)`. Targets: 1D Gaussian mixtures, Sherrington-Kirkpatrick spin-glass Gibbs
states, and 2D targets with a Clifford-point kicked-Ising entangler.

Companion paper (prior QSBM work): DOI [10.1103/qccv-flmq](https://doi.org/10.1103/qccv-flmq).

---

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.13 and JAX 0.11 (CPU). All computations run on CPU statevectors in double precision;
set `JAX_PLATFORMS=cpu` (the helper scripts already do).

## Repository layout

```
python_codes/
  utils/                    core library (Born machine, SRE, Clifford/matchgate gates, targets, training)
  spin_glass.py             Sherrington–Kirkpatrick / EA coupling and Gibbs-target construction
  figstyle.py, pra_style.py shared figure styling
  run_*.py                  data-generation scripts
  fig_*.py, make_fig_*.py   figure-generation scripts
  get_figure_N.py           reproduce Manuscript Figure N from stored data
  get_data_figure_N.py      regenerate the data behind Manuscript Figure N
  verify_phase_min.py       numerical check that M_2,min(p) = M_2(|sqrt p>)
  tests/                    unit tests (e.g. SRE against a brute-force reference)
figures/                    manuscript figures (PDF)
data/runs/                  per-configuration data records (not tracked; see "Data")
```

## Reproducing the figures

Each manuscript figure has a dedicated entry point. `get_figure_N.py` builds the figure from stored data;
`get_data_figure_N.py` regenerates that data first.

```bash
cd python_codes
python get_figure_1.py       # build Figure 1 from data/runs/
python get_data_figure_1.py  # (re)generate the data for Figure 1
```

Figures 1–5 are in the main text; Figure 6 is in Appendix B.

| Fig | Content | Build | Regenerate data | Data source |
|----:|---------|-------|-----------------|-------------|
| 1 | 1D five-peak: outputs and KL divergence vs depth | `get_figure_1.py` | `get_data_figure_1.py` | `run_sweep.py` (decoupling, deepl) |
| 2 | Doping curve; trained M₂ vs N | `get_figure_2.py` | `get_data_figure_2.py` | `run_sweep.py` (nt, deepl) |
| 3 | SK-Gibbs entangler comparison vs temperature | `get_figure_3.py` | `get_data_figure_3.py` | `run_gibbs.py` |
| 4 | 2D targets and trained outputs (kicked-Ising entangler) | `get_figure_4.py` | `get_data_figure_4.py` | `run_twod_curriculum.py` |
| 5 | 2D training convergence (KLD vs epoch) | `get_figure_5.py` | `get_data_figure_5.py` | `run_twod_curriculum.py` |
| 6 | Single-layer entangling power S_vN vs N (Appendix B) | `get_figure_6.py` | `get_data_figure_6.py` | `run_svn_scaling.py` |

## Data

Each data-generation script writes self-describing per-configuration records to `data/runs/`, with every
parameter encoded in both the filename and the stored metadata (system size, depth, entangler family, seed,
temperature, and so on). The SK-Gibbs records additionally store the coupling matrix, the entangler gates, all
trained angles, the loss trajectory, and the target distribution, so each datapoint is reproducible from the
file alone. The full data set is large and is not tracked in git; regenerate it with the `get_data_figure_*`
/ `run_*` scripts, or obtain the archived records from the author.

## Verification

```bash
cd python_codes
python -m pytest tests/            # unit tests (SRE against brute force, etc.)
python verify_phase_min.py         # phi=0 optimality of M_2,min on the SK targets
```

## Citation

If you use this code, please cite the accompanying paper. Machine-readable metadata is in
[`CITATION.cff`](CITATION.cff); GitHub renders a "Cite this repository" button from it.

```bibtex
@article{plodzien_qsbm_nonstabilizerness,
  title  = {Role of nonstabilizerness for quantum Born-machine generative modeling},
  author = {P{\l}odzie{\'n}, Marcin},
  year   = {2026}
}
```

## License

MIT license. See [LICENSE](LICENSE).
