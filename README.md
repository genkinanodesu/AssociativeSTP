# AssociativeSTP

Companion code for the manuscript "Reshaping Neural Representation via
Associative, Presynaptic Short-Term Plasticity."

This repository contains the analysis notebooks, ring-network simulation code,
cached simulation fits, and PDF outputs used for the manuscript figures. It is
intended as a reproducibility archive for the paper rather than as a general
software package.

## Layout

- `figures/`: PDF outputs used in the manuscript.
- `notebooks/`: Jupyter notebooks for the weak-coupling calculations and
  supplementary analyses.
- `ring_simulation/`: exact-gradient ring-network simulation code, the
  manuscript config, cached `fit.npz` files, and regenerated ring-network PDFs.
- `requirements.txt`: direct Python dependencies used by the code.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install Jupyter separately if you want to run or edit the notebooks
interactively:

```bash
pip install jupyter
```

## Weak-Coupling Figures

The notebooks reproduce the analytic weak-coupling results and supplementary
STD response calculations. Run them from the repository root or from
`notebooks/`; generated PDFs are written to `figures/`.

```bash
jupyter nbconvert --execute --inplace notebooks/pre-post-components.ipynb
jupyter nbconvert --execute --inplace notebooks/optimal-profiles.ipynb
jupyter nbconvert --execute --inplace notebooks/STP-response.ipynb
jupyter nbconvert --execute --inplace notebooks/STP-phase-response.ipynb
jupyter nbconvert --execute --inplace notebooks/STP-phase-response-suppl.ipynb
```

## Ring-Network Figures

The cached fits for the manuscript config are included under
`ring_simulation/saved_runs/ring/`.

Regenerate the manuscript comparison figures without retraining:

```bash
cd ring_simulation
python compare_condition_runs.py cfgs/ring.json --reuse-only --paper
```

To rerun training from the config, omit `--reuse-only` or use:

```bash
cd ring_simulation
python run_from_cfg.py cfgs/ring.json --paper --force-retrain
python compare_condition_runs.py cfgs/ring.json --reuse-only --paper
```

Full retraining can be slow. The cached fits are sufficient for redrawing the
ring-network panels in the manuscript. The command above writes its outputs to
`ring_simulation/saved_runs/ring/paper/`; the corresponding copies used in the
figure map are included under `figures/`.

## Notes on Reproducibility

- All figures in this repository are generated from simulations or analytic
  calculations; no experimental data are included.
- The ring-network comparison uses cached optimized parameters in
  `ring_simulation/saved_runs/ring/*/fit.npz` by default. Recomputed Monte Carlo
  estimates may differ slightly from the cached PDFs unless the same seeds and
  sampling counts are used.
- The notebooks and scripts write outputs in place. If you want to preserve the
  included PDFs exactly, run commands in a copy of the repository.

## Figure Map

| Figure | PDF | Source |
| --- | --- | --- |
| Fig. 1 | `figures/pre-post-components.pdf` | `notebooks/pre-post-components.ipynb` |
| Fig. 2 | `figures/optimal-profiles.pdf` | `notebooks/optimal-profiles.ipynb` |
| Fig. 3 | `figures/optimal-profiles-awake-sleep.pdf` | `notebooks/optimal-profiles.ipynb` |
| Fig. 4 | `figures/condition_compare.pdf` | `ring_simulation/compare_condition_runs.py` |
| Fig. 5 | `figures/condition_compare_replay.pdf` | `ring_simulation/compare_condition_runs.py` |
| Fig. S1 | `figures/STP-response.pdf` | `notebooks/STP-response.ipynb` |
| Fig. S2 | `figures/STP-phase-response-f.pdf` | `notebooks/STP-phase-response.ipynb` |
| Fig. S3 | `figures/STP-phase-response-C.pdf` | `notebooks/STP-phase-response.ipynb` |
| Fig. S4 | `figures/STP-phase-small-r.pdf` | `notebooks/STP-phase-response-suppl.ipynb` |
| Fig. S5 | `figures/STP-phase-medium-r.pdf` | `notebooks/STP-phase-response-suppl.ipynb` |
| Fig. S6 | `figures/STP-phase-large-r.pdf` | `notebooks/STP-phase-response-suppl.ipynb` |
| Fig. S7 | `figures/STP-phase-multi-r.pdf` | `notebooks/STP-phase-response-suppl.ipynb` |
| Fig. S8 | `figures/STP-phase-large-delta-nu.pdf` | `notebooks/STP-phase-response-suppl.ipynb` |
| Fig. S9 | `figures/suppl-grad-U.pdf` | `notebooks/optimal-profiles.ipynb` |
| Fig. S10 | `figures/suppl-optimal-U-different-amp.pdf` | `notebooks/optimal-profiles.ipynb` |
| Fig. S11 | `figures/suppl-optimal-U-various-constraints.pdf` | `notebooks/optimal-profiles.ipynb` |
| Fig. S12 | `figures/condition_compare_asymmetry_metrics.pdf` | `ring_simulation/compare_condition_runs.py` |
| Fig. S13 | `figures/condition_compare_correlogram.pdf` | `ring_simulation/compare_condition_runs.py` |
