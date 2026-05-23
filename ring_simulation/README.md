# Ring Simulation

This directory contains the exact-gradient ring-network simulation code used for
the recurrent-network analyses in Figs. 4, 5, S12, and S13 of the manuscript.

The manuscript config is:

```text
cfgs/ring.json
```

Cached fits are included for the three conditions compared in the manuscript:

- `with_stp`: associative STP
- `pre_stp`: non-associative STP with fixed release probability profile
- `no_stp`: static synapses

Regenerate the manuscript comparison PDFs from cached fits:

```bash
python compare_condition_runs.py cfgs/ring.json --reuse-only --paper
```

The outputs are written to:

```text
saved_runs/ring/paper/
```

Use `run_from_cfg.py` or `run_ring.sh` only when you want to rerun the
optimization. The cached `fit.npz` files are enough to redraw the manuscript
comparison figures.

`run_from_cfg.py` expands `cfgs/ring.json` into the three conditions above and
writes each run under `saved_runs/ring/<condition>/`. `compare_condition_runs.py`
then loads those fits, estimates the comparison metrics, and writes the
manuscript panels under `saved_runs/ring/paper/` when `--paper` is supplied.
