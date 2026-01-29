# One-Dimensional Counterflow Combustion Study (Partially Cracked Ammonia)

This repository contains scripts, notebooks, and data for a one-dimensional numerical study of non-premixed counterflow diffusion flames with partially cracked ammonia. The workflow explores stability/extinction behavior, NOx rate-of-production (ROP) diagnostics, and machine-learning-ready datasets generated from parametric sweeps.

## Repository structure

- `mechanisms/` – Cantera mechanism files used by the simulations.
- `scripts/` – Python scripts for stability maps, ROP analysis, and post-processing.
  - `scripts/no_emission/` – NO/NOx-focused scripts and sweeps.
- `notebooks/` – Jupyter notebooks for batch simulation and ML analysis.
- `data/` – Input/output tabular data (e.g., extinction datasets, surrogate data).
- `results/` – Generated figures, plots, and summary CSV/XLSX outputs.
  - `results/no_emission/` – NO/NOx sweeps and plots.

Example output files include `results/figures/*.png` and `results/no_emission/summary_results*.csv`.

## How to run (high-level)

Most scripts are standalone and expect Cantera plus standard scientific Python packages (NumPy, Pandas, Matplotlib). Typical usage:

```bash
python scripts/PCAwithMaxT.py
python scripts/ROPtest.py
python scripts/no_emission/ROPtestFinal.py
```

Notebooks in `notebooks/` can be run in Jupyter after confirming the `mechanisms/`, `data/`, and `results/` paths.

## Notes

- Outputs are written under `results/` and `data/` (no new dependencies were added here).
- Update any path constants in scripts if you relocate files beyond the structure above.
