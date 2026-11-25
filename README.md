# rapid-topology-optimization

Code for the joint optimization of node topology and per-path RAPID shape
factors in distributed fiber-optic photoacoustic NDT.

The repository contains the core source code and the data behind the reported
coverage numbers. Figure sources, rendered figures, run logs and the manuscript
are not included.

---

## 1. Layout

```
matlab/                GA implementation
  GA_r2_seed1.m        serial GA with fixed seed, writes one CSV line per generation
  GA_r2_par.m          same GA, parfor over the population evaluation
  fun_ahd.m            fitness (AHP-weighted coverage, blind / weak terms, RMSE)
  check_paper_cost.m   fitness of a given layout with the same function
  make_par_variant.py  derives the parallel variant from the serial script
  run_three_seeds.ps1  sequential three-seed driver
python/
  coverage/            coverage, path validity and clearance computations
  ga/                  clearance-constrained GA driver and result export
  montecarlo/          perturbation Monte Carlo
  experiment/          RAPID reconstruction from archived per-path DI values
data/
  ga_results/          GA outcomes (.npz), one file per run
  mc_trials/           Monte Carlo trials (.csv)
  layout/              optimized layout: coverage field, masks, nodes, betas
  reference/           reference-layout coverage field and masks, with README.txt
```

`GA_r2_*.m` keep their original names: the `r2` suffix is part of the file
names used by the driver scripts, not a tag added here.

## 2. Environment

Python 3.14 was used for the runs behind this code. The only third-party
requirement is `numpy` (2.5.3 in those runs); everything else comes from the
standard library. MATLAB is required for `matlab/`; `GA_r2_par.m` uses
`parfor` and therefore needs the Parallel Computing Toolbox.

## 3. Paths

The scripts were run inside a project working copy and contain absolute path
constants such as

```python
ROOT = Path(r"...\projects\<project>")
```

Script bodies are archived unchanged. To run them, point `ROOT` (and, where
present, `HERE`) at your local copy.

## 4. Scripts

| Script | Purpose |
|---|---|
| `coverage/matched_comparison.py` | Coverage of two layouts at matched settings: same 1 mm ROI grid, same line-of-sight rule, same path-count field, differing only in the β assignment |
| `coverage/r1_3_ablations.py` | Per-path β on a fixed reference layout, and the distribution over random feasible layouts |
| `coverage/clearance_sensitivity.py` | Coverage as a function of the minimum clearance required between a sensing path and the re-entrant corner |
| `coverage/cov_recheck_r420.py` | Coverage recomputed for one moved sensor, everything else unchanged |
| `coverage/verify_r2_geometry.py` | Read-only geometry audit: cut-out crossings and overlap, path-sample check, coverage with all paths vs valid paths, longest feasible path |
| `coverage/which_layout_made_fig10.py` | Tests alternative transducer/sensor indices against two published reconstruction positions |
| `ga/ga_clearance.py` | GA driver with a corner-clearance constraint; 6 transducers on fibre, 6 off-fibre sensors, 36 per-path β, AHP weights 0.648/0.230/0.122, 1 cm candidate grid, d1 = 50 mm, d2 = d3 = 10 mm, L_lim = 30 mm; 5 mm search grid, final individuals re-scored at 1 mm |
| `ga/verify_ga_result.py` | Node list, minimum corner clearance and 1 mm coverage of a GA outcome |
| `ga/export_new_layout.py` | Exports a GA outcome as coverage fields and masks (`x_mm`, `y_mm`, `P`, `mask_*`, `long`, `summary`, `nodes`, `betas`, `matched`) |
| `montecarlo/mc_recompute.py` | Perturbation Monte Carlo: transducers Gaussian along x, σ = 20/3 mm, truncated at 20 mm; sensors isotropic Gaussian, σ = 20/3 mm per axis; redraw on plate, cut-out or separation violation; coverage on the 1 mm grid |
| `montecarlo/mc_new_layout.py` | The same Monte Carlo applied to a given GA layout |
| `experiment/repro_experiment_rapid.py` | Reproduces the RAPID reconstruction of `L_RAPID_Location.m` on the optimized layout from the 36 per-path DI values carried in the script |

Usage of the argument-taking scripts:

```
python -I ga_clearance.py --clearance 0 --tag ctl --seed 1
python -I export_new_layout.py ga_ctl_seed1.npz outdir
python -I mc_new_layout.py ga_ctl_seed1.npz 1000
python -I mc_recompute.py 1000
```

`ga_clearance.py` writes `ga_<tag>_seed<k>.npz` with the node indices, the β
matrix and the fitness history; `export_new_layout.py` and
`mc_new_layout.py` read that file. `verify_ga_result.py` takes one or more
`ga_<tag>.npz` names.

## 5. Data

| Path | Content |
|---|---|
| `data/ga_results/ga_c0_seed{1,2,3}.npz` | GA outcomes, three seeds, `c0` setting (the suffix records the `--clearance` value in mm) |
| `data/ga_results/ga_c5_seed{1,2,3}.npz` | GA outcomes, three seeds, `c5` setting |
| `data/ga_results/ga_L_seed1{1..5}.npz` | GA outcomes used for the layout export and the Monte Carlo studies |
| `data/mc_trials/mc_trials_L_seed1{1..5}.csv` | 1000 perturbation trials for the corresponding layouts (1000 data rows each) |
| `data/mc_trials/mc_trials_baseline_beta102.csv` | 1000 trials for the reference layout at β = 1.02 |
| `data/mc_trials/mc_trials_optimized.csv` | 1000 trials for the optimized layout |
| `data/layout/` | `x_mm`, `y_mm`, `P` (path count), `mask_blind/weak/effective`, `nodes`, `betas`, `matched`, `summary` |
| `data/reference/` | Same fields for the reference layout at β = 1.01/1.02/1.03; grid and file description in `data/reference/README.txt` |

Each `.npz` holds `t` (transducer indices on the candidate grid), `s` (sensor
indices), `b` (6 × 6 per-path β) and `hist` (fitness history). The 301 × 301
coverage fields are 1 mm grid exports over x, y = 100..400 mm, with NaN outside
the L-shaped plate.

Not included: the rendered figures and their drawing scripts, the Origin export
of the measured waveforms, and the MATLAB run logs.

## 6. Notes

- The fitness function `fun_ahd.m` and the Python coverage code share the same
  line-of-sight rule: a path entering the region x > 300 mm and y > 300 mm of
  the L-shaped plate is invalid.
- `experiment/repro_experiment_rapid.py` carries the per-path DI values inline;
  they are a copy of the arrays archived in `L_RAPID_Location.m`, which is not
  part of this repository.
- No license file yet.
