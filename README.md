# Proximal Angle-Based Learning for Multiple Treatments

Code for the simulations and MIMIC-IV application reported in the manuscript and supplement.

## Installation

Use Python 3.12. Package versions are pinned in `requirements.txt`. The analysis environment used Linux and PyTorch 2.9.1 with CUDA 12.8.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For CUDA 12.8, install the matching PyTorch build:

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

Run all commands below from the repository root. Replace `cuda` with `cpu` for CPU execution. Each command uses one device; the reference GPU was an NVIDIA RTX 4090 with 24 GiB of memory. Full simulations can take several days, and application analyses can take hours, depending on hardware. Floating-point results may vary across environments.

## Simulations

Run in this order:

```bash
python run_simulation.py --outcome continuous --device cuda
python run_simulation.py --outcome survival --device cuda
python make_figures.py --part simulation
```

The commands run the continuous linear/nonlinear and linear Weibull AFT settings, each with `U0`/`U1` and training sizes 2,000/5,000. `U0` and `U1` denote the manuscript's settings without and with the unmeasured-confounder interaction. Each setting uses 100 replications and 500,000 test observations per replication; survival horizons are 0.75, 1.00, and 1.25.

The DGPs are in `pabl/dgp.py` and `pabl/dgp_survival.py`. Seeds are assigned in `run_simulation.py`: training seed `20260511 + 10000 * rep + 42` and test seed `20260511 + 10000 * rep + 999`. Simulation data are generated in memory.

To run a subset:

```bash
python run_simulation.py --outcome continuous --n 2000 --rep-start 0 --rep-stop 10 --device cuda
```

`--rep-stop` is exclusive. Completed replication files are skipped when resuming. Use the same environment for all replications in an output directory. The plotting command requires all 100 replications of every continuous and survival setting.

## MIMIC-IV application

The application requires authorized access to [MIMIC-IV 3.1](https://physionet.org/content/mimiciv/3.1/); patient data are not included. Complete the PhysioNet credentialing, required training, and data-use agreement, then load the official hospital tables into PostgreSQL under the `mimiciv_hosp` schema.

Use PostgreSQL 14 with `psql` available on your command path. Required tables are `admissions`, `patients`, `emar`, `prescriptions`, `microbiologyevents`, `labevents`, and `transfers`. Configure PostgreSQL authentication outside the repository. The database user needs table-read and temporary-table permissions.

```bash
python run_real_data.py --database mimiciv31 --device cuda
python make_figures.py --part real-data
```

Replace `mimiciv31` with your database name. `real_data/cohort.sql` reconstructs the cohort; `real_data/preprocessing.py` applies the preprocessing. The analysis checks the expected supported cohort of 11,944 subjects and uses five master folds with seed `20260813`.

The three outcomes are DAOH28, complete-follow-up 90-day survival, and semi-synthetic censored 90-day survival. To run one, add `--outcome daoh28`, `--outcome survival90`, or `--outcome censored` to the analysis command. The agreement figure requires all three analyses.

The semi-synthetic censoring mechanism is in `real_data/censoring.py`, with seed `20260820` and coverage probability 0.70 per treatment arm. Censoring times are uniform on [1, 20] days without coverage and [91, 121] days with coverage, yielding 3,373 censored subjects (28.24%). Only aggregate results are saved.

## Outputs and manuscript results

All outputs are written under `results/`.

| Result | Output |
| --- | --- |
| Simulation policy values | `simulation/<setting>/rep_000.csv` through `rep_099.csv`; combined tables in `figures/<setting>/values.csv` |
| Main continuous comparison | `figures/continuous_*_5000/boxplot_true_val.png` |
| Supplementary continuous comparison | `figures/continuous_*_2000/boxplot_true_val.png` |
| Main survival comparison | `figures/survival_linear_U*_5000/boxplot_true_val_t1.png` |
| Supplementary survival comparisons | Remaining survival panels under `figures/survival_linear_U*_<n>/`, with suffixes `t0p75`, `t1`, or `t1p25` |
| Application tables | `real_data/daoh28.csv`, `real_data/survival90.csv`, and `real_data/censored.csv` |
| Application agreement values and figure | `real_data/*_agreement.json` and `figures/blood_culture_recommendation_agreement.pdf` |

Simulation commands save policy values with setting and seed identifiers; plotting creates 20 individual panels. Application tables report policy values, standard errors, and treatment recommendation percentages. The agreement JSON files contain pairwise recommendation percentages used to draw the PDF. Manuscript typesetting is separate.

## Code organization

- `run_simulation.py`, `run_real_data.py`, and `make_figures.py`: execution entry points.
- `pabl/`: shared estimators, DGPs, simulation workflows, tuning, and plotting.
- `real_data/`: cohort SQL, preprocessing, censoring, and application estimators.
- `requirements.txt`: dependency versions; `.gitignore`: generated outputs, local data, environments, and credentials excluded from version control.
