# ABI-CDMs: Amortized Bayesian Inference of Conflict Diffusion Models

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **Paper**: *No Single Model Fits All: Conflict Decision-Making Models Vary More Across Datasets Than Across Tasks*
>
> Wanke Pan, Jiashun Wang, Klaus Oberauer, Hu Chuan-Peng
>
> School of Psychology, Nanjing Normal University; LMU Munich; University of Zurich

This repository contains all code and data to reproduce the analyses in the paper. It implements **amortized Bayesian inference** (also called neural simulation-based inference, NSBI) for four cognitive process models of conflict decision-making: the **Diffusion Decision Model (DDM)**, the **Diffusion Model for Conflict (DMC)**, the **Shrinking Spotlight (SSP)**, and the **Dual-Stage Two-Phase (DSTP)** model.

---

## Repository Structure

```
ABI-CDMs/
├── README.md                          # This file
├── README_zh.md                       # Chinese version
├── LICENSE                            # AGPL v3
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment (recommended)
├── setup.py                           # pip install -e .
├── .gitignore
├── nsbi_module/                       # Core library
│   ├── __init__.py                    # Package exports
│   ├── NSBI_CDMs.py                   # Main model class
│   ├── trainer.py                     # NSBI training infrastructure
│   ├── simulators.py                  # Model simulators (DDM/DMC/SSP/DSTP)
│   ├── dists.py                       # Probability distributions
│   ├── default_settings.py            # Default parameters & config
│   ├── model_metrics.py               # RMSE, G², aBIC computation
│   ├── analysis_utils.py              # Analysis helper functions
│   ├── plotting.py                    # Publication-quality plotting
│   ├── utils.py                       # General utilities
│   ├── utils_preprocessing.py         # Data preprocessing utilities
│   ├── utils_ind_diff.py              # Individual difference utilities
│   ├── study_labels.py                # Dataset metadata & labels
│   ├── dmc_v2_loader.py               # DMC v2 model loader
│   ├── dmc_vs_loader.py               # DMC variant selector loader
│   └── tsdm_loader.py                 # TSDM model loader
├── scripts/                           # Analysis pipeline (in run order)
│   ├── 01_preprocessing/              # Data preprocessing
│   │   └── 21datasets_preprocessing.py
│   ├── 02_training/                   # Model training (NSBI)
│   │   ├── DDM_training.py            # DDM training
│   │   ├── DMC_training.py            # DMC training
│   │   ├── SSP_training.py            # SSP training
│   │   ├── DSTP_training.py           # DSTP training
│   │   └── ... (variant trainings)
│   ├── 03_fitting/                    # Model fitting & prediction
│   │   ├── 22fitting_and_predicting.py
│   │   ├── 22fitting_and_predicting_dmc_v2.py
│   │   ├── 23individual_analysis_preprocess.py
│   │   └── 23individual_analysis_preprocess_dmc_v2.py
│   ├── 04_validation/                 # Parameter & model recovery
│   │   ├── 11parameter_recovery.py
│   │   ├── 12parameter_mapping.py
│   │   └── 13model_recovery.py
│   ├── 05_model_comparison/           # Model comparison (manuscript Fig 2)
│   │   ├── 31prediciontion_comparison_RMSE.py   # Batch entry-point
│   │   ├── 31plot_percentage_RMSE.py            # Panel A/B
│   │   ├── 31plot_consistency_RMSE.py           # Panel C/D
│   │   ├── 31plot_retest_RMSE.py                # Panel E/F
│   │   ├── 32fig2_v8_combined.py                # Combined Fig 2
│   │   └── 32_model_metrics_comparison.py
│   ├── 06_ppc/                        # Posterior predictive checks (Fig 3)
│   │   ├── 24PPC.py                   # PPC computation
│   │   └── 24plot_ppc_fig3.py         # Combined Fig 3
│   ├── 07_parameter_analysis/         # EFA & reliability (manuscript Fig 4)
│   │   ├── 41parameter_analysis.py
│   │   ├── 43_factor_space_visualization.py
│   │   ├── 44fig4_efa_svg.py          # Final Fig 4 (SVG)
│   │   ├── 44fig4_v8_combined.py      # Combined Fig 4
│   │   ├── 43_EAF.Rmd                 # EFA (R)
│   │   ├── 44_factor_analysis.Rmd     # Factor analysis (R)
│   │   ├── 44_fitting_models.R        # Bayesian models (R)
│   │   ├── 44_viz_functions.R         # Visualization helpers (R)
│   │   ├── 45_parameter_consistency.Rmd # Parameter consistency (R)
│   │   └── 44_export_fig4_reliability_data.R
│   └── 08_supplementary/              # Supplementary materials
│       ├── 33_rmse_scaling_sensitivity.py      # RMSE scaling sensitivity
│       ├── 33_rmse_scaling_sensitivity_plot.py
│       ├── 33_ppc_component_metrics.py         # PPC component metrics
│       └── 33_model_metric_supplement.py       # Multi-criterion comparison
├── data/                              # Raw behavioral data (10 datasets)
│   ├── clayson2024.csv
│   ├── clayson2025.csv
│   ├── eisenberg2019.csv
│   ├── erb2023.csv
│   ├── hedge2018.csv
│   ├── kucina2023.csv
│   ├── lee2025.csv
│   ├── reymermet2018.csv
│   ├── ulrich2015.csv
│   └── whitehead2019.csv
├── checkpoints/                       # Model weights (download from OSF/Zenodo)
├── output/                            # Generated figures & results
└── docs/
    └── DATA_SOURCES.md                # Dataset citations & sources
```

---

## System Requirements

- **OS**: Windows, macOS, or Linux
- **Python**: 3.10–3.12
- **R**: 4.0+ (for factor analysis & Bayesian modeling)
- **GPU**: Optional — training benefits from CUDA-capable GPU, but inference works on CPU
- **Disk**: ~2 GB for checkpoints, ~50 MB for data

---

## Installation

### Option A: Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/Chuan-Peng-Lab/ABI-CDMs.git
cd ABI-CDMs

# Create and activate environment
conda env create -f environment.yml
conda activate nsbi-cdms

# Install the nsbi_module package in editable mode
pip install -e .
```

### Option B: pip + venv

```bash
git clone https://github.com/Chuan-Peng-Lab/ABI-CDMs.git
cd ABI-CDMs

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### R Dependencies

For the EFA and Bayesian modeling scripts (`scripts/07_parameter_analysis/*.Rmd`), install the following R packages:

```r
install.packages(c("tidyverse", "psych", "GPArotation", "corrplot"))
install.packages("pacman")
pacman::p_load(brms, tidybayes, posterior)
```

---

## Data

Raw behavioral data from 10 independent studies are included in `data/`. Each CSV contains trial-level response time and accuracy data for conflict tasks (Flanker, Simon, Stroop).

| Dataset | Task(s) | N | Citation |
|---------|---------|---|----------|
| `ulrich2015.csv` | Flanker, Simon | 40 | Ulrich et al. (2015) |
| `hedge2018.csv` | Flanker, Simon, Stroop | 48 | Hedge et al. (2018) |
| `whitehead2019.csv` | Flanker, Simon, Stroop | 40 | Whitehead et al. (2019) |
| `eisenberg2019.csv` | Stroop | 59 | Eisenberg et al. (2019) |
| `kucina2023.csv` | Flanker, Simon, Stroop | 52 | Kucina et al. (2023) |
| `erb2023.csv` | Flanker | 63 | Erb et al. (2023) |
| `clayson2024.csv` | Flanker, Simon, Stroop | 76 | Clayson et al. (2024) |
| `lee2025.csv` | Flanker, Simon | 315 | Lee et al. (2025) |
| `clayson2025.csv` | Flanker, Simon, Stroop | 606 | Clayson et al. (2025) |
| `reymermet2018.csv` | Stroop | 76 | Rey-Mermet et al. (2018) |

See `docs/DATA_SOURCES.md` for full citation details.

---

## Checkpoints

Pre-trained model weights are available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21623907.svg)](https://doi.org/10.5281/zenodo.21623907)

**DOI**: [10.5281/zenodo.21623907](https://doi.org/10.5281/zenodo.21623907)

Download `checkpoints_ABI-CDMs.zip` (93.6 MB) and extract into this repository's root directory. After extraction, `checkpoints/` should contain:
```
checkpoints/
├── DDM/          # DDM model weights (24 MB)
├── DMC/          # DMC model weights (28 MB)
├── SSP/          # SSP model weights (28 MB)
├── DSTP/         # DSTP model weights (28 MB)
└── driftdm_dmc/  # DMC v2 extended weights (18 MB)
```

---

## Reproducing Results

All scripts should be run **from the repository root directory** (`ABI-CDMs/`).

### Quick Start: Load Pre-trained Models

```python
from nsbi_module import NSBICDM

# Load a pre-trained DMC model
model = NSBICDM("DMC", checkpoint_path="checkpoints/DMC")

# Generate predictions for a subject
predictions = model.predict(data, n_samples=1000)
```

### Full Analysis Pipeline

Run scripts in the following order. Each step depends on outputs from previous steps.

#### Step 1: Preprocessing
```bash
python scripts/01_preprocessing/21datasets_preprocessing.py
```
*Input*: `data/*.csv`
*Output*: Preprocessed datasets (in memory / passed to fitting)

#### Step 2: Training (skip if using pre-trained checkpoints)
```bash
# Train all four models
python scripts/02_training/DDM_training.py
python scripts/02_training/DMC_training.py
python scripts/02_training/SSP_training.py
python scripts/02_training/DSTP_training.py
```
*Output*: Model checkpoints in `checkpoints/`

#### Step 3: Model Fitting
```bash
cd scripts/03_fitting
python 22fitting_and_predicting.py         # Main fitting
python 22fitting_and_predicting_dmc_v2.py  # DMC v2 variant
python 23individual_analysis_preprocess.py # Individual subject indices
cd ../..
```
*Output*: `23subj_indices_*.csv`, `23model_prediction_indices*.csv`

#### Step 4: Validation
```bash
python scripts/04_validation/11parameter_recovery.py
python scripts/04_validation/13model_recovery.py
```

#### Step 5: Model Comparison → **Manuscript Fig 2**
```bash
# Batch computation of all model comparison metrics
python scripts/05_model_comparison/31prediciontion_comparison_RMSE.py

# Generate combined Fig 2 (2x3 landscape layout)
python scripts/05_model_comparison/32fig2_v8_combined.py
```
*Output*: `output/fig2.svg`, `output/fig2.png`

#### Step 6: Posterior Predictive Checks → **Manuscript Fig 3**
```bash
python scripts/06_ppc/24PPC.py              # Compute PPC
python scripts/06_ppc/24plot_ppc_fig3.py    # Generate Fig 3
```
*Output*: `output/fig3.svg`, `output/fig3.png`

#### Step 7: Parameter Analysis → **Manuscript Fig 4**
```bash
cd scripts/07_parameter_analysis

# EFA and factor score extraction (Python)
python 41parameter_analysis.py

# Bayesian cross-task consistency models (R)
Rscript 44_fitting_models.R

# Factor space visualization
python 43_factor_space_visualization.py

# Final Fig 4 (native SVG)
python 44fig4_efa_svg.py

cd ../..
```
*Output*: `output/fig4.svg`, `output/fig4.png`

#### Step 8: Supplementary Materials
```bash
# RMSE scaling sensitivity analysis
python scripts/08_supplementary/33_rmse_scaling_sensitivity.py
python scripts/08_supplementary/33_rmse_scaling_sensitivity_plot.py

# PPC component-level metrics
python scripts/08_supplementary/33_ppc_component_metrics.py

# Multi-criterion model comparison (RMSE, G², aBIC)
python scripts/08_supplementary/33_model_metric_supplement.py
```
*Output*: Supplementary CSV tables and figures

---

## Manuscript Figure Mapping

| Figure | Description | Script | Output |
|--------|-------------|--------|--------|
| **Fig 1** | Task & model overview schematic | PPT (manual) | — |
| **Fig 2** | Model comparison, cross-task consistency, retest consistency | `scripts/05_model_comparison/32fig2_v8_combined.py` | `output/fig2.{svg,png}` |
| **Fig 3** | Posterior predictive checks (CAF + Delta) | `scripts/06_ppc/24plot_ppc_fig3.py` | `output/fig3.{svg,png}` |
| **Fig 4** | EFA, reliability, and factor space | `scripts/07_parameter_analysis/44fig4_efa_svg.py` | `output/fig4.{svg,png}` |
| **S1** | RMSE scaling sensitivity | `scripts/08_supplementary/33_rmse_scaling_sensitivity_plot.py` | `output/S_rmse_scaling_sensitivity.{svg,png}` |
| **S2** | PPC component metrics | `scripts/08_supplementary/33_ppc_component_metrics.py` | `output/S_ppc_component_metrics.{svg,png}` |
| **S3** | Multi-criterion model comparison | `scripts/08_supplementary/33_model_metric_supplement.py` | `output/S_model_metrics.{svg,png}` |

---

## License

This project is licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0). See `LICENSE` for details.

---

## Citation

If you use this code or data in your research, please cite:

```
Pan, W., Wang, J., Oberauer, K., & Hu, C.-P. (2026).
No Single Model Fits All: Conflict Decision-Making Models
Vary More Across Datasets Than Across Tasks.
[Journal/Preprint info TBD]
```

---

## Contact

**Wanke Pan** — [panwanke2023@gmail.com](mailto:panwanke2023@gmail.com)

GitHub: [https://github.com/Chuan-Peng-Lab/ABI-CDMs](https://github.com/Chuan-Peng-Lab/ABI-CDMs)
