# ABI-CDMs

Amortized Bayesian inference for conflict diffusion models, accompanying the manuscript *No Single Model Fits All: Conflict Decision-Making Models Vary More Across Studies Than Across Tasks*.

The repository contains the release-ready pipeline for four primary models—DDM, DMC, SSP, and DSTP—plus the constrained and dRiftDM-aligned specifications reported in the supplementary material. Notebook demonstrations, ad hoc tests, and unrelated experimental models are excluded.

## Repository layout

```text
ABI-CDMs/
├── data/                       # Nine retained raw CSV files
├── checkpoints/                # Downloaded pretrained estimators
├── figures/
│   ├── main/                   # Canonical Figures 1–5 (PNG only)
│   └── supplement/             # Generated supplementary figures
├── nsbi_module/                # Reusable inference and analysis library
├── results/
│   ├── intermediate/           # Generated HDF5, pickle, CSV, and RDS files
│   └── tables/                 # Generated result tables
├── scripts/                    # Pipeline in execution order
├── docs/DATA_SOURCES.md        # Raw and analysis sample documentation
└── SOURCE_PROVENANCE.md        # Mapping to the development repository
```

All Python paths are derived from the repository root. Run the commands below from the repository root; changing into individual script directories is not required.

## Installation

Python 3.10–3.12 and R 4.0 or newer are recommended. A GPU is optional for inference but strongly recommended for training.
Inkscape is required only for the optional SVG-to-PDF export and serves as the final PNG-rendering fallback for **Figure 4**.

```bash
conda env create -f environment.yml
conda activate abi-cdms
pip install -e .
```

For the factor and reliability analyses, install the R packages used by the files in `scripts/07_parameter_analysis/`, including `tidyverse`, `psych`, `GPArotation`, `brms`, `tidybayes`, `posterior`, `cmdstanr`, and `svglite`.

## Data scope

The release contains nine raw CSV files:

- eight studies used in the cross-sectional analysis;
- Raw-file participant counts differ from final analysis counts after task selection, session selection, and quality filtering. See [DATA_SOURCES.md](docs/DATA_SOURCES.md) for the authoritative distinction.

## Pretrained checkpoints

Download the archived checkpoints from [Zenodo](https://doi.org/10.5281/zenodo.21623907), then extract them into `checkpoints/`:

```text
checkpoints/
├── DDM/
├── DMC/
├── SSP/
├── DSTP/
└── driftdm_dmc/
```

This archive covers the four primary models and the six-parameter dRiftDM-aligned DMC. The reduced DMC, SSP, and DSTP checkpoints used for **Figure S7**, and the seven-parameter variable-start DMC used by the current **Figure S8** pipeline, are not in the current archive; use the training commands below to rebuild them.

## Reproduce the analysis

### 1. Preprocess raw data

```bash
python scripts/01_preprocessing/prepare_datasets.py
```

Creates `results/intermediate/datasets_cross_sectional.h5` and `datasets_retest.h5`.

### 2. Train estimators (optional)

Skip this step when using the pretrained checkpoints.

```bash
python scripts/02_training/train_ddm.py
python scripts/02_training/train_dmc.py
python scripts/02_training/train_ssp.py
python scripts/02_training/train_dstp.py
```

The supplementary constrained specifications have separate, semantically named entry points:

```bash
python scripts/02_training/train_dmc_fixed_shape.py
python scripts/02_training/train_ssp_fixed_ratio.py
python scripts/02_training/train_dstp_fixed_ratio.py
python scripts/02_training/train_driftdm_aligned_dmc.py
python scripts/02_training/train_driftdm_aligned_dmc_variable_start.py
```

The first three reduce weakly identifiable parameter combinations. The six-parameter dRiftDM-aligned model fixes the automatic-activation shape and centers the starting point; the seven-parameter version additionally estimates symmetric starting-point variability. They are intentionally kept as distinct model registrations and checkpoint directories.

### 3. Fit models and summarize predictions

```bash
python scripts/03_fitting/fit_core_models.py
python scripts/03_fitting/fit_extended_dmc.py
python scripts/03_fitting/summarize_core_fits.py
python scripts/03_fitting/summarize_extended_dmc.py
```

### 4. Run validation analyses

```bash
python scripts/04_validation/figure_s01_s03_parameter_recovery.py
python scripts/04_validation/figure_s01_s04_model_recovery.py
python scripts/04_validation/figure_s10_parameter_mapping.py
python scripts/04_validation/figure_s07_reduced_model_recovery.py
```

### 5. Generate model-comparison and PPC figures

```bash
python scripts/05_model_comparison/figure_02_model_comparison.py
python scripts/06_ppc/generate_ppc_data.py
python scripts/06_ppc/figure_03_posterior_predictive_checks.py
python scripts/06_ppc/figure_s05_caf.py
```

### 6. Estimate factors and generate reliability figures

Render `scripts/07_parameter_analysis/estimate_factor_scores.Rmd`, then run the reliability models before generating the final figures:

```bash
Rscript scripts/07_parameter_analysis/fit_reliability_models.R
python scripts/07_parameter_analysis/figure_04_latent_factors.py
python scripts/07_parameter_analysis/figure_05_factor_space.py
Rscript scripts/07_parameter_analysis/figure_s15_retest_icc.R
python scripts/07_parameter_analysis/figure_s16_representational_similarity.py
```

The Bayesian reliability step is computationally expensive and caches its models in `results/intermediate/`.

### 7. Generate robustness figures

```bash
python scripts/08_supplementary/figure_s17_rmse_scaling.py
python scripts/08_supplementary/figure_s18_ppc_component_metrics.py
python scripts/08_supplementary/figure_s19_model_metric_comparison.py
python scripts/08_supplementary/figure_s09_dstp_vs_dmc.py
```

## Canonical manuscript figures

| Figure | Generator | Published files |
|---|---|---|
| **Figure 1** | Design asset; no analysis generator | `figures/main/figure_01_workflow.png` |
| **Figure 2** | `scripts/05_model_comparison/figure_02_model_comparison.py` | `figure_02_model_comparison.png` |
| **Figure 3** | `scripts/06_ppc/figure_03_posterior_predictive_checks.py` | `figure_03_posterior_predictive_checks.png` |
| **Figure 4** | `scripts/07_parameter_analysis/figure_04_latent_factors.py` | `figure_04_latent_factors.png` |
| **Figure 5** | `scripts/07_parameter_analysis/figure_05_factor_space.py` | `figure_05_factor_space.png` |

To export the five PNG masters as PDFs:

```bash
python scripts/09_export/export_main_figure_pdfs.py
```

## License and citation

The code is licensed under AGPL-3.0; see `LICENSE`. If you use this release, cite the accompanying paper and the archived software record:

**Pan, W., Wang, J., Oberauer, K., & Hu, C.-P. (2026). *No Single Model Fits All: Conflict Decision-Making Models Vary More Across Studies Than Across Tasks*.**

Checkpoint archive: **[10.5281/zenodo.21623907](https://doi.org/10.5281/zenodo.21623907)**.
