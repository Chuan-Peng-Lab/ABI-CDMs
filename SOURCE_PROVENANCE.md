# ABI-CDMs source provenance and release map

ABI-CDMs is the public release repository. The development repository remains `52nsbi-cdms`; release synchronization is selective rather than a directory mirror.

| Field | Value |
|---|---|
| Development repository | `E:\05code\01academicProject\10nsbi-projects\52nsbi-cdms` |
| Source baseline used for this sync | `9ce01b2d48bc15d10bcf53dc03827bdf9144a1e1` |
| Release repository | `E:\05code\01academicProject\10nsbi-projects\ABI-CDMs` |
| Mapping updated | 2026-08-02 |

## Transformation contract

Files marked `adapted` are not safe to overwrite with a blind copy. ABI-CDMs applies these release transformations:

- qualified `nsbi_module.*` imports;
- repository-root-derived paths from `nsbi_module/project_paths.py`;
- semantic filenames aligned with manuscript figure numbers;
- generated artifacts routed to `results/` and `figures/`;
- notebook demonstrations, ad hoc tests, and models unrelated to the manuscript removed;
- English code identifiers, comments, docstrings, and errors retained in publication code.

## Core code map

| Development source | ABI-CDMs target | Status |
|---|---|---|
| `2_code_analysis/21datasets_preprocessing.py` | `scripts/01_preprocessing/prepare_datasets.py` | adapted |
| `1_code_training/DDM_training.ipynb` | `scripts/02_training/train_ddm.py` | reduced to core entry point |
| `1_code_training/DMC_training.ipynb` | `scripts/02_training/train_dmc.py` | reduced to core entry point |
| `1_code_training/SSP_training.ipynb` | `scripts/02_training/train_ssp.py` | reduced to core entry point |
| `1_code_training/DSTP_training.ipynb` | `scripts/02_training/train_dstp.py` | reduced to core entry point |
| `1_code_training/DMC_fix_alpha_training.ipynb` | `scripts/02_training/train_dmc_fixed_shape.py` | reduced to the **Figure S7** training entry point |
| `1_code_training/SSP_fix_sd_rate_training.ipynb` | `scripts/02_training/train_ssp_fixed_ratio.py` | reduced to the **Figure S7** training entry point |
| `1_code_training/DSTP_fix_sd_rate_training.ipynb` | `scripts/02_training/train_dstp_fixed_ratio.py` | reduced to the **Figure S7** training entry point |
| `1_code_training/drfitdm_DMC_fix_alpha_training.py` + `nsbi_module/dmc_v2_loader.py` | `scripts/02_training/train_driftdm_aligned_dmc.py` + `nsbi_module/dmc_v2_loader.py` | simulator centralized; six-parameter DMC_v2 |
| `1_code_training/drfitdm_DMC_vs_training.py` + `nsbi_module/dmc_vs_loader.py` | `scripts/02_training/train_driftdm_aligned_dmc_variable_start.py` + `nsbi_module/dmc_variable_start_loader.py` | renamed; seven-parameter **Figure S8** model |
| `2_code_analysis/22fitting_and_predicting.py` | `scripts/03_fitting/fit_core_models.py` | adapted |
| `2_code_analysis/22fitting_and_predicting_dmc_v2.py` | `scripts/03_fitting/fit_extended_dmc.py` | adapted |
| `2_code_analysis/23individual_analysis_preprocess.py` | `scripts/03_fitting/summarize_core_fits.py` | adapted |
| `2_code_analysis/23individual_analysis_preprocess_dmc_v2.py` | `scripts/03_fitting/summarize_extended_dmc.py` | adapted |
| `2_code_analysis/11parameter_recovery.py` | `scripts/04_validation/figure_s01_s03_parameter_recovery.py` | adapted |
| `2_code_analysis/13model_recovery.py` | `scripts/04_validation/figure_s01_s04_model_recovery.py` | adapted |
| `2_code_analysis/12parameter_mapping.py` | `scripts/04_validation/figure_s10_parameter_mapping.py` | adapted |
| Reduced-model training diagnostics | `scripts/04_validation/figure_s07_reduced_model_recovery.py` | canonical **Figure S7** diagnostic entry point |
| `2_code_analysis/32fig2_v8_combined.py` | `scripts/05_model_comparison/figure_02_model_comparison.py` | adapted |
| `2_code_analysis/24PPC.py` | `scripts/06_ppc/generate_ppc_data.py` | adapted |
| `2_code_analysis/24plot_ppc_fig3.py` | `scripts/06_ppc/figure_03_posterior_predictive_checks.py` | adapted |
| `2_code_analysis/24plot_ppc_S5_caf_twoblock.py` | `scripts/06_ppc/figure_s05_caf.py` | adapted |
| `2_code_analysis/43_EAF.Rmd` | `scripts/07_parameter_analysis/estimate_factor_scores.Rmd` | adapted |
| `2_code_analysis/44_fitting_models.R` | `scripts/07_parameter_analysis/fit_reliability_models.R` | adapted |
| `2_code_analysis/44_viz_functions.R` | `scripts/07_parameter_analysis/reliability_plot_helpers.R` | adapted |
| `2_code_analysis/44_export_fig4_reliability_data.R` | `scripts/07_parameter_analysis/export_figure_04_reliability_data.R` | adapted |
| `2_code_analysis/44fig4_efa_svg.py` + `44fig4_v8_combined.py` | private builders plus `figure_04_latent_factors.py` and `figure_05_factor_space.py` | split into canonical entry points |
| `2_code_analysis/45plot_retest_icc_S15.R` | `scripts/07_parameter_analysis/figure_s15_retest_icc.R` | adapted |
| `2_code_analysis/42task_difference_analysis.py` | `scripts/07_parameter_analysis/figure_s16_representational_similarity.py` | adapted |
| `2_code_analysis/33_*.py` | `scripts/08_supplementary/figure_s17_*.py` through `figure_s19_*.py` | adapted |
| `2_code_analysis/36_bayesflow_driftdm_z_comparison/36_dstp_dmcv2_rmse_comparison.py` | `scripts/08_supplementary/figure_s09_dstp_vs_dmc.py` | reduced to the canonical **Figure S9** panel |

The reusable files in `nsbi_module/` retain the same source names unless explicitly listed as excluded below. Local changes primarily qualify imports and centralize paths.

## Canonical figure map

| Development artifact | ABI-CDMs artifact |
|---|---|
| `figs/fig1/fig1_cdms_workflow.*` | `figures/main/figure_01_workflow.*` |
| `figs/32fig2_v8_combined_2x3_selectable.*` | `figures/main/figure_02_model_comparison.*` |
| `figs/Fig3_model_predictions.*` | `figures/main/figure_03_posterior_predictive_checks.*` |
| `figs/Fig4_latent_factors_selectable.*` | `figures/main/figure_04_latent_factors.*` |
| `figs/Fig5_factor_space_selectable.*` | `figures/main/figure_05_factor_space.*` |

## Data map

Nine raw CSV files are retained as content-identical release inputs: `clayson2024`, `clayson2025`, `eisenberg2019`, `hedge2018`, `kucina2023`, `lee2025`, `reymermet2018`, `ulrich2015`, and `whitehead2019`.

`erb2023.csv` is intentionally excluded from ABI-CDMs and must not be restored during future syncs.

## Intentionally excluded code

- the dRiftDM fitting half of the **Figure S8** comparison, which requires the
  external dRiftDM R package and posterior-predictive outputs; its ABI model
  definition and training entry point are retained here;
- TSDM, which is not used in the manuscript or supplementary material;
- standalone simulator demonstrations mislabeled as training scripts;
- exploratory cluster, histogram, Sankey, and one-off comparison scripts;
- files named as tests that are not part of the published validation set;
- manuscript-PDF overlay tools, which belong to the manuscript repository rather than ABI-CDMs.

## Future synchronization

1. Compare the current development HEAD with the baseline above.
2. Sync only mapped source files and reapply the transformation contract.
3. Never copy generated HDF5, pickle, RDS, CSV result, or figure files as source code.
4. Regenerate release artifacts and validate paths after each sync.
5. Update this baseline and the `release_exports` mapping in the development repository's `ara.yaml` together.
