# ABI-CDMs 中文说明

本仓库是论文 *No Single Model Fits All: Conflict Decision-Making Models Vary More Across Studies Than Across Tasks* 的正式代码与数据发布版，包含 DDM、DMC、SSP、DSTP 四个主要模型，以及补充材料使用的约束模型和 dRiftDM 对齐模型。Notebook 演示、临时测试和无关实验模型不纳入发布流程。

## 发布版结构

```text
data/                 9 个保留的原始 CSV
checkpoints/          预训练模型权重
figures/main/         论文主图 1–5（仅 PNG）
figures/supplement/   生成的补充图
nsbi_module/          可复用的推断与分析模块
results/intermediate/ 生成的中间结果
results/tables/       生成的结果表
scripts/              按执行顺序组织的核心流程
```

Python 脚本统一从仓库根目录推导路径，可以直接在仓库根目录运行，不再要求切换到各脚本子目录。

## 安装

```bash
conda env create -f environment.yml
conda activate abi-cdms
pip install -e .
```

R 分析还需要 `tidyverse`、`psych`、`GPArotation`、`brms`、`tidybayes`、`posterior`、`cmdstanr` 和 `svglite`。
Inkscape 仅用于可选的 SVG→PDF 导出，并作为 **Figure 4** PNG 渲染的最终后备工具。

## 数据范围

发布版保留 9 个数据文件：8 个横断研究，另加 `clayson2024.csv` 作为重测研究。`erb2023.csv` 不属于本次正式分析，已经从 ABI-CDMs 删除。

原始文件人数与清理后的任务级分析人数并不相同。论文横断分析汇总为 8 个研究、1,375 名参与者；C24 重测分析最终为 150 名参与者。详细定义见 `docs/DATA_SOURCES.md`。

## 复现流程

```bash
# 1. 数据预处理
python scripts/01_preprocessing/prepare_datasets.py

# 2. 可选：重新训练四个核心模型
python scripts/02_training/train_ddm.py
python scripts/02_training/train_dmc.py
python scripts/02_training/train_ssp.py
python scripts/02_training/train_dstp.py

# 补充材料约束模型与 dRiftDM 对齐模型
python scripts/02_training/train_dmc_fixed_shape.py
python scripts/02_training/train_ssp_fixed_ratio.py
python scripts/02_training/train_dstp_fixed_ratio.py
python scripts/02_training/train_driftdm_aligned_dmc.py
python scripts/02_training/train_driftdm_aligned_dmc_variable_start.py

# 3. 拟合与汇总
python scripts/03_fitting/fit_core_models.py
python scripts/03_fitting/fit_extended_dmc.py
python scripts/03_fitting/summarize_core_fits.py
python scripts/03_fitting/summarize_extended_dmc.py

# 4. 主图
python scripts/05_model_comparison/figure_02_model_comparison.py
python scripts/06_ppc/generate_ppc_data.py
python scripts/06_ppc/figure_03_posterior_predictive_checks.py
python scripts/07_parameter_analysis/figure_04_latent_factors.py
python scripts/07_parameter_analysis/figure_05_factor_space.py
```

当前 Zenodo 权重包包含四个主要模型和六参数的 dRiftDM 对齐 DMC。**Figure S7** 所用的三个约束模型，以及当前 **Figure S8** 流程所用的七参数可变起始点 DMC，尚未包含在该权重包中，需要用上述入口重新训练。六参数与七参数 dRiftDM 对齐模型使用不同的模型注册名和 checkpoint 目录，不能混用。

补充模型恢复与 DMC–DSTP 比较可分别运行：

```bash
python scripts/04_validation/figure_s07_reduced_model_recovery.py
python scripts/08_supplementary/figure_s09_dstp_vs_dmc.py
```

因子图生成前，需要先渲染 `estimate_factor_scores.Rmd` 并运行 `fit_reliability_models.R`。中间数据统一写入 `results/intermediate/`，结果表写入 `results/tables/`。

## 主图映射

| 图 | 生成脚本 | 正式输出 |
|---|---|---|
| **Figure 1** | 设计图，无分析生成脚本 | `figure_01_workflow.png` |
| **Figure 2** | `figure_02_model_comparison.py` | `figure_02_model_comparison.png` |
| **Figure 3** | `figure_03_posterior_predictive_checks.py` | `figure_03_posterior_predictive_checks.png` |
| **Figure 4** | `figure_04_latent_factors.py` | `figure_04_latent_factors.png` |
| **Figure 5** | `figure_05_factor_space.py` | `figure_05_factor_space.png` |

预训练权重存档：**[10.5281/zenodo.21623907](https://doi.org/10.5281/zenodo.21623907)**。

代码采用 AGPL-3.0 许可证。英文完整说明与引用格式见 `README.md`。
