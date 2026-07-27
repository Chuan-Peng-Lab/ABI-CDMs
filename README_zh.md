# ABI-CDMs: 冲突扩散模型的摊销贝叶斯推断

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **论文**: *No Single Model Fits All: Conflict Decision-Making Models Vary More Across Datasets Than Across Tasks*
>
> 潘晚坷, 王家顺, Klaus Oberauer, 胡传鹏
>
> 南京师范大学心理学院; 慕尼黑大学; 苏黎世大学

本仓库包含了论文中所有分析和图表生成的代码与数据。实现了对四种冲突决策认知过程模型的**摊销贝叶斯推断**（也称神经模拟推断，NSBI）：**扩散决策模型 (DDM)**、**冲突扩散模型 (DMC)**、**收缩聚光灯模型 (SSP)** 和 **双阶段两阶段模型 (DSTP)**。

---

## 仓库结构

```
ABI-CDMs/
├── README.md                          # 英文说明
├── README_zh.md                       # 本文件（中文说明）
├── LICENSE                            # AGPL v3 许可证
├── requirements.txt                   # Python 依赖
├── environment.yml                    # Conda 环境配置（推荐）
├── setup.py                           # 安装脚本
├── nsbi_module/                       # 核心库（可 pip install -e .）
│   ├── NSBI_CDMs.py                   # 主模型类
│   ├── trainer.py                     # 训练器
│   ├── simulators.py                  # 四种模型的模拟器
│   ├── model_metrics.py               # RMSE/G²/aBIC 计算
│   ├── plotting.py                    # 论文级绘图
│   └── ...                            # 更多工具模块
├── scripts/                           # 分析流程脚本
│   ├── 01_preprocessing/              # 数据预处理
│   ├── 02_training/                   # 模型训练（.py，由 .ipynb 转换）
│   ├── 03_fitting/                    # 模型拟合与预测
│   ├── 04_validation/                 # 参数恢复与模型恢复
│   ├── 05_model_comparison/           # 模型比较 → 论文 Fig 2
│   ├── 06_ppc/                        # 后验预测检查 → 论文 Fig 3
│   ├── 07_parameter_analysis/         # 因子分析与信度 → 论文 Fig 4
│   └── 08_supplementary/              # 补充材料
├── data/                              # 原始行为数据（10 个数据集）
├── checkpoints/                       # 模型权重（需从 OSF/Zenodo 下载）
├── output/                            # 生成的图表
└── docs/
    └── DATA_SOURCES.md                # 数据来源与引用
```

---

## 安装

### 方法 A: Conda（推荐）

```bash
git clone https://github.com/Chuan-Peng-Lab/ABI-CDMs.git
cd ABI-CDMs

conda env create -f environment.yml
conda activate nsbi-cdms

pip install -e .
```

### 方法 B: pip + venv

```bash
git clone https://github.com/Chuan-Peng-Lab/ABI-CDMs.git
cd ABI-CDMs

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### R 依赖

```r
install.packages(c("tidyverse", "psych", "GPArotation", "corrplot"))
install.packages("pacman")
pacman::p_load(brms, tidybayes, posterior)
```

---

## 模型权重下载

预训练模型权重托管在 Zenodo 上：

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21623907.svg)](https://doi.org/10.5281/zenodo.21623907)

**DOI**: [10.5281/zenodo.21623907](https://doi.org/10.5281/zenodo.21623907)

下载 `checkpoints_ABI-CDMs.zip`（93.6 MB）并解压到 `checkpoints/` 目录。解压后 `checkpoints/` 结构为：
```
checkpoints/
├── DDM/          # DDM 模型权重 (24 MB)
├── DMC/          # DMC 模型权重 (28 MB)
├── SSP/          # SSP 模型权重 (28 MB)
├── DSTP/         # DSTP 模型权重 (28 MB)
└── driftdm_dmc/  # DMC v2 扩展权重 (18 MB)
```

---

## 复现论文结果

脚本使用 CWD 相对路径。**每步需先 `cd` 到对应 `scripts/XX_*/` 子目录后运行**（Step 2 训练脚本也可从仓库根目录运行）。

### 完整流程

```bash
# Step 1: 数据预处理
cd scripts/01_preprocessing
python 21datasets_preprocessing.py
cd ../..

# Step 2: 模型训练（如使用预训练权重可跳过）
python scripts/02_training/DDM_training.py
python scripts/02_training/DMC_training.py
python scripts/02_training/SSP_training.py
python scripts/02_training/DSTP_training.py

# Step 3: 模型拟合
cd scripts/03_fitting
python 22fitting_and_predicting.py
python 22fitting_and_predicting_dmc_v2.py
python 23individual_analysis_preprocess.py
cd ../..

# Step 4: 验证
cd scripts/04_validation
python 11parameter_recovery.py
python 13model_recovery.py
cd ../..

# Step 5: 模型比较 → Fig 2
cd scripts/05_model_comparison
python 31prediciontion_comparison_RMSE.py
python 32fig2_v8_combined.py
cd ../..

# Step 6: 后验预测检查 → Fig 3
cd scripts/06_ppc
python 24PPC.py
python 24plot_ppc_fig3.py
cd ../..

# Step 7: 因子分析 → Fig 4
cd scripts/07_parameter_analysis
python 41parameter_analysis.py
Rscript 44_fitting_models.R
python 43_factor_space_visualization.py
python 44fig4_efa_svg.py
cd ../..

# Step 8: 补充材料
cd scripts/08_supplementary
python 33_rmse_scaling_sensitivity.py
python 33_rmse_scaling_sensitivity_plot.py
python 33_ppc_component_metrics.py
python 33_model_metric_supplement.py
cd ../..
```

---

## 论文图表对应关系

| 图表 | 说明 | 生成脚本 |
|------|------|---------|
| **Fig 2** | 模型比较、跨任务一致性、重测一致性 | `scripts/05_model_comparison/32fig2_v8_combined.py` |
| **Fig 3** | 后验预测检查 (CAF + Delta) | `scripts/06_ppc/24plot_ppc_fig3.py` |
| **Fig 4** | 探索性因子分析、信度、因子空间 | `scripts/07_parameter_analysis/44fig4_efa_svg.py` |

---

## 数据来源

10 个公开数据集，详见 `docs/DATA_SOURCES.md`。

---

## 引用

如使用本代码或数据，请引用：

```
Pan, W., Wang, J., Oberauer, K., & Hu, C.-P. (2026).
No Single Model Fits All: Conflict Decision-Making Models
Vary More Across Datasets Than Across Tasks.
[待补充期刊信息]
```

---

## 联系方式

**潘晚坷** — [panwanke2023@gmail.com](mailto:panwanke2023@gmail.com)

GitHub: [https://github.com/Chuan-Peng-Lab/ABI-CDMs](https://github.com/Chuan-Peng-Lab/ABI-CDMs)
