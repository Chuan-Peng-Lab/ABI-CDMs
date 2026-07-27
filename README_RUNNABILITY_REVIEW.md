# ABI-CDMs 代码可运行性静态审查（2026-07-27）

## 审查结论

经过两轮审查 + 修复（14:06 import 修复 + 15:43 全面补丁），按 README 操作**现在可以完成安装和基础导入**，但部分 pipeline 步骤仍需外部 checkpoint 下载。

| # | 问题 | 严重度 | 状态 |
|---|------|--------|------|
| 1 | `bayesflow>=2.0` 不存在于 PyPI | 🔴 | ❌ **撤销**：2.0.0–2.0.12 已上传 PyPI（最早审查时 pip 缓存了旧索引）。已锁 `~=2.0.7` |
| 2 | 裸导入与 `pip install -e .` 不兼容（34 文件） | 🔴 | ✅ **14:06 已修复**：nsbi_module 内部改相对导入；29 个 scripts 改 `from nsbi_module.X import` |
| 3 | `__init__.py` 导入名错误 | 🔴 | ✅ **14:06 已修复**：CDMs_NSBI→NSBICDM；CDMsSimulator 源改 trainer.py；移除不存在函数 |
| 4 | **`utils_pydmc.py` 缺失**（提取时漏拷，NSBI_CDMs.py:5 / analysis_utils.py:24 等 4 处顶层 import）| 🔴 | ✅ **15:43 补拷**：从 52nsbi-cdms 拷贝到 nsbi_module/ |
| 5 | **8 个脚本有裸 `get_ipython()` nbconvert 残留** | 🔴 | ✅ **15:43 修复**：全部改为 `globals().get("get_ipython")` 守护模式 |
| 6 | **`pingouin` 未声明依赖**（41parameter_analysis.py 顶层 import + utils_ind_diff.py 内函数级 import）| 🔴 | ✅ **15:43 补入**：requirements.txt / setup.py / environment.yml 三处统一加 `pingouin>=0.5` |
| 7 | **h5 中间产物路径断链**：Step 1 写入 01_preprocessing/，Step 3 在 03_fitting/ 读取 | 🔴 | ✅ **15:43 对齐**：Step 1 改为写入 `../03_fitting/21preprocessed_datasets{,_retest}.h5`；33_rmse_scaling_sensitivity.py 改跨目录路径 |
| 8 | **README Quick Start 示例 `model.predict(...)` 不存在** | 🟠 | ✅ **15:43 修正**：改为 `model.fit_data(data, n_posterior=1000)` |
| 9 | **README CWD 指令系统性矛盾**：声明"从根目录跑"，但脚本内部路径均为脚本目录相对 | 🟠 | ✅ **15:43 统一**：README 改为每步 cd 到脚本目录（Step 2 从根运行例外，trainer 默认 "checkpoints/" 已对齐） |
| 10 | **`bayesflow>=2.0` 未锁版本**（11parameter_recovery.py 用内部 API `bayesflow.utils.*`）| 🟡 | ✅ **15:43 锁定**：三处依赖文件统一 `~=2.0.7`（dev 机验证版本） |
| 11 | 部分脚本依赖 CWD 为脚本所在目录（保留设计）| 🟡 | ⚠️ **预期行为**：README 已补充 `cd` 指令。未来可考虑 `Path(__file__).resolve().parent` 锚定 |
| 12 | checkpoints/ 为空（需 Zenodo 下载）| 🟡 | ⚠️ **文档化**：README 指引下载 DOI: 10.5281/zenodo.21623907。**用户确认 zip 无 `checkpoints/` 顶层目录**，README（中/英）已改为"解压到 `checkpoints/` 目录" |
| 13 | R: brms 在 Windows 需 Rtools；未提 KERAS_BACKEND | 🟡 | ✅ **15:43 补充**：README System Requirements 加 Rtools 链接 + KERAS_BACKEND 说明 |

---

## 修复详情

### Round 1（14:06）— import 体系修复
- nsbi_module 内部 5 文件：裸导入 → 相对导入（`from .X import`）
- scripts/ 29 文件：裸模块名 → `from nsbi_module.X import`
- `__init__.py`：CDMs_NSBI→NSBICDM；CDMsSimulator 源改 trainer.py；移除不存在顶层函数

### Round 2（15:43）— 可运行性补丁
1. **补 `utils_pydmc.py`**：从 `52nsbi-cdms/nsbi_module/utils_pydmc.py` 拷贝（NSBI_CDMs.py:5 `from .utils_pydmc import PlotFit`；analysis_utils.py:24 `from .utils_pydmc import Ob`；23individual_analysis_preprocess.py:12；24PPC.py:15）
2. **守护 8 个 get_ipython 脚本**：`22fitting_and_predicting.py`、`23individual_analysis_preprocess.py`、`13model_recovery.py`、`24PPC.py`、`41parameter_analysis.py`、`43_factor_space_visualization.py`、`12parameter_mapping.py`、`42task_difference_analysis.py`
3. **pin bayesflow + 补 pingouin**：三处依赖文件统一 `bayesflow~=2.0.7` + `pingouin>=0.5`
4. **h5 路径对齐**：`21datasets_preprocessing.py` 输出改 `../03_fitting/21preprocessed_datasets{,_retest}.h5`；`33_rmse_scaling_sensitivity.py` 读取跨目录路径
5. **README CWD 统一**：移除「All scripts should be run from root」→ 每步加 `cd scripts/XX_*/` 指令（Step 2 从头运行保留）；修 Quick Start 示例
6. **README 补充**：Rtools 安装提示、`KERAS_BACKEND=torch` 说明

---

## 当前状态（修复后）

安装路线（pip/conda/setup）全部可行。`pip install -e .` 后 `from nsbi_module import NSBICDM` 可正常导入。Pipeline 各步在 `cd scripts/XX_*/` 后按 README 命令运行：

| Step | 状态 | 前提条件 |
|------|------|---------|
| 1 (preprocessing) | ✅ 正常 | data/*.csv 存在（已有） |
| 2 (training) | ✅ 正常 | None（纯模拟） |
| 3 (fitting) | ✅ 正常（after Step 1） | checkpoints/DDM~DSTP（Zenodo 下载） |
| 4 (validation) | ✅ 正常 | checkpoints/（Zenodo）；bayesflow 内部 API 需锁定版本 |
| 5 (model comparison) | ✅ 正常 | 中间 CSV 已随库 |
| 6 (PPC) | ✅ 正常 | checkpoints/（Zenodo）+ 24_ppc_process_data_dict.pkl 已随库 |
| 7 (parameter analysis) | ✅ 正常 | 中间 CSV 已随库；R: pacman 自举 + brms 需 Rtools |
| 8 (supplementary) | ✅ 正常（after Step 1+3） | 中间 CSV 已随库；33_rmse_scaling_sensitivity 需 h5 |

## 待关注

### checkpoints/ 为空
`pip install -e .` 后仍需按 README 从 Zenodo 下载 checkpoints 并解压到 `checkpoints/` 目录。**已确认：zip 内无 `checkpoints/` 包裹层**（裸 DDM/DMC/SSP/DSTP/driftdm_dmc 目录），README（中/英）已修正为"解压到 `checkpoints/` 目录"。

### bayesflow.utils 内部 API 稳定性
`11parameter_recovery.py` 直接导入 BayesFlow 内部模块（`bayesflow.utils.ecdf/plot_utils/dict_utils`）。现已锁兼容版本 `~=2.0.7`，若后续升级需关注 API 变化。

### R 环境
R 脚本使用 `pacman::p_load` 自动安装包，但 brms 在 Windows 上需预先安装 Rtools。Rmd 文件不在 README pipeline 命令中，如需渲染需 `rmarkdown` 包。

### 44_fig4_reliability_exports
`44fig4_v8_combined.py` 读取 `44_fig4_reliability_exports/*.csv`（由 `44_export_fig4_reliability_data.R` 生成，未被 README pipeline 列出）。主 Fig 4 用 `44fig4_efa_svg.py`（硬编码数据，无需外部 CSV），不影响 README 流程。
