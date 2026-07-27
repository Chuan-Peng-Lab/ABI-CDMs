# ABI-CDMs 代码可运行性静态审查（2026-07-27）

## 审查结论

审查发现 **4 个问题**，其中 2 个已修复（import 与 __init__.py bug），1 个已撤销（bayesflow 不存在），剩余 1 个为预置数据/路径依赖问题。

| # | 问题 | 严重度 | 状态 |
|---|------|--------|------|
| 1 | `bayesflow>=2.0` 不存在于 PyPI | 🔴致命 | ❌ **撤销**：bayesflow 2.0.0–2.0.12 已上传 PyPI（审查时 pip 缓存了旧索引） |
| 2 | 裸导入与 `pip install -e .` 不兼容 | 🔴致命 | ✅ **已修复**：nsbi_module 内部改相对导入，29 个 scripts 改 `from nsbi_module.X import` |
| 3 | `__init__.py` 导入名错误 | 🔴致命 | ✅ **已修复**：CDMs_NSBI→NSBICDM，CDMsSimulator 源模块修正，移除不存在函数 |
| 4 | 部分脚本依赖 CWD 为脚本所在目录 | 🟡次要 | ⚠️ **预置行为**：`read_csv("./...")` 等数据路径需脚本目录 CWD，README 的 `cd` 是必要的 |

安装路线（pip/conda/setup）现已可行。`pip install -e .` 后 `from nsbi_module import NSBICDM` 可正常导入。

---

## 修复详情

### 已修复 ①：nsbi_module 内部导入（5 文件 → 相对导入）
- `nsbi_module/NSBI_CDMs.py`：`from simulators import` → `from .simulators import`
- `nsbi_module/trainer.py`：`from default_settings import` → `from .default_settings import`
- `nsbi_module/analysis_utils.py`：4 处裸导入 → 相对导入
- `nsbi_module/model_metrics.py`：`from utils_preprocessing import` → `from .utils_preprocessing import`
- `nsbi_module/utils_ind_diff.py`：`from study_labels import` → `from .study_labels import`

### 已修复 ②：scripts 导入（29 文件 → 包限定）
所有 `scripts/` 下的 Python 文件中，以下裸模块名统一加 `nsbi_module.` 前缀：
`NSBI_CDMs`, `utils`, `model_metrics`, `analysis_utils`, `plotting`, `study_labels`, `utils_ind_diff`, `utils_pydmc`, `default_settings`, `utils_preprocessing`, `dmc_v2_loader`, `dmc_vs_loader`, `tsdm_loader`, `simulators`, `dists`, `utils_flexDDM`

例如：
- `from NSBI_CDMs import NSBICDM` → `from nsbi_module.NSBI_CDMs import NSBICDM`
- `from utils import FitStore` → `from nsbi_module.utils import FitStore`
- `import simulators as simulators` → `import nsbi_module.simulators as simulators`

### 已修复 ③：`nsbi_module/__init__.py`
- `CDMs_NSBI`（不存在）→ `NSBICDM`（实际类名）
- `from nsbi_module.simulators import CDMsSimulator` → `from nsbi_module.trainer import CDMsSimulator`（CDMsSimulator 在 trainer.py 非 simulators.py）
- 移除 `compute_rmse`/`compute_g_square`（ModelMetricEvaluator 实例方法，非顶层函数）
- 移除 `get_best_model_by_metric`（在 utils_ind_diff.py 非 model_metrics.py）

---

## 待关注

### CWD 依赖（预置行为）
`scripts/03_fitting/` 和 `scripts/07_parameter_analysis/` 中的脚本使用 `pd.read_csv("./...")` / `read.csv("./...")` 读取数据，需 CWD 为脚本所在目录。README 中的 `cd scripts/XX` 指令是必要的（非 bug，是当前设计的行为）。未来可考虑改为基于 `__file__` 的绝对路径。

### bayesflow.utils 内部 API 稳定性
`scripts/04_validation/11parameter_recovery.py:783-786` 直接导入 BayesFlow 内部模块 `bayesflow.utils.ecdf` / `bayesflow.utils.plot_utils` / `bayesflow.utils.dict_utils`。这些是 BayesFlow 2.x 的内部工具模块，未承诺公开 API 稳定性，跨版本可能变化。目前代码的 `trainer.py` 正确使用了 2.x 公开 API（`bf.BasicWorkflow`、`bf.networks.FlowMatching` 等），建议 11parameter_recovery.py 也统一用公开 API 或锁定 BayesFlow 版本。

### checkpoints 为空
`pip install -e .` 后仍需按 README 从 Zenodo 下载 checkpoints 并解压到 `checkpoints/`，否则 Step 3+ 会因找不到模型权重报错。
