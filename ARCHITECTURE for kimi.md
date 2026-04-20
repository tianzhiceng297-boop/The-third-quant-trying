# 项目整体架构与依赖关系分析

> 本文档由 Kimi 于 2026-04-13 生成，用于快速了解 A-Share Quantamental Framework 的全局架构与模块依赖。

---

## 一、项目概览

这是一个 **A股量化基本面投资框架**（Quantamental Framework），采用工业级分层流水线架构，涵盖从数据采集、因子生产、风险中性化到组合优化与绩效评估的完整链路。

- **Python 文件总数**: 18 个
- **代码总行数**: ~3,225 行
- **核心特点**: 模块间**无 Python `import` 依赖**，各脚本完全独立，通过 `parquet` 文件系统进行数据传递

---

## 二、目录结构与模块清单

```
quant_project/
├── check/                          # 环境检查层
│   └── env_check.py                # 63行
├── data_lake/                      # Layer 1: 数据基础设施层
│   ├── 01_build_universe_v5_final.py              # 208行
│   ├── 02_download_daily_quotes_v8_pinnacle.py    # 248行
│   └── 03_download_financials_pit_v11_patch.py    # 240行
├── alpha_engine/                   # Layer 2-4: 因子引擎与组合优化层
│   ├── 01_build_pead_factor_v2.py                 # 172行 (业绩动量)
│   ├── 01_build_reversal_factor_v3.py             # 146行 (低波反转)
│   ├── 01_build_microstructure_factor_v6.py       # 125行 (微观结构)
│   ├── 01_build_pure_fundamentals_v2_pro.py       # 159行 (纯基本面)
│   ├── 01_build_qarp_factor_v2_pro.py             # 158行 (质量价值)
│   ├── 01_build_consensus_factor.py               # 196行 (分析师共识)
│   ├── 01_build_adv_micro_factor.py               # 124行 (高级微观)
│   ├── 01_build_meta_crowding.py                  # 121行 (因子拥挤度)
│   ├── 02_vectorized_backtest_v2_strict.py        # 176行 (回测)
│   ├── 03_risk_neutralization_v2_fast.py          # 144行 (快速中性化)
│   ├── 03_risk_neutralization_v3_ai.py            # 117行 (AI中性化)
│   ├── 04_classic_linear_optimizer.py             # 177行 (组合优化)
│   ├── 05_portfolio_performance_eval_v4_perfect.py# 198行 (绩效评估)
│   ├── 06_drawdown_attribution.py                 # 143行 (回撤归因)
│   └── jygj.py                                    # 310行 (协整分析工具)
├── src/                            # 源码目录 (当前为空)
├── KMFCC/                          # (当前为空)
├── data/                           # 数据目录
├── logs/                           # 日志目录
└── ARCHITECTURE.md                 # 架构文档
```

---

## 三、模块分层架构

| 层级 | 目录 | 模块职责 | 执行方式 |
|------|------|----------|----------|
| **Layer 0** | `check/` | 环境依赖与 API 连通性检查 | 一次性 |
| **Layer 1** | `data_lake/` | 构建股票池、下载日线行情与财务数据 | **必须顺序执行** |
| **Layer 2** | `alpha_engine/01_*` | 8 类单因子构建 | 可并行执行 |
| **Layer 2.5** | `alpha_engine/01_build_meta_crowding.py` | 元因子（依赖其他因子输出） | 等 Layer 2 完成后 |
| **Layer 3** | `alpha_engine/03_risk_neutralization_v*.py` | SVD 风险中性化 | 顺序执行 |
| **Layer 4** | `alpha_engine/02_*, 04_*` | 回测与组合优化 | 顺序执行 |
| **Layer 5** | `alpha_engine/05_*, 06_*` | 绩效评估与回撤归因 | 最后执行 |

---

## 四、模块间依赖关系（数据流/执行顺序）

> **注意**：各 Python 脚本之间没有代码层面的 `import` 关系，全部为独立运行的脚本。以下依赖是指**数据输入输出**和**执行先后**关系。

### 4.1 数据流全景图

```
Tushare API
    │
    ├───► data_lake/01_build_universe ──► stock_basics.parquet
    │                                       trade_calendar.parquet
    │                                       unified_name_history.parquet
    │
    ├───► data_lake/02_download_daily_quotes ──► daily_quotes/*.parquet
    │
    └───► data_lake/03_download_financials ───► financials/*.parquet
                                                       │
    ┌──────────────────────────────────────────────────┘
    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        alpha_engine (因子生产层)                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ PEAD因子 │ │ 反转因子 │ │ 微观结构 │ │ 基本面   │ │ QARP     │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘   │
│       │            │            │            │            │          │
│  ┌────┴────────────┴────────────┴────────────┴────────────┘          │
│  │                    01_build_meta_crowding                           │
│  │                    (因子拥挤度，依赖上述所有因子)                      │
│  └─────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────┐    ┌────────────────────────┐
│ 03_risk_neutralization │───►│ 02_vectorized_backtest │
│   (SVD 正交化/行业中性) │    │   (IC/分组收益回测)     │
└───────────┬────────────┘    └────────────────────────┘
            │
            ▼
    ┌──────────────────┐
    │ 04_classic_linear│
    │   optimizer      │
    │ (组合优化/调仓)   │
    └────────┬─────────┘
             │
             ▼
    ┌────────────────────────┐
    │ 05_portfolio_performance│
    │   (绩效评估/TCA模拟)    │
    └────────────────────────┘
             │
             ▼
    ┌────────────────────────┐
    │ 06_drawdown_attribution │
    │   (最大回撤归因分析)     │
    └────────────────────────┘
```

### 4.2 详细依赖矩阵

| 模块 | 上游数据输入 | 下游消费者 |
|------|-------------|-----------|
| `01_build_universe` | Tushare API | 所有其他模块 |
| `02_download_daily_quotes` | Tushare API | 所有因子模块、回测、优化器、绩效评估 |
| `03_download_financials` | Tushare API | PEAD、QARP、基本面因子 |
| `01_build_pead` | daily_quotes, financials | Meta拥挤度、优化器、回测 |
| `01_build_reversal` | daily_quotes | Meta拥挤度、优化器、回测 |
| `01_build_microstructure` | daily_quotes | Meta拥挤度 |
| `01_build_pure_fundamentals` | daily_quotes, financials, stock_basics | Meta拥挤度、优化器 |
| `01_build_qarp` | daily_quotes, financials, stock_basics | - |
| `01_build_consensus` | daily_quotes, Tushare API, stock_basics | - |
| `01_build_adv_micro` | daily_quotes | 优化器、回测 |
| `01_build_meta_crowding` | 所有 Layer 2 因子输出 | - |
| `03_risk_neutralization_v3_ai` | ai_composite_alpha, daily_quotes, stock_basics | 回测 |
| `04_classic_linear_optimizer` | daily_quotes, trade_calendar, names, 所有因子 | `target_weights_final.parquet` |
| `05_portfolio_performance_eval` | daily_quotes, trade_calendar, target_weights | - |
| `06_drawdown_attribution` | daily_quotes, trade_calendar, target_weights | - |

---

## 五、Python 外部依赖关系（按模块）

### 5.1 汇总：所有使用到的第三方库

| 库名称 | 用途 | 使用模块数 |
|--------|------|-----------|
| `pandas` | 数据处理主体 | 16 |
| `numpy` | 数值计算 | 15 |
| `pyarrow` | Parquet 读写引擎 | 13 |
| `tushare` | A股数据源 | 4 |
| `loguru` | 结构化日志 | 15 |
| `tqdm` | 进度条 | 15 |
| `concurrent.futures` | 多线程并发 | 15 |
| `python-dotenv` | 环境变量管理 | 4 |
| `filelock` | 文件锁防并发冲突 | 3 |
| `scipy` | 统计检验 (spearmanr) | 1 |
| `yfinance` | 雅虎财经数据 | 1 |
| `statsmodels` | 协整检验/OLS/ADF | 1 |
| `matplotlib` | 图表绘制 | 1 |

### 5.2 按模块细分的 import 清单

#### **data_lake/** (数据层)

| 模块 | 标准库 | 第三方库 |
|------|--------|---------|
| `01_build_universe_v5_final.py` | `sys` | `tushare`, `python-dotenv`, `filelock` |
| `02_download_daily_quotes_v8_pinnacle.py` | `os`, `sys`, `time`, `glob`, `datetime`, `concurrent.futures` | `tushare`, `loguru`, `python-dotenv`, `tqdm`, `filelock`, `pyarrow`(运行时) |
| `03_download_financials_pit_v11_patch.py` | `os`, `sys`, `time`, `glob`, `random`, `datetime`, `concurrent.futures` | `tushare`, `loguru`, `python-dotenv`, `tqdm`, `filelock`, `pyarrow`(运行时) |

#### **alpha_engine/01_*** (因子构建层)

| 模块 | 标准库 | 第三方库 |
|------|--------|---------|
| `01_build_pead_factor_v2.py` | `os`, `sys`, `gc`, `shutil`, `concurrent.futures` | `numpy`, `pandas`, `loguru`, `tqdm`, `pyarrow`(运行时) |
| `01_build_reversal_factor_v3.py` | `os`, `sys`, `gc`, `shutil`, `concurrent.futures` | `numpy`, `pandas`, `loguru`, `tqdm`, `pyarrow`(运行时) |
| `01_build_microstructure_factor_v6.py` | `os`, `sys`, `gc`, `shutil`, `concurrent.futures` | `numpy`, `pandas`, `loguru`, `tqdm`, `pyarrow`(运行时) |
| `01_build_pure_fundamentals_v2_pro.py` | `os`, `sys`, `gc`, `shutil`, `concurrent.futures` | `numpy`, `pandas`, `loguru`, `tqdm`, `pyarrow`(运行时) |
| `01_build_qarp_factor_v2_pro.py` | `os`, `sys`, `gc`, `shutil`, `concurrent.futures` | `numpy`, `pandas`, `loguru`, `tqdm`, `pyarrow`(运行时) |
| `01_build_consensus_factor.py` | `os`, `sys`, `gc`, `shutil`, `concurrent.futures` | `numpy`, `pandas`, `loguru`, `tqdm`, `tushare`, `python-dotenv`, `pyarrow`(运行时) |
| `01_build_adv_micro_factor.py` | `os`, `sys`, `gc`, `shutil`, `concurrent.futures` | `numpy`, `pandas`, `loguru`, `tqdm`, `pyarrow`(运行时) |
| `01_build_meta_crowding.py` | `os`, `sys`, `functools.reduce` | `numpy`, `pandas`, `loguru`, `tqdm`, `pyarrow`(运行时) |

#### **alpha_engine/** (回测、中性化、优化、评估)

| 模块 | 标准库 | 第三方库 |
|------|--------|---------|
| `02_vectorized_backtest_v2_strict.py` | `os`, `sys`, `concurrent.futures` | `numpy`, `pandas`, `loguru`, `tqdm`, `scipy.stats`, `pyarrow`(运行时) |
| `03_risk_neutralization_v2_fast.py` | `os`, `sys`, `concurrent.futures` | `numpy`, `pandas`, `loguru`, `tqdm`, `pyarrow`(运行时) |
| `03_risk_neutralization_v3_ai.py` | `os`, `sys`, `concurrent.futures` | `numpy`, `pandas`, `loguru`, `tqdm`, `pyarrow`(运行时) |
| `04_classic_linear_optimizer.py` | `os`, `sys`, `concurrent.futures`, `functools.reduce` | `numpy`, `pandas`, `loguru`, `tqdm`, `pyarrow`(运行时) |
| `05_portfolio_performance_eval_v4_perfect.py` | `os`, `sys`, `concurrent.futures` | `numpy`, `pandas`, `loguru`, `tqdm`, `pyarrow`(运行时) |
| `06_drawdown_attribution.py` | `os`, `sys`, `concurrent.futures` | `numpy`, `pandas`, `loguru`, `tqdm`, `pyarrow`(运行时) |

#### **工具脚本**

| 模块 | 标准库 | 第三方库 |
|------|--------|---------|
| `jygj.py` | `warnings` | `pandas`, `numpy`, `yfinance`, `statsmodels`, `matplotlib` |
| `check/env_check.py` | `sys`, `platform`, `importlib` | `tushare`(运行时检查) |

---

## 六、关键发现

1. **零内部 Python 依赖**：所有 18 个脚本都是**独立可执行文件**，不存在 `from src.xxx import ...` 或互相 `import` 的情况。`src/` 目录目前为空。
2. **数据传递靠文件系统**：模块间耦合完全通过读写 `parquet` 文件实现，这保证了各层可以独立开发、测试和并行运行。
3. **依赖高度同质化**：`alpha_engine/` 下的 15 个模块中，有 14 个共享几乎完全相同的 import 模式（`os, sys, numpy, pandas, loguru, concurrent.futures, tqdm`）。
4. **唯一的外部数据源**：除 `jygj.py` 使用 `yfinance` 外，其余所有数据均来自 **Tushare API**。
5. **可选/降级依赖**：部分库（如 `tushare`, `dotenv`）采用了 `try/except` 运行时导入，脚本头部并未直接 `import`。

---

## 七、推荐执行顺序

```bash
# Layer 1: 数据基建 (必须按顺序)
python data_lake/01_build_universe_v5_final.py
python data_lake/02_download_daily_quotes_v8_pinnacle.py
python data_lake/03_download_financials_pit_v11_patch.py

# Layer 2: 因子生产 (可并行)
python alpha_engine/01_build_pead_factor_v2.py
python alpha_engine/01_build_reversal_factor_v3.py
python alpha_engine/01_build_microstructure_factor_v6.py
python alpha_engine/01_build_pure_fundamentals_v2_pro.py
python alpha_engine/01_build_qarp_factor_v2_pro.py
python alpha_engine/01_build_consensus_factor.py
python alpha_engine/01_build_adv_micro_factor.py

# Layer 2.5: 元因子
python alpha_engine/01_build_meta_crowding.py

# Layer 3: 风险中性化
python alpha_engine/03_risk_neutralization_v3_ai.py

# Layer 4: 回测与优化
python alpha_engine/02_vectorized_backtest_v2_strict.py
python alpha_engine/04_classic_linear_optimizer.py

# Layer 5: 绩效评估
python alpha_engine/05_portfolio_performance_eval_v4_perfect.py
python alpha_engine/06_drawdown_attribution.py
```
