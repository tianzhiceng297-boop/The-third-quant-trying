# A-Share Quantamental Framework 架构文档

> 本文档描述 A股量化基本面框架 的整体架构、模块依赖关系和数据流。
> 生成时间: 2026-04-13

---

## 一、项目概述

本项目是一个工业级 A股量化投资框架，采用分层架构设计，涵盖数据采集、因子构建、风险中性化、组合优化和绩效评估全链路。

### 目录结构

```
quant_project/
├── check/                    # 环境检查
│   └── env_check.py          # 依赖环境和API连通性检查
├── data_lake/                # 数据层 (Layer 1)
│   ├── 01_build_universe_v5_final.py
│   ├── 02_download_daily_quotes_v8_pinnacle.py
│   ├── 03_download_financials_pit_v11_patch.py
│   ├── daily_quotes/         # 日线数据存储 (parquet)
│   ├── financials/           # 财务数据存储 (parquet)
│   └── *.parquet             # 元数据文件
├── alpha_engine/             # 因子引擎层 (Layer 2-4)
│   ├── 01_build_*_factor*.py # 因子构建模块 (8个因子)
│   ├── 02_vectorized_backtest_v2_strict.py
│   ├── 03_risk_neutralization_v*.py
│   ├── 04_classic_linear_optimizer.py
│   ├── 05_portfolio_performance_eval_v4_perfect.py
│   ├── 06_drawdown_attribution.py
│   ├── jygj.py               # 协整分析工具
│   └── factors/              # 因子输出目录
├── data/                     # 数据目录
├── logs/                     # 日志目录
└── src/                      # 源码目录
```

---

## 二、模块分层架构

### Layer 1: 数据基础设施层 (data_lake/)

| 模块 | 功能 | 输入依赖 | 输出 |
|------|------|----------|------|
| `01_build_universe_v5_final.py` | 构建全市场股票池、交易日历、名称历史 | Tushare API | `stock_basics.parquet`, `trade_calendar.parquet`, `unified_name_history.parquet` |
| `02_download_daily_quotes_v8_pinnacle.py` | 并发下载日线行情(价格/复权因子) | Tushare API, stock_basics | `daily_quotes/*.parquet` |
| `03_download_financials_pit_v11_patch.py` | 下载财务报表数据(PIT) | Tushare API, stock_basics | `financials/*.parquet` |

**核心特性:**
- 智能退避重试机制 (指数退避)
- 文件锁防止并发冲突
- 原子写入 (先写tmp再rename)
- 增量更新支持

### Layer 2: Alpha 因子构建层 (alpha_engine/01_*)

| 模块 | 因子类型 | 输入依赖 | 输出 |
|------|----------|----------|------|
| `01_build_pead_factor_v2.py` | 业绩动量因子 (SUE) | daily_quotes, financials | `pead_alpha.parquet` |
| `01_build_reversal_factor_v3.py` | 低波反转因子 | daily_quotes | `reversal_alpha.parquet` |
| `01_build_microstructure_factor_v6.py` | 微观结构因子(MAX/Amihud) | daily_quotes | `micro_features.parquet` |
| `01_build_pure_fundamentals_v2_pro.py` | 纯基本面因子(BP/EP/增速) | daily_quotes, financials, stock_basics | `pure_fundamentals.parquet` |
| `01_build_qarp_factor_v2_pro.py` | QARP质量价值因子 | daily_quotes, financials, stock_basics | `qarp_alpha.parquet` |
| `01_build_consensus_factor.py` | 分析师共识因子 | daily_quotes, Tushare API, stock_basics | `consensus_alpha.parquet` |
| `01_build_adv_micro_factor.py` | 高级微观因子(振幅/价量相关) | daily_quotes | `adv_micro_alpha.parquet` |
| `01_build_meta_crowding.py` | 因子拥挤度元特征 | 所有其他因子输出 | `meta_crowding_features.parquet` |

**核心特性:**
- MapReduce 架构 (Map: 时序计算, Reduce: 截面处理)
- 临时分片磁盘溢写防OOM
- 向量化横截面标准化 (Rank-Z/MAD)
- 行业中性化处理

### Layer 2.5-3: 因子合成与风险中性化

| 模块 | 功能 | 输入依赖 | 输出 |
|------|------|----------|------|
| `02_vectorized_backtest_v2_strict.py` | 严格向量化回测(IC/分组收益) | daily_quotes, trade_calendar, `pure_ai_alpha.parquet` | 回测报告 |
| `03_risk_neutralization_v2_fast.py` | 快速Barra中性化(行业/市值) | `ai_composite_alpha.parquet`, daily_quotes, stock_basics, names | `pure_alpha.parquet` |
| `03_risk_neutralization_v3_ai.py` | AI版本中性化 | 同上 | `pure_ai_alpha.parquet` |

**核心特性:**
- NumPy SVD 正交化
- 2D Tensor 处理停牌/时间折叠问题
- 剔除ST股和微盘股
- PIT (Point-in-Time) 名称历史处理

### Layer 4: 组合优化与执行

| 模块 | 功能 | 输入依赖 | 输出 |
|------|------|----------|------|
| `04_classic_linear_optimizer.py` | 线性多因子打分+调仓优化 | daily_quotes, trade_calendar, names, 所有因子 | `target_weights_final.parquet` |
| `05_portfolio_performance_eval_v4_perfect.py` | 真实交易成本模拟(TCA) | daily_quotes, trade_calendar, target_weights | 绩效报告 |
| `06_drawdown_attribution.py` | 最大回撤归因分析 | daily_quotes, trade_calendar, target_weights | 回撤分析报告 |

**核心特性:**
- 60日均线趋势择时 (熊市暂停)
- 双轨宽容带调仓 (Hysteresis Buffer)
- 购买力引擎 (防止爆仓)
- 涨停/跌停物理限制模拟

### 工具模块

| 模块 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `jygj.py` | 协整分析(配对交易) | yfinance API | 协整分析报告/图表 |
| `env_check.py` | 环境检查 | 系统环境 | 环境状态报告 |

---

## 三、Python 依赖关系

### 3.1 标准库依赖

| 模块 | 用途 |
|------|------|
| `os, sys` | 文件路径、环境变量、系统退出 |
| `gc` | 垃圾回收 (内存管理) |
| `glob` | 文件通配符匹配 |
| `time, datetime` | 时间处理 |
| `concurrent.futures` | 多线程并发 |
| `functools.reduce` | 多表合并 |
| `importlib` | 动态导入检查 |
| `warnings` | 警告过滤 |
| `shutil` | 目录清理 |
| `platform` | 系统信息 |
| `random` | 随机数(退避策略) |

### 3.2 第三方库依赖

| 库 | 版本要求 | 用途 |
|----|---------|------|
| `pandas` | >=2.0 | 数据处理主体 |
| `numpy` | >=1.24 | 数值计算 |
| `pyarrow` | >=12.0 | Parquet读写后端 |
| `tushare` | >=1.3 | A股数据源 |
| `yfinance` | >=0.2 | 雅虎财经(协整分析) |
| `scipy` | >=1.10 | 统计分析 (spearmanr) |
| `statsmodels` | >=0.14 | 协整检验、ADF、OLS |
| `matplotlib` | >=3.7 | 图表绘制 |
| `loguru` | >=0.7 | 结构化日志 |
| `tqdm` | >=4.65 | 进度条 |
| `filelock` | >=3.12 | 文件锁 |
| `python-dotenv` | >=1.0 | 环境变量管理 |

### 3.3 依赖安装命令

```bash
pip install pandas numpy pyarrow tushare yfinance scipy statsmodels matplotlib loguru tqdm filelock python-dotenv
```

---

## 四、数据流依赖图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Tushare API (外部数据源)                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│  01_build_    │          │  02_download_ │          │  03_download_ │
│  universe     │          │  daily_quotes │          │  financials   │
└───────┬───────┘          └───────┬───────┘          └───────┬───────┘
        │                          │                          │
        ▼                          ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Data Lake (原始数据存储)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │stock_basics  │  │trade_calendar│  │daily_quotes  │  │financials    │ │
│  │.parquet     │  │.parquet      │  │/*.parquet   │  │/*.parquet    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       Alpha Engine (因子生产层)                           │
│                                                                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │PEAD因子     │ │反转因子     │ │微观结构     │ │基本面因子   │       │
│  │(业绩动量)   │ │(低波动)     │ │(MAX/Amihud)│ │(BP/EP/增速) │       │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘       │
│         │               │               │               │               │
│  ┌──────┴───────────────┴───────────────┴───────────────┴──────┐       │
│  │                      Meta 因子拥挤度                          │       │
│  └─────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Risk Neutralization (风险中性化)                     │
│                      ┌───────────────────────┐                          │
│                      │  SVD 正交化 + 行业中性 │                          │
│                      │  剔除市值/流动性暴露   │                          │
│                      └───────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Portfolio Optimization (组合优化)                    │
│                      ┌───────────────────────┐                          │
│                      │  多因子打分 + 调仓策略 │                          │
│                      │  趋势过滤 + 熊市暂停   │                          │
│                      └───────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Performance Evaluation (绩效评估)                    │
│                      ┌───────────────────────┐                          │
│                      │  TCA成本模拟          │                          │
│                      │  回撤归因 + 风险分析   │                          │
│                      └───────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 五、执行顺序

```
# Layer 1: 数据基建 (必须按顺序执行)
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

# Layer 2.5: 元因子 (依赖其他因子)
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

---

## 六、关键设计特点

### 6.1 内存管理
- 使用 `gc.collect()` 主动触发垃圾回收
- `del` 显式删除大对象
- `shutil.rmtree()` 清理临时分片目录
- PyArrow 引擎高效读取Parquet目录

### 6.2 并发处理
- 所有数据拉取使用 `ThreadPoolExecutor`
- 文件锁 (`filelock`) 防止并发写入冲突
- 智能退避策略应对API限流

### 6.3 数据完整性
- 原子写入模式 (tmp → rename)
- PIT (Point-in-Time) 处理避免未来函数
- `pd.merge_asof(direction='backward')` 严格时序对齐
- 停牌处理使用 2D Tensor 对齐交易日历

### 6.4 风险控制
- 横截面标准化 (Rank-Z / MAD)
- 行业中性化消除共线性
- SVD 正交化剔除风险暴露
- 趋势择时自动空仓保护

### 6.5 交易成本
- 买入成本: 0.15% (佣金+滑点)
- 卖出成本: 0.25% (印花税+佣金+滑点)
- 购买力引擎防止爆仓
- 涨停/跌停物理限制模拟

---

## 七、文件输出映射

| 输入模块 | 输出文件 | 下游消费者 |
|----------|----------|------------|
| 01_build_universe | `stock_basics.parquet` | 所有因子模块, 中性化, 优化器 |
| 01_build_universe | `trade_calendar.parquet` | 回测, 优化器, 绩效评估 |
| 01_build_universe | `unified_name_history.parquet` | 中性化, 优化器 |
| 02_download_daily_quotes | `daily_quotes/*.parquet` | 所有因子模块 |
| 03_download_financials | `financials/*.parquet` | PEAD, QARP, 基本面因子 |
| 01_build_pead | `pead_alpha.parquet` | Meta拥挤度, 优化器, 回测 |
| 01_build_reversal | `reversal_alpha.parquet` | Meta拥挤度, 优化器, 回测 |
| 01_build_microstructure | `micro_features.parquet` | Meta拥挤度 |
| 01_build_pure_fundamentals | `pure_fundamentals.parquet` | Meta拥挤度, 优化器 |
| 01_build_qarp | `qarp_alpha.parquet` | - |
| 01_build_consensus | `consensus_alpha.parquet` | - |
| 01_build_adv_micro | `adv_micro_alpha.parquet` | 优化器, 回测 |
| 01_build_meta_crowding | `meta_crowding_features.parquet` | 机器学习合成 |
| 03_risk_neutralization | `pure_ai_alpha.parquet` | 回测 |
| 04_classic_linear_optimizer | `target_weights_final.parquet` | 绩效评估, 回撤归因 |

---

## 八、注意事项

1. **API Token**: 所有使用 Tushare 的模块需要配置 `TUSHARE_TOKEN` 环境变量
2. **执行顺序**: Layer 1 必须按顺序执行，Layer 2 因子可并行但需在 Layer 1 完成后
3. **内存要求**: 全市场数据加载需要至少 16GB 内存
4. **存储空间**: 完整数据下载约需 10-20GB 磁盘空间
5. **网络要求**: Tushare API 需要稳定的网络连接和足够的积分

---

*本文档自动生成，如有更新请同步修改。*
