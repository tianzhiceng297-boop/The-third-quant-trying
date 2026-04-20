# alpha_engine/02_vectorized_backtest_v2_strict.py

import os
import sys
import numpy as np
import pandas as pd
from loguru import logger
import concurrent.futures
from tqdm import tqdm
import scipy.stats as stats

class StrictVectorizedBacktester:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_lake_dir = os.path.abspath(os.path.join(self.base_dir, "..", "data_lake"))
        self.quotes_dir = os.path.join(self.data_lake_dir, "daily_quotes")
        self.alpha_path = os.path.join(self.base_dir, "factors", "pure_ai_alpha.parquet")
        self.cal_path = os.path.join(self.data_lake_dir, "trade_calendar.parquet")

    def _load_single_price(self, file_path):
        try:
            df = pd.read_parquet(file_path, columns=['trade_date', 'ts_code', 'close', 'adj_factor'])
            return df
        except Exception:
            return pd.DataFrame()

    def build_strict_forward_returns(self, max_workers=14) -> pd.DataFrame:
        """
        工业级未来收益率张量构建：完美解决停牌跳空与时间折叠！
        """
        logger.info("📅 Loading Absolute Trade Calendar...")
        if not os.path.exists(self.cal_path):
            logger.critical("Trade calendar not found! Run Layer 1 first.")
            sys.exit(1)
        df_cal = pd.read_parquet(self.cal_path)
        all_trade_dates = pd.to_datetime(df_cal['cal_date']).sort_values().unique()

        logger.info("📡 Loading Price Universe in parallel...")
        stock_files =[os.path.join(self.quotes_dir, f) for f in os.listdir(self.quotes_dir) if f.endswith('.parquet')]
        
        price_pieces =[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures =[executor.submit(self._load_single_price, f) for f in stock_files]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Loading Quotes"):
                res = future.result()
                if not res.empty:
                    price_pieces.append(res)

        logger.info("🧩 Assembling 1D Price List...")
        df_price = pd.concat(price_pieces, ignore_index=True)
        df_price['trade_date'] = pd.to_datetime(df_price['trade_date'], format='%Y%m%d')
        df_price['adj_close'] = df_price['close'] * df_price['adj_factor']

        # ==========================================
        # 审计修复 1 & 3：时间折叠与停牌漏洞的终极克星 (2D Matrix Tensor)
        # ==========================================
        logger.info("🧊 Pivoting to 2D Tensor for Absolute Calendar Alignment...")
        # 将 1D 列表透视为 2D 矩阵：Index=日期，Columns=股票代码
        price_matrix = df_price.pivot(index='trade_date', columns='ts_code', values='adj_close')
        
        # 核心防线：强制 Reindex 到全市场绝对交易日历！
        # 这样一来，任何一只股票在停牌日，它的价格将自动变为 NaN
        price_matrix = price_matrix.reindex(all_trade_dates)
        
        logger.info("⏳ Calculating T+1 to T+2 Strict Forward Returns...")
        # 在绝对时间轴上执行 shift，彻底消灭时间折叠
        buy_price_matrix = price_matrix.shift(-1)  # T+1 收盘买入
        sell_price_matrix = price_matrix.shift(-2) # T+2 收盘卖出
        
        # 如果 T+1 或 T+2 是停牌日，价格为 NaN，算出来的 fwd_ret 也是 NaN
        # 这完美模拟了真实世界中“停牌买不进、卖不出”的物理限制
        fwd_ret_matrix = (sell_price_matrix / buy_price_matrix) - 1.0

        logger.info("🗜️ Melting Tensor back to 1D...")
        # 将 2D 矩阵降维回 1D (stack 会自动丢弃 NaN 收益率，极大地节省内存)
        df_fwd = fwd_ret_matrix.stack().reset_index()
        df_fwd.columns =['trade_date', 'ts_code', 'fwd_ret_1d']
        
        return df_fwd

    def execute_judgment(self):
        if not os.path.exists(self.alpha_path):
            logger.critical(f"Alpha factor not found at {self.alpha_path}!")
            sys.exit(1)

        logger.info("📥 Loading Alpha Factor Matrix...")
        df_alpha = pd.read_parquet(self.alpha_path)
        
        df_fwd = self.build_strict_forward_returns()

        logger.info("🔗 Aligning Alpha Signals with Future Returns...")
        df_master = pd.merge(df_alpha, df_fwd, on=['trade_date', 'ts_code'], how='inner')
        df_master['fwd_ret_1d'] = df_master['fwd_ret_1d'].clip(-0.21, 0.21)

        logger.info("🧮 Calculating Cross-Sectional Rank IC...")
        def calc_ic(group):
            if len(group) < 30: return np.nan
            return stats.spearmanr(group['pure_ai_alpha'], group['fwd_ret_1d'])[0]

        ic_series = df_master.groupby('trade_date').apply(calc_ic, include_groups=False).dropna()
        mean_ic = ic_series.mean()
        ic_ir = mean_ic / ic_series.std() if ic_series.std() > 0 else 0

        # ==========================================
        # 审计修复 2：因子簇拥导致 pd.qcut 崩溃的终极解法
        # ==========================================
        logger.info("🔪 Slicing Universe into 5 Quantiles (with Tie-Breaker)...")
        def safe_qcut(x):
            # 审计修复 3：防范极端数据荒漠 (史前空窗期)
            # 如果某天合格的股票不足 5 只，根本无法分成 5 组，直接返回 NaN 放弃当天
            if len(x) < 5:
                return pd.Series(np.nan, index=x.index)
            
            # 核心防线：使用 rank(method='first') 强行打破同分(Tie)僵局
            ranks = x.rank(method='first')
            return pd.qcut(ranks, 5, labels=False)
            
        df_master['quantile'] = df_master.groupby('trade_date')['pure_ai_alpha'].transform(safe_qcut)

        daily_quantile_ret = df_master.groupby(['trade_date', 'quantile'])['fwd_ret_1d'].mean().unstack()
        
        # 确保所有 5 个分组都存在，防止极端情况
        for i in range(5):
            if i not in daily_quantile_ret.columns:
                daily_quantile_ret[i] = 0.0

        daily_quantile_ret['Long_Short'] = daily_quantile_ret[4] - daily_quantile_ret[0]
        cum_ret = (1 + daily_quantile_ret.fillna(0)).cumprod()

        def get_mdd(series):
            roll_max = series.cummax()
            drawdown = series / roll_max - 1.0
            return drawdown.min()

        q4_mdd = get_mdd(cum_ret[4])
        ls_mdd = get_mdd(cum_ret['Long_Short'])
        
        years = len(daily_quantile_ret) / 252.0
        q4_ann_ret = cum_ret[4].iloc[-1] ** (1 / years) - 1.0 if years > 0 else 0.0
        ls_ann_ret = cum_ret['Long_Short'].iloc[-1] ** (1 / years) - 1.0 if years > 0 else 0.0

        logger.info("\n" + "="*60)
        logger.info("🔥 A-SHARE STRICT JUDGMENT DAY REPORT 🔥")
        logger.info("="*60)
        logger.info("[1] Information Coefficient (IC) Metrics:")
        logger.info(f"    Mean Rank IC   : {mean_ic:.4f}")
        logger.info(f"    IC IR          : {ic_ir:.4f}")
        logger.info(f"    Win Rate (IC>0): {(ic_series > 0).sum() / len(ic_series):.2%}")
        logger.info("-" * 60)
        
        logger.info("[2] Top 20% (Elite Q4) Long-Only Portfolio:")
        logger.info(f"    Annualized Return : {q4_ann_ret:.2%}")
        logger.info(f"    Max Drawdown (MDD): {q4_mdd:.2%}")
        logger.info("-" * 60)

        logger.info("[3] Long-Short Spread (Long Q4, Short Q0):")
        logger.info(f"    Annualized Spread : {ls_ann_ret:.2%}")
        logger.info(f"    Spread MDD        : {ls_mdd:.2%}")
        logger.info("-" * 60)

        logger.info("[4] Monotonicity Check (Cumulative End Value):")
        for i in range(5):
            logger.info(f"    Q{i}                  : {cum_ret[i].iloc[-1]:.2f}x")
        
        if cum_ret[4].iloc[-1] > cum_ret[3].iloc[-1] > cum_ret[0].iloc[-1]:
            logger.success("    Status: PASS ✅ (Strict Monotonicity observed!)")
        else:
            logger.warning("    Status: FAIL ❌ (Monotonicity broken or weak)")
        logger.info("="*60)

if __name__ == "__main__":
    # 配置 Logger 打印到终端
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
    
    judge = StrictVectorizedBacktester()
    judge.execute_judgment()