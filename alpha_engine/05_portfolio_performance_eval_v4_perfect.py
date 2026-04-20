# alpha_engine/05_portfolio_performance_eval_v4_perfect.py

import os
import sys
import numpy as np
import pandas as pd
from loguru import logger
import concurrent.futures
from tqdm import tqdm

class PerfectLiveSimulator:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_lake_dir = os.path.abspath(os.path.join(self.base_dir, "..", "data_lake"))
        self.quotes_dir = os.path.join(self.data_lake_dir, "daily_quotes")
        self.cal_path = os.path.join(self.data_lake_dir, "trade_calendar.parquet")
        
        self.weights_path = os.path.join(self.base_dir, "factors", "target_weights_final.parquet")
        
        # 机构级 TCA 摩擦成本
        self.BUY_COST = 0.0015  # 佣金 0.03% + 滑点冲击 0.12%
        self.SELL_COST = 0.0025 # 印花税 0.10% + 佣金 0.03% + 滑点冲击 0.12%

    def _load_single_price(self, file_path):
        try: return pd.read_parquet(file_path, columns=['trade_date', 'ts_code', 'close', 'high', 'low', 'vol', 'adj_factor'])
        except Exception: return pd.DataFrame()

    def build_physical_environment(self, max_workers=14):
        logger.info("📅 1/4 Loading Absolute Trade Calendar...")
        df_cal = pd.read_parquet(self.cal_path)
        self.all_dates = pd.to_datetime(df_cal['cal_date']).sort_values().unique()

        logger.info("📡 2/4 Loading Price & Micro-structure Universe...")
        stock_files =[os.path.join(self.quotes_dir, f) for f in os.listdir(self.quotes_dir) if f.endswith('.parquet')]
        
        price_pieces =[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures =[executor.submit(self._load_single_price, f) for f in stock_files]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Loading Quotes"):
                res = future.result()
                if not res.empty: price_pieces.append(res)

        df_price = pd.concat(price_pieces, ignore_index=True)
        df_price['trade_date'] = pd.to_datetime(df_price['trade_date'])
        df_price['adj_close'] = df_price['close'] * df_price['adj_factor']

        logger.info("🧱 3/4 Forging 2D Physical Tensors (Prices & Limits)...")
        adj_close_mat = df_price.pivot(index='trade_date', columns='ts_code', values='adj_close').reindex(self.all_dates)
        self.ret_mat = (adj_close_mat / adj_close_mat.shift(1) - 1.0).fillna(0.0).values
        
        high_mat = df_price.pivot(index='trade_date', columns='ts_code', values='high').reindex(self.all_dates).values
        low_mat = df_price.pivot(index='trade_date', columns='ts_code', values='low').reindex(self.all_dates).values
        vol_mat = df_price.pivot(index='trade_date', columns='ts_code', values='vol').reindex(self.all_dates).fillna(0).values

        # 物理限制掩码
        is_trading = vol_mat > 0
        is_limit_up = (high_mat == low_mat) & (self.ret_mat > 0.09)
        is_limit_down = (high_mat == low_mat) & (self.ret_mat < -0.09)

        self.can_buy_mat = is_trading & (~is_limit_up)
        self.can_sell_mat = is_trading & (~is_limit_down)
        self.stock_cols = adj_close_mat.columns.tolist()

    def run_true_simulation(self):
        self.build_physical_environment()

        logger.info("📥 4/4 Loading Fortress Order Book & Initializing Cash Account...")
        if not os.path.exists(self.weights_path):
            logger.critical("Target weights not found! Run Layer 4 first.")
            sys.exit(1)

        df_weights = pd.read_parquet(self.weights_path)
        df_weights['trade_date'] = pd.to_datetime(df_weights['trade_date'])
        
        # 清洗重影订单
        df_weights = df_weights.drop_duplicates(subset=['trade_date', 'ts_code'], keep='last')
        target_w_mat = df_weights.pivot(index='trade_date', columns='ts_code', values='exec_weight')
        target_w_mat = target_w_mat.reindex(index=self.all_dates, columns=self.stock_cols).fillna(0.0).values

        num_days, num_stocks = self.ret_mat.shape
        
        portfolio_capital = np.zeros(num_days)
        cash = 1.0 
        holdings = np.zeros(num_stocks) 
        
        total_tca_paid = 0.0
        total_turnover = 0.0

        logger.info("⚔️ Commencing Real-world Execution with Buying Power Engine...")
        for t in tqdm(range(num_days - 1), desc="Simulating Daily Execution"):
            
            # 1. 市场开盘波动
            today_ret = self.ret_mat[t+1]
            # 兼容底层计算 NaN
            today_ret = np.nan_to_num(today_ret, nan=0.0)
            holdings = holdings * (1.0 + today_ret)
            
            # 2. 计算盘前净值
            gross_nav = np.nansum(holdings) + cash
            portfolio_capital[t+1] = gross_nav
            
            # 3. 调取目标与计算 Delta
            target_weights = target_w_mat[t]
            prev_target = target_w_mat[t-1] if t > 0 else np.zeros_like(target_weights)
            
            # 强行锁死无意义漂移摩擦
            if np.array_equal(target_weights, prev_target) and np.nansum(target_weights) > 0:
                desired_trades = np.zeros_like(holdings)
            else:
                target_holdings = target_weights * gross_nav
                desired_trades = target_holdings - holdings
            
            can_buy = self.can_buy_mat[t+1]
            can_sell = self.can_sell_mat[t+1]
            
            # 4. 先执行卖出，获取现金
            sell_orders = np.minimum(0, desired_trades)
            actual_sells = sell_orders * can_sell 
            sell_amount = np.nansum(np.abs(actual_sells))
            
            # 扣除卖出手续费
            sell_tca = sell_amount * self.SELL_COST
            cash += (sell_amount - sell_tca)
            total_tca_paid += sell_tca
            total_turnover += sell_amount
            
            # 5. 再执行买入 (🚨 修复 NaN 破产危机的核心：购买力引擎)
            buy_orders = np.maximum(0, desired_trades)
            actual_buys = buy_orders * can_buy 
            total_buy_needed = np.nansum(actual_buys)
            
            # 考虑到买入不仅要付股票钱，还要付手续费，计算所需的总资金
            total_cost_needed = total_buy_needed * (1 + self.BUY_COST)
            
            # 如果想买的超过了手里的现金，按现金购买力缩减所有买单
            if total_cost_needed > cash:
                # 绝对安全底线：现金即使为负，可用购买力也只能算 0
                available_cash = max(0.0, cash)
                scale_factor = (available_cash / (1 + self.BUY_COST)) / total_buy_needed if total_buy_needed > 0 else 0.0
                actual_buys = actual_buys * scale_factor
                
            buy_amount = np.nansum(actual_buys)
            buy_tca = buy_amount * self.BUY_COST
            cash -= (buy_amount + buy_tca)
            total_tca_paid += buy_tca
            total_turnover += buy_amount
            
            # 6. 最终持仓清算
            holdings = holdings + actual_buys + actual_sells

        # ==========================================
        # 终极绩效核算 (纯多头真实收益)
        # ==========================================
        # 第一天没法交易，从第二天开始切片
        nav_series = pd.Series(portfolio_capital[1:], index=self.all_dates[1:])
        daily_ret_series = nav_series.pct_change().fillna(0.0)
        
        cum_ret = (1 + daily_ret_series).cumprod()
        
        years = len(daily_ret_series) / 252.0
        ann_ret = cum_ret.iloc[-1] ** (1 / years) - 1.0 if years > 0 else 0.0
        ann_vol = daily_ret_series.std() * np.sqrt(252)
        
        sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 0 else 0.0
        
        roll_max = cum_ret.cummax()
        drawdown = cum_ret / roll_max - 1.0
        mdd = drawdown.min()
        calmar = abs(ann_ret / mdd) if mdd < 0 else 0.0
        
        win_rate = (daily_ret_series > 0).sum() / len(daily_ret_series[daily_ret_series != 0])
        
        annual_turnover = (total_turnover / 2.0) / years 
        total_tca_impact = total_tca_paid / 1.0 

        logger.info("\n" + "="*70)
        logger.info("🩸 THE TRUTH MIRROR: V4 PERFECT EXECUTION TEAR SHEET 🩸")
        logger.info("="*70)
        logger.info(f"   Test Period         : {nav_series.index[0].date()} to {nav_series.index[-1].date()}")
        logger.info(f"   Total Trading Days  : {len(nav_series)}")
        logger.info("-" * 70)
        logger.info(f"   Gross Capital Ending: {cum_ret.iloc[-1]:.2f}x")
        logger.info(f"   Annualized Return   : {ann_ret:.2%}")
        logger.info(f"   Annualized Vol        : {ann_vol:.2%}")
        logger.info("-" * 70)
        logger.info(f"   Max Drawdown (MDD)  : {mdd:.2%} 🛡️")
        logger.info(f"   Sharpe Ratio          : {sharpe:.2f}")
        logger.info(f"   Calmar Ratio          : {calmar:.2f}")
        logger.info(f"   Daily Win Rate        : {win_rate:.2%}")
        logger.info("-" * 70)
        logger.info("   [FRICTION & EXECUTION METRICS]")
        logger.info(f"   Annualized Turnover : {annual_turnover:.2f}x (Single-sided)")
        logger.info(f"   Total TCA Paid      : {total_tca_impact:.2%} of initial capital")
        logger.info("="*70)

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
    PerfectLiveSimulator().run_true_simulation()