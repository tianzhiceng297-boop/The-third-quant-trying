# alpha_engine/06_drawdown_attribution.py

import os
import sys
import numpy as np
import pandas as pd
from loguru import logger
import concurrent.futures
from tqdm import tqdm

class DrawdownAttribution:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_lake_dir = os.path.abspath(os.path.join(self.base_dir, "..", "data_lake"))
        self.quotes_dir = os.path.join(self.data_lake_dir, "daily_quotes")
        self.cal_path = os.path.join(self.data_lake_dir, "trade_calendar.parquet")
        
        self.weights_path = os.path.join(self.base_dir, "factors", "target_weights_final.parquet")

    def _load_single_price(self, file_path):
        try: return pd.read_parquet(file_path, columns=['trade_date', 'ts_code', 'close', 'adj_factor', 'amount'])
        except Exception: return pd.DataFrame()

    def run_autopsy(self):
        if not os.path.exists(self.weights_path):
            logger.critical("Order book not found! Run Layer 4 first.")
            sys.exit(1)

        logger.info("📥 1/3 Loading Market Data & Rebuilding Radar Memory...")
        df_cal = pd.read_parquet(self.cal_path)
        all_trade_dates = pd.to_datetime(df_cal['cal_date']).sort_values().unique()

        stock_files =[os.path.join(self.quotes_dir, f) for f in os.listdir(self.quotes_dir) if f.endswith('.parquet')]
        
        price_pieces =[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=14) as executor:
            futures =[executor.submit(self._load_single_price, f) for f in stock_files]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Loading Quotes"):
                res = future.result()
                if not res.empty: price_pieces.append(res)

        df_price = pd.concat(price_pieces, ignore_index=True)
        df_price['trade_date'] = pd.to_datetime(df_price['trade_date'])
        df_price['adj_close'] = df_price['close'] * df_price['adj_factor']

        price_matrix = df_price.pivot(index='trade_date', columns='ts_code', values='adj_close').reindex(all_trade_dates)
        amt_matrix = df_price.pivot(index='trade_date', columns='ts_code', values='amount').reindex(all_trade_dates)
        
        market_ret = (price_matrix / price_matrix.shift(1) - 1.0).mean(axis=1)
        
        # 重建雷达核心指标
        vol_20 = market_ret.rolling(20).std()
        vol_60 = market_ret.rolling(60).std()
        div_up = vol_20 > vol_60
        
        market_amt = amt_matrix.sum(axis=1)
        amt_20 = market_amt.rolling(20).mean()
        amt_60 = market_amt.rolling(60).mean()
        liq_up = amt_20 > amt_60
        
        pro_up = market_ret.rolling(20).mean() > 0
        
        crash_ratio = ((price_matrix / price_matrix.shift(1) - 1.0) < -0.08).sum(axis=1) / price_matrix.notna().sum(axis=1)

        logger.info("⚔️ 2/3 Calculating Strategy P&L and Locating Max Drawdown...")
        df_weights = pd.read_parquet(self.weights_path)
        df_weights['trade_date'] = pd.to_datetime(df_weights['trade_date'])
        
        buy_matrix = price_matrix.shift(-1)
        sell_matrix = price_matrix.shift(-2)
        fwd_ret_matrix = (sell_matrix / buy_matrix) - 1.0
        
        df_fwd = fwd_ret_matrix.stack().reset_index()
        df_fwd.columns =['trade_date', 'ts_code', 'fwd_ret_1d']
        
        df_sim = pd.merge(df_weights, df_fwd, on=['trade_date', 'ts_code'], how='inner')
        df_sim['fwd_ret_1d'] = df_sim['fwd_ret_1d'].clip(-0.21, 0.21)
        
        # 【修复1】：使用正确的 exec_weight 计算收益
        df_sim['ret_contrib'] = df_sim['exec_weight'] * df_sim['fwd_ret_1d']
        
        daily_portfolio_ret = df_sim.groupby('trade_date')['ret_contrib'].sum()
        daily_portfolio_ret = daily_portfolio_ret.reindex(all_trade_dates).fillna(0.0)
        
        # 【修复2】：使用 exec_weight 统计每日仓位
        daily_exposure = df_weights.groupby('trade_date')['exec_weight'].sum().reindex(all_trade_dates).fillna(0.0)
        
        cum_ret = (1 + daily_portfolio_ret).cumprod()
        roll_max = cum_ret.cummax()
        drawdown = cum_ret / roll_max - 1.0
        
        mdd_end_date = drawdown.idxmin()
        mdd_start_date = cum_ret.loc[:mdd_end_date].idxmax()
        mdd_value = drawdown.min()

        logger.info("\n" + "="*80)
        logger.info(f"🚨 MAXIMUM DRAWDOWN LOCATED: {mdd_value:.2%}")
        logger.info(f"   Peak Date  : {mdd_start_date.date()}")
        logger.info(f"   Trough Date: {mdd_end_date.date()}")
        logger.info("="*80)

        logger.info("🔬 3/3 Generating Black-box Flight Recorder Log (Peak to Trough + 5 Days)...")
        
        inspect_start = mdd_start_date
        inspect_end = mdd_end_date + pd.Timedelta(days=10) 
        mask = (all_trade_dates >= inspect_start) & (all_trade_dates <= inspect_end)
        dates_to_inspect = all_trade_dates[mask]
        
        print(f"\n{'Date':<12} | {'Mkt Ret':<8} | {'Str Ret':<8} | {'Exp%':<6} | {'Div':<5} | {'Liq':<5} | {'Pro':<5} | {'Crash%':<7} | {'Radar State':<22} | {'System Execution State'}")
        print("-" * 115)
        
        for d in dates_to_inspect:
            m_ret = market_ret.loc[d] if pd.notna(market_ret.loc[d]) else 0
            s_ret = daily_portfolio_ret.loc[d]
            exp = daily_exposure.loc[d]
            
            div = "UP" if div_up.loc[d] else "DN"
            liq = "UP" if liq_up.loc[d] else "DN"
            pro = "UP" if pro_up.loc[d] else "DN"
            cr = crash_ratio.loc[d] if pd.notna(crash_ratio.loc[d]) else 0
            
            state = "UNKNOWN"
            if not div_up.loc[d] and liq_up.loc[d]: state = "S1_RES_BULL"
            elif not div_up.loc[d] and not liq_up.loc[d]: state = "S2_RES_BEAR"
            elif div_up.loc[d] and liq_up.loc[d] and pro_up.loc[d]: state = "S3_DIV_BULL"
            elif div_up.loc[d] and not liq_up.loc[d] and pro_up.loc[d]: state = "S4_EXH_BULL"
            elif div_up.loc[d] and not liq_up.loc[d] and not pro_up.loc[d]: state = "S5_CONT_DECLINE"
            elif div_up.loc[d] and liq_up.loc[d] and not pro_up.loc[d]: state = "S6_VOL_DECLINE"
            
            # 【修复3】：直接读取实盘订单簿里存好的 filtered_state
            today_orders = df_weights[df_weights['trade_date'] == d]
            if not today_orders.empty:
                sys_state = today_orders['filtered_state'].iloc[0]
            else:
                sys_state = "CASH (0%)"
            
            print(f"{d.date().strftime('%Y-%m-%d'):<12} | {m_ret:>7.2%} | {s_ret:>7.2%} | {exp:>5.1%} | {div:<5} | {liq:<5} | {pro:<5} | {cr:>6.2%} | {state:<22} | {sys_state}")

        print("="*115)

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
    DrawdownAttribution().run_autopsy()