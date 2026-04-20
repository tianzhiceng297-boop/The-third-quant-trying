# alpha_engine/04_classic_linear_optimizer.py

import os
import sys
import numpy as np
import pandas as pd
from loguru import logger
import concurrent.futures
from tqdm import tqdm
from functools import reduce

class ClassicLinearOptimizer:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_lake_dir = os.path.abspath(os.path.join(self.base_dir, "..", "data_lake"))
        self.quotes_dir = os.path.join(self.data_lake_dir, "daily_quotes")
        self.cal_path = os.path.join(self.data_lake_dir, "trade_calendar.parquet")
        self.names_path = os.path.join(self.data_lake_dir, "unified_name_history.parquet")
        
        self.factors_dir = os.path.join(self.base_dir, "factors")
        self.output_path = os.path.join(self.factors_dir, "target_weights_final.parquet")
        
        # 铁律：10个交易日调仓，持有 100 只股票分散风险！
        self.REBALANCE_FREQ = 10
        self.TARGET_N_STOCKS = 100

    def _load_single_price(self, file_path):
        try: return pd.read_parquet(file_path, columns=['trade_date', 'ts_code', 'close', 'adj_factor', 'amount'])
        except: return pd.DataFrame()

    def build_market_proxy(self, max_workers=14):
        logger.info("📡 1/4 Loading Price Universe...")
        df_cal = pd.read_parquet(self.cal_path)
        all_dates = pd.to_datetime(df_cal['cal_date']).sort_values().unique()

        stock_files =[os.path.join(self.quotes_dir, f) for f in os.listdir(self.quotes_dir) if f.endswith('.parquet')]
        price_pieces =[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures =[executor.submit(self._load_single_price, f) for f in stock_files]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Loading Quotes"):
                res = future.result()
                if not res.empty: price_pieces.append(res)

        df_price = pd.concat(price_pieces, ignore_index=True)
        df_price['trade_date'] = pd.to_datetime(df_price['trade_date'])
        
        # 计算 20 日均成交额，用于屏蔽微盘股
        df_price = df_price.sort_values(['ts_code', 'trade_date'])
        df_price['amt_ma20'] = df_price.groupby('ts_code')['amount'].transform(lambda x: x.rolling(20).mean())
        
        price_matrix = (df_price['close'] * df_price['adj_factor']).groupby([df_price['trade_date'], df_price['ts_code']]).first().unstack().reindex(all_dates)
        
        return price_matrix, df_price[['trade_date', 'ts_code', 'amt_ma20']], all_dates

    def execute_pipeline(self):
        price_matrix, df_amt, all_dates = self.build_market_proxy()
        
        # ---------------------------------------------------------
        # 🛡️ 极简防线：60日均线趋势择时 (大道至简的止损)
        # ---------------------------------------------------------
        logger.info("🛡️ 2/4 Deploying 60-Day Simple Trend Filter...")
        market_index = price_matrix.mean(axis=1) # 等权全A指数
        ma_60 = market_index.rolling(60).mean()
        
        # 规则：大盘跌破 60 日线，认定为熊市，剥夺开仓权！
        bear_market_mode = market_index < ma_60

        # ---------------------------------------------------------
        # 💡 玻璃盒合并：经典的线性多因子打分 (Linear Scoring)
        # ---------------------------------------------------------
        logger.info("⚖️ 3/4 Assembling the Glass-box Linear Factor Model...")
        
        # 载入我们打磨好的纯净因子 (全部经过 Rank-Z 标准化，量纲绝对公平)
        df_pead = pd.read_parquet(os.path.join(self.factors_dir, "pead_alpha.parquet"))
        df_rev = pd.read_parquet(os.path.join(self.factors_dir, "reversal_alpha.parquet"))
        df_fund = pd.read_parquet(os.path.join(self.factors_dir, "pure_fundamentals.parquet"))
        df_adv_micro = pd.read_parquet(os.path.join(self.factors_dir, "adv_micro_alpha.parquet"))
        data_frames =[df_pead, df_rev, df_fund[['trade_date', 'ts_code', 'pure_bp']], df_adv_micro]
        df_master = reduce(lambda left, right: pd.merge(left, right, on=['trade_date', 'ts_code'], how='inner'), data_frames)
        df_master = pd.merge(df_master, df_amt, on=['trade_date', 'ts_code'], how='inner')

        # 挂载 PIT 历史名称，排雷 ST
        df_names = pd.read_parquet(self.names_path)
        df_names['start_date'] = pd.to_datetime(df_names['start_date'])
        df_names = df_names.dropna(subset=['start_date']).sort_values('start_date')
        df_master = df_master.sort_values('trade_date')
        df_master = pd.merge_asof(
            df_master, df_names[['start_date', 'ts_code', 'name']], 
            left_on='trade_date', right_on='start_date', by='ts_code', direction='backward'
        )

        # 物理隔离垃圾股 (ST & 微盘)
        mask_st = df_master['name'].str.contains('ST|PT|退', na=False)
        df_master = df_master[~mask_st].copy()
        thresholds = df_master.groupby('trade_date')['amt_ma20'].transform(lambda x: x.quantile(0.20))
        df_master = df_master[df_master['amt_ma20'] > thresholds].copy()
        
        # ==========================================
        # 🌟 终极打分公式 (The Magic Formula)
        # 40% 纯低估值 + 40% 业绩爆发 + 20% 低波反转防追高
        # ==========================================
        df_master['composite_score'] = (
            0.30 * df_master['pure_bp'] + 
            0.30 * df_master['pead_alpha'] + 
            0.15 * df_master['reversal_alpha'] +
            0.25 * df_master['adv_micro_alpha']  # 注入全新动力！
        )

        logger.info("🔐 4/4 Executing 100-Stock Diversification & Hysteresis Lock...")
        score_mat = df_master.pivot(index='trade_date', columns='ts_code', values='composite_score').reindex(all_dates)
        
        target_w_mat = pd.DataFrame(0.0, index=all_dates, columns=score_mat.columns)
        last_weights = pd.Series(0.0, index=score_mat.columns)
        days_since_rebalance = 0
        rebalance_count = 0
        
        for i in tqdm(range(len(all_dates)), desc="Generating Linear Orders"):
            is_bear = bear_market_mode.iloc[i]
            
            if is_bear:
                # 熊市断腕：清仓保命，留住现金
                last_weights[:] = 0.0
                days_since_rebalance = 0
            else:
                # 正常牛市/震荡市：每双周开启一次选股雷达
                if days_since_rebalance == 0 or last_weights.sum() == 0.0:
                    today_scores = score_mat.iloc[i].dropna()
                    
                    if len(today_scores) >= self.TARGET_N_STOCKS:
                        # 强行打分排序，第一名是最好的
                        ranks = today_scores.rank(method='first', ascending=False)
                        
                        # =====================================================
                        # 🛡️ 机构级双轨宽容带 (Hysteresis Buffer) 压制换手率
                        # 进前 100 名才买；但只要没掉出前 150 名，坚决不卖！
                        # =====================================================
                        held_stocks = last_weights[last_weights > 0].index.tolist()
                        keep_stocks =[s for s in held_stocks if s in ranks.index and ranks[s] <= 150]
                        num_needed = self.TARGET_N_STOCKS - len(keep_stocks)
                        
                        if num_needed > 0:
                            potential_new = ranks.drop(labels=keep_stocks, errors='ignore').sort_values()
                            new_stocks = potential_new.head(num_needed).index.tolist()
                        else:
                            new_stocks =[]
                            
                        final_portfolio = keep_stocks + new_stocks
                        last_weights[:] = 0.0
                        if len(final_portfolio) > 0:
                            # 🚨 物理防线：最高只买 98% 的仓位，永远预留 2% 的现金用来交印花税和滑点！
                            last_weights[final_portfolio] = 0.98 / len(final_portfolio)
                            rebalance_count += 1
                            
                days_since_rebalance = (days_since_rebalance + 1) % self.REBALANCE_FREQ
            
            # 每日继承锁定仓位 (0摩擦)
            target_w_mat.iloc[i] = last_weights

        df_orders = target_w_mat.stack().reset_index()
        df_orders.columns =['trade_date', 'ts_code', 'exec_weight']
        df_orders = df_orders[df_orders['exec_weight'] > 0].copy()
        
        states = pd.Series("S_HOLD", index=all_dates)
        states[bear_market_mode] = "BEAR_HALT"
        df_orders = pd.merge(df_orders, pd.DataFrame({'trade_date': all_dates, 'filtered_state': states}), on='trade_date', how='left')

        df_orders.to_parquet(self.output_path, engine='pyarrow', index=False)
        
        logger.success("="*60)
        logger.success("✅ CLASSIC LINEAR OPTIMIZER (GLASS-BOX) COMPLETE!")
        logger.success(f"📊 Rebalances Triggered : {rebalance_count} times")
        logger.success(f"🛡️ Bear Market Days Avoided: {bear_market_mode.sum()} days")
        logger.success("="*60)

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
    ClassicLinearOptimizer().execute_pipeline()