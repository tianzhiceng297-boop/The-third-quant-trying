# alpha_engine/01_build_adv_micro_factor.py

import os
import sys
import gc
import shutil
import numpy as np
import pandas as pd
from loguru import logger
import concurrent.futures
from tqdm import tqdm

class AdvMicroFactorFactory:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_lake_dir = os.path.abspath(os.path.join(self.base_dir, "..", "data_lake"))
        self.quotes_dir = os.path.join(self.data_lake_dir, "daily_quotes")
        
        self.output_dir = os.path.join(self.base_dir, "factors")
        self.tmp_dir = os.path.join(self.output_dir, "tmp_adv_micro_shards")
        
        os.makedirs(self.output_dir, exist_ok=True)
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

    def _process_single_stock(self, ts_code: str) -> str:
        quotes_path = os.path.join(self.quotes_dir, f"{ts_code}.parquet")
        tmp_shard_path = os.path.join(self.tmp_dir, f"{ts_code}.parquet")
        
        if not os.path.exists(quotes_path): return "Missing Data"

        try:
            # 提取 高、低、收、量 用于构建高阶特征
            df_q = pd.read_parquet(quotes_path, columns=['trade_date', 'close', 'high', 'low', 'vol', 'adj_factor'])
            df_q['trade_date'] = pd.to_datetime(df_q['trade_date'])
            df_q = df_q.sort_values('trade_date').drop_duplicates('trade_date').reset_index(drop=True)
            
            # 真实前复权价格
            df_q['adj_close'] = (df_q['close'] * df_q['adj_factor']).astype(np.float64)
            df_q['adj_high'] = (df_q['high'] * df_q['adj_factor']).astype(np.float64)
            df_q['adj_low'] = (df_q['low'] * df_q['adj_factor']).astype(np.float64)
            
            # ==========================================
            # 武器 1: 日内振幅黑洞 (Intraday Amplitude)
            # ==========================================
            # 振幅 = (最高 - 最低) / 昨收
            prev_close = df_q['adj_close'].shift(1)
            df_q['daily_amp'] = np.where(prev_close > 0, (df_q['adj_high'] - df_q['adj_low']) / prev_close, 0)
            # 取过去 20 天平均振幅
            df_q['amp_20d'] = df_q['daily_amp'].rolling(20).mean()
            
            # ==========================================
            # 武器 2: 量价背离 / 聪明钱 (Price-Volume Corr)
            # ==========================================
            # 过去 20 天 收盘价与成交量的皮尔逊相关系数
            # 避免全 0 或停牌导致的除零警告
            df_q['pv_corr_20d'] = df_q['adj_close'].rolling(20).corr(df_q['vol'])
            df_q['pv_corr_20d'] = df_q['pv_corr_20d'].replace([np.inf, -np.inf], np.nan).fillna(0.0)

            df_q['ts_code'] = ts_code
            df_alpha = df_q[['trade_date', 'ts_code', 'amp_20d', 'pv_corr_20d']].dropna()
            
            if df_alpha.empty: return "Empty Result"
            
            df_alpha.to_parquet(tmp_shard_path, engine='pyarrow', index=False)
            del df_q, df_alpha
            return "Success"
            
        except Exception as e:
            return f"Error: {str(e)}"

    def _vectorized_cross_sectional_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("⚡ Applying Vectorized Rank-Z Cross-Sectionally...")
        
        grouped = df.groupby('trade_date')
        
        for col in['amp_20d', 'pv_corr_20d']:
            # 强制横截面排名标准化，免疫极端异常值
            rank_col = grouped[col].rank(pct=True, ascending=True)
            mean = rank_col.groupby(df['trade_date']).transform('mean')
            std = rank_col.groupby(df['trade_date']).transform('std')
            df[col] = np.where(std > 1e-8, (rank_col - mean) / std, 0.0)
            
            del rank_col, mean, std
            gc.collect()
            
        # ==========================================
        # 合成：高阶微观 Alpha (Advanced Micro Alpha)
        # 极度惩罚高振幅 (-0.5)，极度惩罚量价高度正相关 (-0.5)
        # ==========================================
        df['adv_micro_alpha'] = -0.5 * df['amp_20d'] - 0.5 * df['pv_corr_20d']
        
        return df[['trade_date', 'ts_code', 'adv_micro_alpha']]

    def run_pipeline(self, max_workers=14):
        stock_files =[f.replace('.parquet', '') for f in os.listdir(self.quotes_dir) if f.endswith('.parquet')]
        logger.info(f"🏭 Advanced Micro Factory starting. Target: {len(stock_files)} stocks.")
        
        success_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures =[executor.submit(self._process_single_stock, code) for code in stock_files]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="[Map] Micro Shards"):
                if "Success" in future.result(): success_count += 1
                    
        if success_count == 0:
            logger.error("Pipeline Halted.")
            return
            
        gc.collect() 

        logger.info("[Reduce] Loading Panel Data via PyArrow...")
        df_panel = pd.read_parquet(self.tmp_dir, engine='pyarrow')
        df_panel = self._vectorized_cross_sectional_processing(df_panel)
        
        save_path = os.path.join(self.output_dir, "adv_micro_alpha.parquet")
        df_panel = df_panel.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
        df_panel.to_parquet(save_path, engine='pyarrow', index=False)
        
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        logger.success(f"🏁 Advanced Micro Alpha Matrix built successfully!")

if __name__ == "__main__":
    logger.add("adv_micro_factory.log", rotation="50 MB", enqueue=True)
    AdvMicroFactorFactory().run_pipeline(max_workers=14)