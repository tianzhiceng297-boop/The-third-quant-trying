# alpha_engine/01_build_microstructure_factor_v6.py

import os
import sys
import gc
import shutil
import numpy as np
import pandas as pd
from loguru import logger
import concurrent.futures
from tqdm import tqdm

class MicrostructureFactoryPro:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_lake_dir = os.path.abspath(os.path.join(self.base_dir, "..", "data_lake"))
        self.quotes_dir = os.path.join(self.data_lake_dir, "daily_quotes")
        
        self.output_dir = os.path.join(self.base_dir, "factors")
        self.tmp_dir = os.path.join(self.output_dir, "tmp_micro_shards")
        
        os.makedirs(self.output_dir, exist_ok=True)
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

    def _process_single_stock_price(self, ts_code: str) -> str:
        quotes_path = os.path.join(self.quotes_dir, f"{ts_code}.parquet")
        tmp_shard_path = os.path.join(self.tmp_dir, f"{ts_code}.parquet")
        
        if not os.path.exists(quotes_path):
            return "Missing Data"

        try:
            df_q = pd.read_parquet(quotes_path, columns=['trade_date', 'close', 'adj_factor', 'amount'])
            df_q['trade_date'] = pd.to_datetime(df_q['trade_date'], format='%Y%m%d')
            df_q = df_q.sort_values('trade_date').drop_duplicates('trade_date').reset_index(drop=True)
            
            # 审计修复 3: 严厉的零成交额过滤 (防除零黑洞)
            # 将 amount <= 0 的视为缺失值，防止停牌或一字死板导致的分母陷阱
            df_q['amount'] = np.where(df_q['amount'] <= 0, np.nan, df_q['amount'])
            
            df_q['adj_close'] = (df_q['close'] * df_q['adj_factor']).astype(np.float64)
            df_q['ret_1d'] = df_q['adj_close'].pct_change()
            
            # 1. 散户博彩偏好 (Bali's MAX Factor)
            df_q['max_ret_20d'] = df_q['ret_1d'].rolling(20).max().astype(np.float64)
            
            # 2. Amihud 非流动性溢价
            # 乘以 10^8 放大数值至合理区间，并使用 np.log1p 进行对数压缩，压制极端的右尾偏态
            raw_amihud = df_q['ret_1d'].abs() / df_q['amount']
            df_q['amihud_daily'] = np.log1p(raw_amihud * 1e8) 
            df_q['amihud_20d'] = df_q['amihud_daily'].rolling(20).mean().astype(np.float64)

            df_q['ts_code'] = ts_code
            df_alpha = df_q[['trade_date', 'ts_code', 'max_ret_20d', 'amihud_20d']].dropna()
            
            if df_alpha.empty: return "Empty Result"
            
            df_alpha.to_parquet(tmp_shard_path, engine='pyarrow', index=False)
            del df_q, df_alpha
            return "Success"
            
        except Exception as e:
            return f"Error: {str(e)}"

    def _vectorized_cross_sectional_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("⚡ Applying Rank-Z Standardization Cross-Sectionally...")
        
        # 审计修复 1 & 2：彻底抛弃绝对值 MAD！采用机构级 Rank-Z (百分位标准化)
        # 将偏态分布强行转化为绝对均匀分布，彻底免疫量纲冲突！
        for col in['max_ret_20d', 'amihud_20d']:
            # 1. 横截面百分比排名 (0.0 到 1.0)
            rank_col = df.groupby('trade_date')[col].rank(pct=True, ascending=True)
            
            # 2. 对百分位进行 Z-Score 标准化
            # 此时的均值必然是 0.5，标准差固定，彻底消除一切长尾离群值！
            mean = rank_col.groupby(df['trade_date']).transform('mean')
            std = rank_col.groupby(df['trade_date']).transform('std')
            
            # 直接覆盖原列，输出极其纯净、正态、正交的特征向量
            df[col] = np.where(std > 1e-8, (rank_col - mean) / std, 0.0)
            
        # 注意：我们不再进行线性相加！
        # 将独立的 max_ret_20d 和 amihud_20d 直接输出。
        # 让未来的 LightGBM 自己去判断：当 PEAD 很高时，高 MAX 意味着突破；当 PEAD 很低时，高 MAX 意味着游资炒作彩票！
        
        df = df[['trade_date', 'ts_code', 'max_ret_20d', 'amihud_20d']]
        return df

    def run_pipeline(self, max_workers=14):
        stock_files =[f.replace('.parquet', '') for f in os.listdir(self.quotes_dir) if f.endswith('.parquet')]
        logger.info(f"🏭 Microstructure Factory V6 starting. Target: {len(stock_files)} stocks.")
        
        success_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures =[executor.submit(self._process_single_stock_price, code) for code in stock_files]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="[Map] Micro Shards"):
                if "Success" in future.result():
                    success_count += 1
                    
        if success_count == 0:
            logger.error("Pipeline Halted.")
            return
            
        gc.collect() 

        logger.info("[Reduce] Loading Panel Data via PyArrow...")
        df_panel = pd.read_parquet(self.tmp_dir, engine='pyarrow')

        df_panel = self._vectorized_cross_sectional_processing(df_panel)
        
        save_path = os.path.join(self.output_dir, "micro_features.parquet")
        df_panel = df_panel.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
        df_panel.to_parquet(save_path, engine='pyarrow', index=False)
        
        logger.info("🧹 Cleaning up shards...")
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        
        logger.success(f"🏁 Microstructure Features Matrix built successfully!")
        logger.success(f"Saved to: {save_path}")

if __name__ == "__main__":
    logger.add("micro_factory.log", rotation="50 MB", enqueue=True)
    factory = MicrostructureFactoryPro()
    factory.run_pipeline(max_workers=14)