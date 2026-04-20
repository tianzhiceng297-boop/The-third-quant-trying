# alpha_engine/01_build_reversal_factor_v3.py

import os
import sys
import gc
import shutil
import numpy as np
import pandas as pd
from loguru import logger
import concurrent.futures
from tqdm import tqdm

class ReversalFactorFactoryPro:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_lake_dir = os.path.abspath(os.path.join(self.base_dir, "..", "data_lake"))
        self.quotes_dir = os.path.join(self.data_lake_dir, "daily_quotes")
        
        self.output_dir = os.path.join(self.base_dir, "factors")
        self.tmp_dir = os.path.join(self.output_dir, "tmp_reversal_shards")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 审计修复 2：启动前强制清理残留的僵尸切片 (Pre-flight Cleanup)
        if os.path.exists(self.tmp_dir):
            logger.info("🧹 Pre-flight: Cleaning up residual shards from previous crashed runs...")
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

    def _process_single_stock_price(self, ts_code: str) -> str:
        quotes_path = os.path.join(self.quotes_dir, f"{ts_code}.parquet")
        tmp_shard_path = os.path.join(self.tmp_dir, f"{ts_code}.parquet")
        
        if not os.path.exists(quotes_path):
            return "Missing Data"

        try:
            # 审计修复 3：引入 amount (成交额)，为流动性过滤储备弹药
            df_q = pd.read_parquet(quotes_path, columns=['trade_date', 'close', 'adj_factor', 'amount'])
            
            # Pandas 3.0 规范：严谨的排序与去重
            df_q['trade_date'] = pd.to_datetime(df_q['trade_date'], format='%Y%m%d')
            df_q = df_q.sort_values('trade_date').drop_duplicates('trade_date').reset_index(drop=True)
            
            # 审计修复 1：显式声明 Float64，迎合 Arrow 后端，防止精度隐式丢失
            df_q['adj_close'] = (df_q['close'] * df_q['adj_factor']).astype(np.float64)
            
            # Pandas 3.0 规范：移除废弃的 fill_method 参数
            df_q['ret_1d'] = df_q['adj_close'].pct_change()
            df_q['ret_20d'] = df_q['adj_close'].pct_change(periods=20)
            
            df_q['vol_20d'] = (df_q['ret_1d'].rolling(20).std() * np.sqrt(252)).astype(np.float64)
            
            # 流动性弹药：20日均成交额
            df_q['amt_ma20'] = df_q['amount'].rolling(20).mean().astype(np.float64)

            df_q['ts_code'] = ts_code
            df_alpha = df_q[['trade_date', 'ts_code', 'ret_20d', 'vol_20d', 'amt_ma20']].dropna()
            
            if df_alpha.empty: return "Empty Result"
            
            df_alpha.to_parquet(tmp_shard_path, engine='pyarrow', index=False)
            
            # 内存防爆：显式删除 DataFrame
            del df_q, df_alpha
            return "Success"
            
        except Exception as e:
            return f"Error: {str(e)}"

    def _vectorized_cross_sectional_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("🛡️ Applying Liquidity Filter (Killing Zombie Stocks)...")
        
        # 审计修复 3：流动性过滤 (剔除全市场成交额垫底的 10% 僵尸股)
        # 僵尸股的波动率为 0，如果不剔除，会骗取极高的低波反转得分！
        thresholds = df.groupby('trade_date')['amt_ma20'].transform(lambda x: x.quantile(0.10))
        df = df[df['amt_ma20'] > thresholds].copy()
        df = df.reset_index(drop=True) # 重置索引，防止内存碎片
        
        logger.info("⚡ Applying Vectorized MAD & Z-Score Cross-Sectionally...")
        grouped = df.groupby('trade_date')
        
        for col in['ret_20d', 'vol_20d']:
            median = grouped[col].transform('median')
            abs_diff = (df[col] - median).abs()
            mad = abs_diff.groupby(df['trade_date']).transform('median')
            
            lower_bound = median - 3 * 1.4826 * mad
            upper_bound = median + 3 * 1.4826 * mad
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            mean = grouped[col].transform('mean')
            std = grouped[col].transform('std')
            df[col] = np.where(std > 1e-8, (df[col] - mean) / std, 0.0)
            
            # 审计修复 2 (OOM防御)：及时清理中间巨型张量
            del median, abs_diff, mad, lower_bound, upper_bound, mean, std
            gc.collect()
            
        # ==========================================
        # 核心物理逻辑：高质量防御性反转 (Defensive Reversal Alpha)
        # ==========================================
        df['reversal_alpha'] = -0.7 * df['ret_20d'] - 0.3 * df['vol_20d']
        
        # 清理已不需要的特征列，极致压缩落盘体积
        df = df[['trade_date', 'ts_code', 'reversal_alpha']]
        
        return df

    def run_pipeline(self, max_workers=14):
        stock_files =[f.replace('.parquet', '') for f in os.listdir(self.quotes_dir) if f.endswith('.parquet')]
        logger.info(f"🏭 Low-Vol Reversal Factory V3 starting. Target: {len(stock_files)} stocks.")
        
        success_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures =[executor.submit(self._process_single_stock_price, code) for code in stock_files]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="[Map] Time-Series Shards"):
                if "Success" in future.result():
                    success_count += 1
                    
        if success_count == 0:
            logger.error("No valid alpha shards generated. Pipeline Halted.")
            return
            
        logger.info(f"Generated {success_count} shards. Force Garbage Collection...")
        gc.collect() 

        logger.info("[Reduce] Reading all shards directly into contiguous memory via PyArrow...")
        df_panel = pd.read_parquet(self.tmp_dir, engine='pyarrow')
        logger.info(f"Loaded Panel Data Shape: {df_panel.shape}")

        df_panel = self._vectorized_cross_sectional_processing(df_panel)
        
        save_path = os.path.join(self.output_dir, "reversal_alpha.parquet")
        df_panel = df_panel.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
        df_panel.to_parquet(save_path, engine='pyarrow', index=False)
        
        logger.info("🧹 Post-flight: Cleaning up temporary shards...")
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        
        logger.success(f"🏁 Defensive Reversal Alpha Matrix built successfully!")
        logger.success(f"Saved to: {save_path}")

if __name__ == "__main__":
    logger.add("reversal_factory.log", rotation="50 MB", enqueue=True)
    factory = ReversalFactorFactoryPro()
    factory.run_pipeline(max_workers=14)