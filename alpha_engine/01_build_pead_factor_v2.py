# alpha_engine/01_build_pead_factor_v2.py

import os
import sys
import gc
import shutil
import numpy as np
import pandas as pd
from loguru import logger
import concurrent.futures
from tqdm import tqdm

class AlphaFactoryPro:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_lake_dir = os.path.abspath(os.path.join(self.base_dir, "..", "data_lake"))
        self.quotes_dir = os.path.join(self.data_lake_dir, "daily_quotes")
        self.fin_dir = os.path.join(self.data_lake_dir, "financials")
        
        self.output_dir = os.path.join(self.base_dir, "factors")
        # 优化点 3：引入 MapReduce 的 Disk Spooling（磁盘溢写），彻底解决内存核爆
        self.tmp_dir = os.path.join(self.output_dir, "tmp_shards")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

    def _process_single_stock_alpha(self, ts_code: str) -> str:
        """
        Map 阶段：计算单只股票的时序因子，并落盘为临时切片 (Shard)
        """
        quotes_path = os.path.join(self.quotes_dir, f"{ts_code}.parquet")
        fin_path = os.path.join(self.fin_dir, f"{ts_code}.parquet")
        tmp_shard_path = os.path.join(self.tmp_dir, f"{ts_code}.parquet")
        
        if not os.path.exists(quotes_path) or not os.path.exists(fin_path):
            return "Missing Data"

        try:
            df_q = pd.read_parquet(quotes_path, columns=['trade_date', 'close'])
            df_q['trade_date'] = pd.to_datetime(df_q['trade_date'], format='%Y%m%d')
            df_q = df_q.sort_values('trade_date').drop_duplicates('trade_date')
            df_q['mom_20d'] = df_q['close'] / df_q['close'].shift(20) - 1.0

            df_f = pd.read_parquet(fin_path)
            
            # ==========================================
            # 优化点 1：极其精准的列名硬匹配，拒绝模糊污染
            # Layer 1.x 中我们给 income 表加了 _inc 后缀
            # ==========================================
            profit_col = None
            if 'n_income_attr_p_inc' in df_f.columns:
                profit_col = 'n_income_attr_p_inc'
            elif 'n_income_attr_p' in df_f.columns: # 兼容老版本
                profit_col = 'n_income_attr_p'
                
            if not profit_col or df_f[profit_col].isna().all():
                return "No Profit Data"

            df_f = df_f[['ann_date', 'end_date', profit_col]].copy()
            df_f['ann_date'] = pd.to_datetime(df_f['ann_date'], format='%Y%m%d', errors='coerce')
            df_f['end_date'] = pd.to_datetime(df_f['end_date'], format='%Y%m%d', errors='coerce')
            
            df_f = df_f.dropna(subset=['ann_date', 'end_date']).sort_values('end_date').drop_duplicates('end_date', keep='last')
            
            df_f['profit_yoy_diff'] = df_f[profit_col] - df_f[profit_col].shift(4)
            df_f['profit_diff_std'] = df_f['profit_yoy_diff'].rolling(8).std()
            df_f['sue'] = df_f['profit_yoy_diff'] / (df_f['profit_diff_std'] + 1e-5)
            
            df_f = df_f.dropna(subset=['sue', 'ann_date']).sort_values('ann_date').drop_duplicates('ann_date', keep='last')
            if df_f.empty: return "No Valid SUE"
            
            # As-of Join: 严禁未来函数
            df_alpha = pd.merge_asof(
                df_q, df_f[['ann_date', 'sue']], 
                left_on='trade_date', right_on='ann_date', direction='backward'
            )
            
            df_alpha['ts_code'] = ts_code
            df_alpha = df_alpha[['trade_date', 'ts_code', 'mom_20d', 'sue']].dropna()
            
            if df_alpha.empty: return "Empty Result"
            
            # 落盘写入临时分片，释放内存！
            df_alpha.to_parquet(tmp_shard_path, engine='pyarrow', index=False)
            return "Success"
            
        except Exception as e:
            return f"Error: {str(e)}"

    def _vectorized_cross_sectional_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        优化点 2：彻底抛弃 apply，拥抱纯向量化 transform 矩阵运算。
        速度提升 10 - 50 倍！
        """
        logger.info("Applying Vectorized MAD & Z-Score Cross-Sectionally...")
        
        # 将 trade_date 设为 groupby 锚点
        grouped = df.groupby('trade_date')
        
        for col in['mom_20d', 'sue']:
            # 1. 向量化 MAD 去极值
            median = grouped[col].transform('median')
            abs_diff = (df[col] - median).abs()
            mad = abs_diff.groupby(df['trade_date']).transform('median')
            
            lower_bound = median - 3 * 1.4826 * mad
            upper_bound = median + 3 * 1.4826 * mad
            
            # 高效切片裁剪
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            # 2. 向量化 Z-Score 标准化
            mean = grouped[col].transform('mean')
            std = grouped[col].transform('std')
            
            # 避免除以 0 的极速 np.where 写法
            df[col] = np.where(std > 1e-8, (df[col] - mean) / std, 0.0)
            
        # 3. 向量化合成 Alpha (极速单列相加)
        df['pead_alpha'] = 0.6 * df['sue'] - 0.4 * df['mom_20d']
        return df

    def run_pipeline(self, max_workers=12):
        stock_files =[f.replace('.parquet', '') for f in os.listdir(self.quotes_dir) if f.endswith('.parquet')]
        logger.info(f"🏭 Pro Alpha Factory starting. Target: {len(stock_files)} stocks.")
        
        # ==========================================
        # 阶段一：Map 阶段 (并发时序运算并溢写磁盘)
        # ==========================================
        success_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures =[executor.submit(self._process_single_stock_alpha, code) for code in stock_files]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="[Map] Time-Series Shards"):
                if "Success" in future.result():
                    success_count += 1
                    
        if success_count == 0:
            logger.error("No valid alpha shards generated. Pipeline Halted.")
            return
            
        logger.info(f"Generated {success_count} shards. Force Garbage Collection...")
        gc.collect() # 强制回收线程池残留内存

        # ==========================================
        # 阶段二：Reduce 阶段 (流式读取合成，规避 List Concat)
        # ==========================================
        logger.info("[Reduce] Reading all shards directly into a contiguous memory block via PyArrow...")
        # pd.read_parquet 极其聪明，传给它一个目录，它会在底层用 C++ 高效组装所有 parquet
        df_panel = pd.read_parquet(self.tmp_dir, engine='pyarrow')
        logger.info(f"Loaded Panel Data Shape: {df_panel.shape}")

        # ==========================================
        # 阶段三：向量化截面运算
        # ==========================================
        df_panel = self._vectorized_cross_sectional_processing(df_panel)
        
        # 排序并落盘
        save_path = os.path.join(self.output_dir, "pead_alpha.parquet")
        df_panel = df_panel.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
        df_panel.to_parquet(save_path, engine='pyarrow', index=False)
        
        # 清理战场
        logger.info("Cleaning up temporary shards...")
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        
        logger.success(f"🏁 Vectorized Alpha Matrix built successfully!")
        logger.success(f"Output Matrix Shape: {df_panel.shape} (Rows, Columns)")
        logger.success(f"Saved to: {save_path}")

if __name__ == "__main__":
    logger.add("alpha_factory.log", rotation="50 MB", enqueue=True)
    factory = AlphaFactoryPro()
    factory.run_pipeline(max_workers=14) # I/O 密集，16核处理器可推至 14