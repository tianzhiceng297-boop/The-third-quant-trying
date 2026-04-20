# alpha_engine/01_build_qarp_factor_v2_pro.py

import os
import sys
import gc
import shutil
import numpy as np
import pandas as pd
from loguru import logger
import concurrent.futures
from tqdm import tqdm

class QARPFactoryElite:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_lake_dir = os.path.abspath(os.path.join(self.base_dir, "..", "data_lake"))
        self.quotes_dir = os.path.join(self.data_lake_dir, "daily_quotes")
        self.fin_dir = os.path.join(self.data_lake_dir, "financials")
        self.basics_path = os.path.join(self.data_lake_dir, "stock_basics.parquet")
        
        self.output_dir = os.path.join(self.base_dir, "factors")
        self.tmp_dir = os.path.join(self.output_dir, "tmp_qarp_shards")
        
        os.makedirs(self.output_dir, exist_ok=True)
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

    def _process_single_stock_alpha(self, ts_code: str) -> str:
        quotes_path = os.path.join(self.quotes_dir, f"{ts_code}.parquet")
        fin_path = os.path.join(self.fin_dir, f"{ts_code}.parquet")
        tmp_shard_path = os.path.join(self.tmp_dir, f"{ts_code}.parquet")
        
        if not os.path.exists(quotes_path) or not os.path.exists(fin_path):
            return "Missing Data"

        try:
            df_q = pd.read_parquet(quotes_path, columns=['trade_date', 'close'])
            df_q['trade_date'] = pd.to_datetime(df_q['trade_date'], format='%Y%m%d')
            df_q = df_q.sort_values('trade_date').drop_duplicates('trade_date')

            df_f = pd.read_parquet(fin_path)
            
            # ==========================================
            # 审计修复 1：彻底消灭季节性陷阱
            # 寻找 roe_yearly (年化 ROE)，而不是累计 roe
            # ==========================================
            roe_col = next((c for c in df_f.columns if 'roe_yearly' in c), None)
            if not roe_col: # 如果没找到年化ROE，退而求其次找基础ROE (可能包含季节性)
                roe_col = next((c for c in df_f.columns if c.startswith('roe')), None)
                
            bps_col = next((c for c in df_f.columns if c.startswith('bps')), None)
            
            if not roe_col or not bps_col:
                return "Missing Quality/Value Metrics"
                
            df_f = df_f.dropna(subset=['ann_date', roe_col, bps_col])
            if df_f.empty: return "No Valid Financials"

            df_f = df_f[['ann_date', roe_col, bps_col]].copy()
            df_f['ann_date'] = pd.to_datetime(df_f['ann_date'], format='%Y%m%d', errors='coerce')
            df_f = df_f.dropna(subset=['ann_date']).sort_values('ann_date').drop_duplicates('ann_date', keep='last')
            
            # As-of Join 穿越时间结界
            df_alpha = pd.merge_asof(
                df_q, df_f, 
                left_on='trade_date', right_on='ann_date', direction='backward'
            )
            
            df_alpha['bp'] = df_alpha[bps_col] / (df_alpha['close'] + 1e-5)
            df_alpha['roe_ann'] = df_alpha[roe_col]
            
            df_alpha['ts_code'] = ts_code
            df_alpha = df_alpha[['trade_date', 'ts_code', 'roe_ann', 'bp']].dropna()
            
            if df_alpha.empty: return "Empty Result"
            
            df_alpha.to_parquet(tmp_shard_path, engine='pyarrow', index=False)
            del df_q, df_f, df_alpha
            return "Success"
            
        except Exception as e:
            return f"Error: {str(e)}"

    def _vectorized_cross_sectional_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        # ==========================================
        # 审计修复 2：引入行业中性化截面打分 (Industry-Neutral Z-Score)
        # ==========================================
        logger.info("🔗 Loading Industry Classifications for Neutralization...")
        df_basics = pd.read_parquet(self.basics_path)[['ts_code', 'industry']].fillna('Unknown')
        df = pd.merge(df, df_basics, on='ts_code', how='left')
        
        logger.info("⚡ Applying Industry-Neutral Vectorized MAD & Z-Score...")
        
        # 核心魔法：我们在 groupby 中加入了 'industry'！
        # 这意味着：银行只和银行比 BP，半导体只和半导体比 BP。
        grouped_ind = df.groupby(['trade_date', 'industry'])
        
        for col in['roe_ann', 'bp']:
            # 行业内 MAD 去极值
            median = grouped_ind[col].transform('median')
            abs_diff = (df[col] - median).abs()
            mad = abs_diff.groupby([df['trade_date'], df['industry']]).transform('median')
            
            lower_bound = median - 3 * 1.4826 * mad
            upper_bound = median + 3 * 1.4826 * mad
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            # 行业内 Z-Score 标准化
            mean = grouped_ind[col].transform('mean')
            std = grouped_ind[col].transform('std')
            df[col] = np.where(std > 1e-8, (df[col] - mean) / std, 0.0)
            
            del median, abs_diff, mad, lower_bound, upper_bound, mean, std
            gc.collect()
            
        # ==========================================
        # 终极 QARP 因子：经过行业中性化的 (50% 高质量 + 50% 便宜估值)
        # ==========================================
        df['qarp_alpha'] = 0.5 * df['roe_ann'] + 0.5 * df['bp']
        
        df = df[['trade_date', 'ts_code', 'qarp_alpha']]
        return df

    def run_pipeline(self, max_workers=14):
        stock_files =[f.replace('.parquet', '') for f in os.listdir(self.quotes_dir) if f.endswith('.parquet')]
        logger.info(f"🏭 Elite QARP Factory starting. Target: {len(stock_files)} stocks.")
        
        success_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures =[executor.submit(self._process_single_stock_alpha, code) for code in stock_files]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="[Map] Time-Series"):
                if "Success" in future.result(): success_count += 1
                    
        if success_count == 0:
            logger.error("Pipeline Halted.")
            return
            
        logger.info("Force GC...")
        gc.collect() 

        logger.info("[Reduce] Loading Panel Data into contiguous memory...")
        df_panel = pd.read_parquet(self.tmp_dir, engine='pyarrow')

        df_panel = self._vectorized_cross_sectional_processing(df_panel)
        
        save_path = os.path.join(self.output_dir, "qarp_alpha.parquet")
        df_panel = df_panel.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
        df_panel.to_parquet(save_path, engine='pyarrow', index=False)
        
        logger.info("🧹 Cleaning up shards...")
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        
        logger.success(f"🏁 Industry-Neutral QARP Alpha Matrix built successfully!")

if __name__ == "__main__":
    logger.add("qarp_factory.log", rotation="50 MB", enqueue=True)
    factory = QARPFactoryElite()
    factory.run_pipeline(max_workers=14)