# alpha_engine/01_build_pure_fundamentals_v2_pro.py

import os
import sys
import gc
import shutil
import numpy as np
import pandas as pd
from loguru import logger
import concurrent.futures
from tqdm import tqdm

class PureFundamentalsFactoryPro:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_lake_dir = os.path.abspath(os.path.join(self.base_dir, "..", "data_lake"))
        self.quotes_dir = os.path.join(self.data_lake_dir, "daily_quotes")
        self.fin_dir = os.path.join(self.data_lake_dir, "financials")
        self.basics_path = os.path.join(self.data_lake_dir, "stock_basics.parquet")
        
        self.output_dir = os.path.join(self.base_dir, "factors")
        self.tmp_dir = os.path.join(self.output_dir, "tmp_pure_fund_shards")
        
        os.makedirs(self.output_dir, exist_ok=True)
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

    def _process_single_stock_fundamentals(self, ts_code: str) -> str:
        """
        Map 阶段：单只股票的量价与财务 As-of Join，落盘临时切片防 OOM
        """
        quotes_path = os.path.join(self.quotes_dir, f"{ts_code}.parquet")
        fin_path = os.path.join(self.fin_dir, f"{ts_code}.parquet")
        tmp_shard_path = os.path.join(self.tmp_dir, f"{ts_code}.parquet")
        
        if not os.path.exists(quotes_path) or not os.path.exists(fin_path):
            return "Missing Data"

        try:
            # 1. 极速读取日线
            df_q = pd.read_parquet(quotes_path, columns=['trade_date', 'close'])
            df_q['trade_date'] = pd.to_datetime(df_q['trade_date'], format='%Y%m%d')
            df_q = df_q.sort_values('trade_date').drop_duplicates('trade_date')

            # 2. 读取财务宽表，寻找核心字段
            df_f = pd.read_parquet(fin_path)
            
            sales_yoy_col = next((c for c in df_f.columns if 'q_sales_yoy' in c or 'tr_yoy' in c or 'or_yoy' in c), None)
            profit_yoy_col = next((c for c in df_f.columns if 'q_netprofit_yoy' in c or 'netprofit_yoy' in c), None)
            bps_col = next((c for c in df_f.columns if c.startswith('bps')), None)
            eps_col = next((c for c in df_f.columns if c.startswith('eps')), None)
            
            if not (bps_col and eps_col and sales_yoy_col and profit_yoy_col):
                return "Missing Core Financial Metrics"
                
            df_f = df_f.dropna(subset=['ann_date', bps_col, eps_col, sales_yoy_col, profit_yoy_col])
            if df_f.empty: return "No Valid Financials"

            df_f = df_f[['ann_date', bps_col, eps_col, sales_yoy_col, profit_yoy_col]].copy()
            df_f['ann_date'] = pd.to_datetime(df_f['ann_date'], format='%Y%m%d', errors='coerce')
            df_f = df_f.dropna(subset=['ann_date']).sort_values('ann_date').drop_duplicates('ann_date', keep='last')
            
            # 3. As-of Join 穿越时间结界
            df_alpha = pd.merge_asof(
                df_q, df_f, 
                left_on='trade_date', right_on='ann_date', direction='backward'
            )
            
            # 4. 计算动态纯估值因子
            df_alpha['pure_bp'] = df_alpha[bps_col] / (df_alpha['close'] + 1e-5)
            df_alpha['pure_ep'] = df_alpha[eps_col] / (df_alpha['close'] + 1e-5)
            df_alpha['sales_yoy'] = df_alpha[sales_yoy_col]
            df_alpha['profit_yoy'] = df_alpha[profit_yoy_col]
            
            df_alpha['ts_code'] = ts_code
            df_alpha = df_alpha[['trade_date', 'ts_code', 'pure_bp', 'pure_ep', 'sales_yoy', 'profit_yoy']].dropna()
            
            if df_alpha.empty: return "Empty Result"
            
            df_alpha.to_parquet(tmp_shard_path, engine='pyarrow', index=False)
            del df_q, df_f, df_alpha
            return "Success"
            
        except Exception as e:
            return f"Error: {str(e)}"

    def _vectorized_cross_sectional_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce 阶段：全市场【行业中性化】向量化打分
        """
        # ==========================================
        # 审计修复 3：引入行业中性化，彻底消灭共线性与偏科陷阱
        # ==========================================
        logger.info("🔗 Loading Industry Classifications for Neutralization...")
        df_basics = pd.read_parquet(self.basics_path)[['ts_code', 'industry']].fillna('Unknown')
        df = pd.merge(df, df_basics, on='ts_code', how='left')
        
        logger.info("⚡ Applying Industry-Neutral Vectorized Rank-Z Standardization...")
        
        # 核心魔法：按日期 + 行业 分组！
        grouped_ind = df.groupby(['trade_date', 'industry'])
        
        factor_cols =['pure_bp', 'pure_ep', 'sales_yoy', 'profit_yoy']
        
        for col in factor_cols:
            # 1. 行业内横截面排名 (0~1)，极其稳健的去极值方式
            rank_col = grouped_ind[col].rank(pct=True, ascending=True)
            
            # 2. 行业内 Z-Score 标准化
            mean = rank_col.groupby([df['trade_date'], df['industry']]).transform('mean')
            std = rank_col.groupby([df['trade_date'], df['industry']]).transform('std')
            
            df[col] = np.where(std > 1e-8, (rank_col - mean) / std, 0.0)
            
            del rank_col, mean, std
            gc.collect()
            
        # 丢弃行业标签，只输出纯净因子
        return df[['trade_date', 'ts_code'] + factor_cols]

    def run_pipeline(self, max_workers=14):
        stock_files =[f.replace('.parquet', '') for f in os.listdir(self.quotes_dir) if f.endswith('.parquet')]
        logger.info(f"🏭 Pure Fundamentals Factory starting. Target: {len(stock_files)} stocks.")
        
        # 审计修复 2：重启多线程并发引擎
        success_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures =[executor.submit(self._process_single_stock_fundamentals, code) for code in stock_files]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="[Map] Fundamentals"):
                if "Success" in future.result(): success_count += 1
                    
        if success_count == 0:
            logger.error("Pipeline Halted.")
            return
            
        logger.info("Force GC...")
        gc.collect() 

        # 审计修复 1：利用 PyArrow 避免 List Append OOM
        logger.info("[Reduce] Loading Panel Data into contiguous memory via PyArrow...")
        df_panel = pd.read_parquet(self.tmp_dir, engine='pyarrow')
        logger.info(f"Loaded Panel Data Shape: {df_panel.shape}")

        df_panel = self._vectorized_cross_sectional_processing(df_panel)
        
        save_path = os.path.join(self.output_dir, "pure_fundamentals.parquet")
        df_panel = df_panel.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
        df_panel.to_parquet(save_path, engine='pyarrow', index=False)
        
        logger.info("🧹 Cleaning up shards...")
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        
        logger.success(f"🏁 Industry-Neutral Pure Fundamentals Matrix built successfully!")
        logger.success(f"Saved to: {save_path}")

if __name__ == "__main__":
    logger.add("pure_fund_factory.log", rotation="50 MB", enqueue=True)
    factory = PureFundamentalsFactoryPro()
    factory.run_pipeline(max_workers=14)