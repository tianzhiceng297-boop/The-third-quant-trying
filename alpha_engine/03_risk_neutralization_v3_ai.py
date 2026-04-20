# alpha_engine/03_risk_neutralization_v3_ai.py

import os
import sys
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
import concurrent.futures

class AINeutralizer:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_lake_dir = os.path.abspath(os.path.join(self.base_dir, "..", "data_lake"))
        
        # 输入：Layer 2.5 刚刚炼出的 AI 复合因子
        self.alpha_path = os.path.join(self.base_dir, "factors", "ai_composite_alpha.parquet")
        self.quotes_dir = os.path.join(self.data_lake_dir, "daily_quotes")
        self.basics_path = os.path.join(self.data_lake_dir, "stock_basics.parquet")
        self.names_path = os.path.join(self.data_lake_dir, "unified_name_history.parquet")
        
        # 输出：洗净铅华的纯粹 AI Alpha
        self.output_path = os.path.join(self.base_dir, "factors", "pure_ai_alpha.parquet")

    def _load_single_amount(self, file_path):
        try: return pd.read_parquet(file_path, columns=['trade_date', 'ts_code', 'amount'])
        except Exception: return pd.DataFrame()

    def build_risk_exposures(self) -> pd.DataFrame:
        logger.info("📡 [Layer 3] 1/4 Loading Daily Liquidity Proxies...")
        stock_files =[os.path.join(self.quotes_dir, f) for f in os.listdir(self.quotes_dir) if f.endswith('.parquet')]
        
        amt_pieces =[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=14) as executor:
            futures =[executor.submit(self._load_single_amount, f) for f in stock_files]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Loading Amts"):
                res = future.result()
                if not res.empty: amt_pieces.append(res)
                    
        df_amt = pd.concat(amt_pieces, ignore_index=True)
        df_amt['trade_date'] = pd.to_datetime(df_amt['trade_date'])
        
        df_amt = df_amt.sort_values(['ts_code', 'trade_date'])
        df_amt['amt_ma20'] = df_amt.groupby('ts_code')['amount'].transform(lambda x: x.rolling(20).mean())
        df_amt['ln_size'] = np.log(df_amt['amt_ma20'] + 1.0)
        return df_amt.dropna(subset=['ln_size'])[['trade_date', 'ts_code', 'ln_size']]

    def execute_neutralization(self):
        if not os.path.exists(self.alpha_path):
            logger.critical("AI Alpha not found! Run Layer 2.5 LightGBM Combiner first.")
            sys.exit(1)

        logger.info("📥 [Layer 3] 2/4 Merging AI Alpha, Industry, and PIT Names...")
        df_alpha = pd.read_parquet(self.alpha_path)
        df_basics = pd.read_parquet(self.basics_path)[['ts_code', 'industry', 'name']].fillna('Unknown')
        df_basics.rename(columns={'name': 'static_name'}, inplace=True)
        
        df_names = pd.read_parquet(self.names_path)
        df_names['start_date'] = pd.to_datetime(df_names['start_date'])
        df_names = df_names.dropna(subset=['start_date']).sort_values('start_date')

        df_risk = self.build_risk_exposures()
        df_master = pd.merge(df_alpha, df_risk, on=['trade_date', 'ts_code'], how='inner')
        df_master = pd.merge(df_master, df_basics, on='ts_code', how='left')

        logger.info("🛡️ [Layer 3] 3/4 Applying Universe Filtration (Killing STs & Micro-Caps)...")
        df_master = df_master.sort_values('trade_date')
        df_master = pd.merge_asof(
            df_master, df_names[['start_date', 'ts_code', 'name']], 
            left_on='trade_date', right_on='start_date', by='ts_code', direction='backward'
        )
        
        df_master['name'] = df_master['name'].fillna(df_master['static_name'])
        mask_st = df_master['name'].str.contains('ST|PT|退', na=False)
        df_master = df_master[~mask_st].copy()

        # 动态剔除后 20% 流动性枯竭的股票
        thresholds = df_master.groupby('trade_date')['ln_size'].transform(lambda x: x.quantile(0.20))
        df_master = df_master[df_master['ln_size'] > thresholds].copy().reset_index(drop=True)

        logger.info("📐 [Layer 3] 4/4 Executing NumPy SVD Orthogonalization on AI Alpha...")
        pure_alphas = np.zeros(len(df_master), dtype=np.float64)
        
        for trade_date, group in tqdm(df_master.groupby('trade_date'), desc="SVD Neutralization"):
            if len(group) < 30: continue
                
            Y = group['ai_composite_alpha'].values
            industries = group['industry'].values
            unique_inds = np.unique(industries)
            if len(unique_inds) <= 1: continue
            
            ind_dummies =[np.where(industries == ind, 1.0, 0.0) for ind in unique_inds[1:]]
            X_cols =[np.ones(len(Y)), group['ln_size'].values] + ind_dummies
            X = np.column_stack(X_cols)
            
            # 极速矩阵伪逆求解
            beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            resid = Y - X.dot(beta)
            pure_alphas[group.index] = resid
            
        df_master['pure_ai_alpha'] = pure_alphas
        
        # 每日截面 Z-Score 标准化
        mean = df_master.groupby('trade_date')['pure_ai_alpha'].transform('mean')
        std = df_master.groupby('trade_date')['pure_ai_alpha'].transform('std')
        df_master['pure_ai_alpha'] = np.where(std > 1e-8, (df_master['pure_ai_alpha'] - mean) / std, 0.0)

        df_final = df_master[['trade_date', 'ts_code', 'pure_ai_alpha']].dropna()
        df_final.to_parquet(self.output_path, engine='pyarrow', index=False)
        
        logger.success("="*60)
        logger.success("✅ Layer 3: AI Factor Risk-Neutralization Complete!")
        logger.success(f"💾 Pure AI Alpha saved to: {self.output_path}")
        logger.success("="*60)

if __name__ == "__main__":
    logger.add("layer3_neutralizer.log", rotation="50 MB", enqueue=True)
    AINeutralizer().execute_neutralization()