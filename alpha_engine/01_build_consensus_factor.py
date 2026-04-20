# alpha_engine/01_build_consensus_factor.py

import os
import sys
import gc
import shutil
import numpy as np
import pandas as pd
from loguru import logger
import concurrent.futures
from tqdm import tqdm

try:
    import tushare as ts
    from dotenv import load_dotenv
except ImportError:
    logger.critical("Missing dependencies! pip install tushare python-dotenv")
    sys.exit(1)

load_dotenv()
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
if not TUSHARE_TOKEN:
    logger.critical("SECURITY HALT: TUSHARE_TOKEN missing!")
    sys.exit(1)

# 智能退避与频控 (分析师研报接口频控严格，阻尼调大)
def smart_safe_api_call(max_retries=5, base_delay=0.8):
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time, random
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    time.sleep(random.uniform(0.15, 0.3))
                    return result
                except Exception as e:
                    err_msg = str(e)
                    if any(kw in err_msg for kw in["频次", "限流", "网络保护"]):
                        time.sleep(base_delay * (2 ** attempt) + random.uniform(1.0, 2.0))
                        continue
                    if any(kw in err_msg for kw in["权限", "token"]):
                        raise e
                    time.sleep(base_delay * (1.5 ** attempt))
            raise ConnectionError(f"API Failed: {func.__name__}")
        return wrapper
    return decorator

class ConsensusFactorFactory:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_lake_dir = os.path.abspath(os.path.join(self.base_dir, "..", "data_lake"))
        self.quotes_dir = os.path.join(self.data_lake_dir, "daily_quotes")
        self.basics_path = os.path.join(self.data_lake_dir, "stock_basics.parquet")
        
        self.output_dir = os.path.join(self.base_dir, "factors")
        self.tmp_dir = os.path.join(self.output_dir, "tmp_consensus_shards")
        
        os.makedirs(self.output_dir, exist_ok=True)
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        ts.set_token(TUSHARE_TOKEN)
        self.pro = ts.pro_api()

    @smart_safe_api_call()
    def _fetch_reports(self, ts_code, start_date):
        # 拉取券商盈利预测研报
        return self.pro.report_rc(ts_code=ts_code, start_date=start_date)

    def _process_single_stock(self, ts_code: str) -> str:
        """Map阶段：将离散的研报事件，转化为连续的日频共识因子"""
        list_date = '20100101'  # 统一设定研报拉取起点为 2010年
        
        quotes_path = os.path.join(self.quotes_dir, f"{ts_code}.parquet")
        tmp_shard_path = os.path.join(self.tmp_dir, f"{ts_code}.parquet")
        
        if not os.path.exists(quotes_path): return "Missing Data"

        try:
            # 1. 极速读取日线
            df_q = pd.read_parquet(quotes_path, columns=['trade_date'])
            df_q['trade_date'] = pd.to_datetime(df_q['trade_date'], format='%Y%m%d')
            df_q = df_q.sort_values('trade_date').drop_duplicates('trade_date')

            # 2. 拉取研报数据 (时间回退保证上市以来的数据全覆盖)
            df_r = self._fetch_reports(ts_code, list_date)
            
            if df_r is None or df_r.empty:
                return "No Analyst Coverage" # 很多小盘股根本没有券商写研报

            df_r['report_date'] = pd.to_datetime(df_r['report_date'], format='%Y%m%d', errors='coerce')
            df_r = df_r.dropna(subset=['report_date']).sort_values('report_date')
            
            # 3. 评级量化映射 (NLP 词法情感降维)
            # 买入/强烈推荐 = 5, 增持/推荐 = 4, 中性 = 3, 减持/卖出 = 1
            def map_rating(rating):
                if not isinstance(rating, str): return 3.0
                if any(k in rating for k in ['买入', '强推', '强烈推荐']): return 5.0
                if any(k in rating for k in ['增持', '推荐', '审慎推荐']): return 4.0
                if any(k in rating for k in ['减持', '卖出', '回避']): return 1.0
                return 3.0 # 中性默认
                
            df_r['rating_score'] = df_r['op_rt'].apply(map_rating)

            # ==========================================
            # 核心算法：As-of 滚动统计研报共识 (The Analyst Consensus)
            # 研报是离散事件，我们需要将其转化为每天都能打分的时序序列
            # ==========================================
            df_alpha = pd.merge_asof(
                df_q, df_r[['report_date', 'rating_score', 'op_rt']], 
                left_on='trade_date', right_on='report_date', direction='backward'
            )
            
            # 过滤掉研报发布超过 90 天的噪音（超过3个月没研报说明机构不再关注了）
            df_alpha['days_since_report'] = (df_alpha['trade_date'] - df_alpha['report_date']).dt.days
            df_alpha = df_alpha[df_alpha['days_since_report'] <= 90].copy()
            
            # 衍生因子 1：分析师覆盖热度 (过去90天内研报热度，机构关注度)
            # 我们给热度一个简单的平滑，如果一直有新研报，热度就高
            df_alpha['attention_alpha'] = np.exp(-df_alpha['days_since_report'] / 30.0) # 30天半衰期衰减
            
            # 衍生因子 2：机构评级共识 (Rating Score)
            df_alpha['rating_alpha'] = df_alpha['rating_score']

            df_alpha['ts_code'] = ts_code
            df_alpha = df_alpha[['trade_date', 'ts_code', 'attention_alpha', 'rating_alpha']].dropna()
            
            if df_alpha.empty: return "Empty Valid Data"
            
            df_alpha.to_parquet(tmp_shard_path, engine='pyarrow', index=False)
            del df_q, df_r, df_alpha
            return "Success"
            
        except Exception as e:
            return f"Error: {str(e)}"

    def _vectorized_cross_sectional_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("🔗 Loading Industry Classifications for Neutralization...")
        df_basics = pd.read_parquet(self.basics_path)[['ts_code', 'industry']].fillna('Unknown')
        df = pd.merge(df, df_basics, on='ts_code', how='left')
        
        logger.info("⚡ Applying Industry-Neutral Vectorized Rank-Z...")
        grouped_ind = df.groupby(['trade_date', 'industry'])
        
        for col in['attention_alpha', 'rating_alpha']:
            # 行业内横向百分比排名
            rank_col = grouped_ind[col].rank(pct=True, ascending=True)
            
            # 行业内 Z-Score 标准化
            mean = rank_col.groupby([df['trade_date'], df['industry']]).transform('mean')
            std = rank_col.groupby([df['trade_date'], df['industry']]).transform('std')
            df[col] = np.where(std > 1e-8, (rank_col - mean) / std, 0.0)
            
            del rank_col, mean, std
            gc.collect()
            
        # ==========================================
        # 终极共识因子：60% 机构看好评级 + 40% 机构关注热度
        # ==========================================
        df['consensus_alpha'] = 0.6 * df['rating_alpha'] + 0.4 * df['attention_alpha']
        
        return df[['trade_date', 'ts_code', 'consensus_alpha', 'attention_alpha', 'rating_alpha']]

    def run_pipeline(self, max_workers=6):
        stock_files =[f.replace('.parquet', '') for f in os.listdir(self.quotes_dir) if f.endswith('.parquet')]
        logger.info(f"🏭 Analyst Consensus Factory starting. Target: {len(stock_files)} stocks.")
        
        # 研报接口频控极严，限制并发数防封禁
        success_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures =[executor.submit(self._process_single_stock, code) for code in stock_files]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="[Map] Analyst Reports"):
                if "Success" in future.result(): success_count += 1
                    
        if success_count == 0:
            logger.error("Pipeline Halted.")
            return
            
        gc.collect() 

        logger.info("[Reduce] Loading Panel Data into contiguous memory...")
        df_panel = pd.read_parquet(self.tmp_dir, engine='pyarrow')

        df_panel = self._vectorized_cross_sectional_processing(df_panel)
        
        save_path = os.path.join(self.output_dir, "consensus_alpha.parquet")
        df_panel = df_panel.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
        df_panel.to_parquet(save_path, engine='pyarrow', index=False)
        
        logger.info("🧹 Cleaning up shards...")
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        logger.success(f"🏁 Industry-Neutral Analyst Consensus Matrix built successfully!")

if __name__ == "__main__":
    logger.add("consensus_factory.log", rotation="50 MB", enqueue=True)
    ConsensusFactorFactory().run_pipeline(max_workers=6)