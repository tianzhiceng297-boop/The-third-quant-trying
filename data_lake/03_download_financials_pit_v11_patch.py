# 03_download_financials_pit_v11_patch.py

import os
import sys
import time
import glob
import random
from datetime import datetime, timedelta
import concurrent.futures

try:
    import tushare as ts
    import pandas as pd
    from loguru import logger
    from dotenv import load_dotenv
    from tqdm import tqdm
    from filelock import FileLock, Timeout
except ImportError as e:
    print(f"CRITICAL BOOT ERROR: Missing dependencies. {e}")
    sys.exit(1)

load_dotenv()
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
if not TUSHARE_TOKEN or TUSHARE_TOKEN.strip() == "":
    logger.critical("SECURITY HALT: TUSHARE_TOKEN is missing!")
    sys.exit(1)

def smart_safe_api_call(max_retries=5, base_delay=0.8):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    time.sleep(random.uniform(0.1, 0.25))
                    return result
                except Exception as e:
                    err_msg = str(e)
                    if any(kw in err_msg for kw in["抱歉", "频次", "限流", "触发网络保护"]):
                        sleep_time = base_delay * (2 ** attempt) + random.uniform(1.0, 3.0)
                        time.sleep(sleep_time)
                        continue
                    if any(kw in err_msg for kw in["权限", "未授权", "token"]):
                        logger.critical(f"FATAL Auth Error: {err_msg}")
                        raise e
                    sleep_time = base_delay * (1.5 ** attempt)
                    time.sleep(sleep_time)
            raise ConnectionError(f"Exhausted {max_retries} retries for {func.__name__}")
        return wrapper
    return decorator

class FinancialPITEngine:
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.dirname(os.path.abspath(__file__))
        self.fin_dir = os.path.join(self.data_dir, "financials")
        self.locks_dir = os.path.join(self.data_dir, ".locks", "financials")
        
        os.makedirs(self.fin_dir, exist_ok=True)
        os.makedirs(self.locks_dir, exist_ok=True)
        self._cleanup_orphaned_locks()
        
        ts.set_token(TUSHARE_TOKEN)
        self.pro = ts.pro_api()
        
        # 潜在的所有 Join Keys
        self.base_keys =['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'update_flag']

    def _cleanup_orphaned_locks(self):
        lock_files = glob.glob(os.path.join(self.locks_dir, "*.lock"))
        for f in lock_files:
            try: os.remove(f)
            except OSError: pass

    def _get_lock_path(self, ts_code: str) -> str:
        return os.path.join(self.locks_dir, f"{ts_code}.lock")

    def _atomic_save(self, df: pd.DataFrame, final_path: str):
        tmp_path = final_path + ".tmp"
        df.to_parquet(tmp_path, engine='pyarrow', index=False)
        os.replace(tmp_path, final_path)

    @smart_safe_api_call()
    def _fetch_income(self, ts_code, start_date): return self.pro.income(ts_code=ts_code, start_date=start_date)
    @smart_safe_api_call()
    def _fetch_balancesheet(self, ts_code, start_date): return self.pro.balancesheet(ts_code=ts_code, start_date=start_date)
    @smart_safe_api_call()
    def _fetch_cashflow(self, ts_code, start_date): return self.pro.cashflow(ts_code=ts_code, start_date=start_date)
    @smart_safe_api_call()
    def _fetch_fina_indicator(self, ts_code, start_date): return self.pro.fina_indicator(ts_code=ts_code, start_date=start_date)

    def _clean_and_suffix(self, df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        """安全返回空表或带后缀的清洗表"""
        if df is None or df.empty:
            return pd.DataFrame()
            
        cols_to_drop = [c for c in['comp_type', 'report_type', 'basic_eps', 'diluted_eps'] if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        
        rename_dict = {col: f"{col}_{suffix}" for col in df.columns if col not in self.base_keys}
        return df.rename(columns=rename_dict)

    def process_single_stock(self, stock_info: dict) -> dict:
        ts_code = stock_info['ts_code']
        list_date = str(stock_info['list_date'])
        save_path = os.path.join(self.fin_dir, f"{ts_code}.parquet")
        
        try:
            with FileLock(self._get_lock_path(ts_code), timeout=15):
                fetch_start = list_date
                df_exist = pd.DataFrame()
                
                if os.path.exists(save_path):
                    try:
                        df_dates = pd.read_parquet(save_path, columns=['end_date'])
                        last_end_val = df_dates['end_date'].max()
                        last_end_dt = pd.to_datetime(str(last_end_val), format='%Y%m%d')
                        
                        lookback_dt = last_end_dt - timedelta(days=365)
                        list_dt = pd.to_datetime(list_date, format='%Y%m%d')
                        fetch_start = max(lookback_dt, list_dt).strftime('%Y%m%d')
                        
                        df_exist_full = pd.read_parquet(save_path)
                        df_exist_full['end_date'] = df_exist_full['end_date'].astype(str)
                        df_exist = df_exist_full[df_exist_full['end_date'] < fetch_start].copy()
                        del df_exist_full
                    except Exception:
                        fetch_start = list_date
                        df_exist = pd.DataFrame()

                # 拉取数据
                df_inc = self._fetch_income(ts_code, fetch_start)
                df_bs  = self._fetch_balancesheet(ts_code, fetch_start)
                df_cf  = self._fetch_cashflow(ts_code, fetch_start)
                df_ind = self._fetch_fina_indicator(ts_code, fetch_start)

                # 后缀处理与空表过滤
                dfs =[
                    self._clean_and_suffix(df_inc, 'inc'),
                    self._clean_and_suffix(df_bs,  'bs'),
                    self._clean_and_suffix(df_cf,  'cf'),
                    self._clean_and_suffix(df_ind, 'ind')
                ]
                valid_dfs = [df for df in dfs if not df.empty]
                
                if not valid_dfs:
                    return {'ts_code': ts_code, 'status': 'skip', 'msg': 'No new financial data'}

                # ==========================================
                # V11 致命修复：动态寻找公共主键 (Dynamic Outer Join)
                # ==========================================
                df_merged = valid_dfs[0]
                for df_part in valid_dfs[1:]:
                    # 只选取两个表【都存在】的 base_keys 进行 join
                    join_keys =[k for k in self.base_keys if k in df_merged.columns and k in df_part.columns]
                    df_merged = pd.merge(df_merged, df_part, on=join_keys, how='outer')

                # 双轨日期兜底
                if 'f_ann_date' in df_merged.columns and 'ann_date' in df_merged.columns:
                    df_merged['ann_date'] = df_merged['ann_date'].fillna(df_merged['f_ann_date'])
                
                if 'ann_date' in df_merged.columns:
                    df_merged = df_merged.dropna(subset=['ann_date'])
                if df_merged.empty:
                    return {'ts_code': ts_code, 'status': 'skip', 'msg': 'No valid ann_date'}

                # 强类型与缺失值保护
                df_merged['ann_date'] = df_merged['ann_date'].astype(str)
                df_merged['end_date'] = df_merged['end_date'].astype(str)
                if 'update_flag' in df_merged.columns:
                    df_merged['update_flag'] = df_merged['update_flag'].fillna(0).astype(int)
                else:
                    df_merged['update_flag'] = 0

                # 拼接与排序覆盖
                if not df_exist.empty:
                    df_final = pd.concat([df_exist, df_merged], ignore_index=True)
                else:
                    df_final = df_merged

                df_final = df_final.sort_values(['end_date', 'ann_date', 'update_flag'])
                df_final = df_final.drop_duplicates(subset=['end_date', 'ann_date'], keep='last')
                df_final = df_final.reset_index(drop=True)
                
                self._atomic_save(df_final, save_path)
                
                return {'ts_code': ts_code, 'status': 'success', 'msg': f'Synced {len(df_merged)} financial snapshots'}

        except Timeout:
            return {'ts_code': ts_code, 'status': 'error', 'msg': 'FileLock timeout'}
        except Exception as e:
            return {'ts_code': ts_code, 'status': 'error', 'msg': str(e)}

    def run_concurrent_pipeline(self, max_workers=5):
        basics_path = os.path.join(self.data_dir, "stock_basics.parquet")
        if not os.path.exists(basics_path):
            logger.error("stock_basics.parquet not found!")
            return

        df_basics = pd.read_parquet(basics_path)
        stocks_to_process = [{'ts_code': r['ts_code'], 'list_date': r['list_date'].strftime('%Y%m%d') if pd.notna(r['list_date']) else '20100101'} for _, r in df_basics.iterrows()]

        logger.info(f"🚀 Igniting V11 Patch Engine. Target: {len(stocks_to_process)} stocks.")
        
        success_count, skip_count = 0, 0
        failed_stocks =[]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_single_stock, stock): stock['ts_code'] for stock in stocks_to_process}
            
            with tqdm(total=len(stocks_to_process), desc="Downloading Financials", unit="stock") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    ts_code = futures[future]
                    try:
                        result = future.result()
                        if result['status'] == 'success': success_count += 1
                        elif result['status'] == 'skip': skip_count += 1
                        else: failed_stocks.append(f"{ts_code} | {result['msg']}")
                    except Exception as e:
                        failed_stocks.append(f"{ts_code} | Exception: {str(e)}")
                    finally:
                        pbar.update(1)

        logger.info("="*50)
        logger.info("📊 V11 FINANCIAL PIPELINE AUDIT REPORT")
        logger.info(f"Success/Updated : {success_count}")
        logger.info(f"Skipped(No New) : {skip_count}")
        logger.info(f"Failed/Errors   : {len(failed_stocks)}")
        logger.info("="*50)

        if failed_stocks:
            fail_rate = len(failed_stocks) / len(stocks_to_process)
            for fail_msg in failed_stocks[:20]: logger.warning(f"  - {fail_msg}")
            if fail_rate > 0.05:
                logger.critical(f"🛑 CRITICAL: Failure rate {fail_rate:.1%} exceeds 5% threshold!")
                sys.exit(1)
        else:
            logger.success("🏁 V11 Pipeline Finished flawlessly! PIT data sealed.")

if __name__ == "__main__":
    logger.add("download_financials.log", rotation="50 MB", enqueue=True)
    engine = FinancialPITEngine()
    engine.run_concurrent_pipeline(max_workers=5)