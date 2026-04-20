# 02_download_daily_quotes_v8_pinnacle.py

import os
import sys
import time
import glob
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

# ==========================================
# 环境与安全校验
# ==========================================
load_dotenv()
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
if not TUSHARE_TOKEN or TUSHARE_TOKEN.strip() == "":
    logger.critical("SECURITY HALT: TUSHARE_TOKEN is missing!")
    sys.exit(1)

# ==========================================
# 工业级智能退避与频控装饰器
# ==========================================
def smart_safe_api_call(max_retries=5, base_delay=0.5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    time.sleep(0.15) 
                    return result
                except Exception as e:
                    err_msg = str(e)
                    if any(kw in err_msg for kw in["抱歉", "频次", "限流", "触发网络保护"]):
                        sleep_time = base_delay * (2 ** attempt) + 1.0
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

class DailyQuotesEngine:
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.dirname(os.path.abspath(__file__))
        self.quotes_dir = os.path.join(self.data_dir, "daily_quotes")
        self.locks_dir = os.path.join(self.data_dir, ".locks", "quotes")
        
        os.makedirs(self.quotes_dir, exist_ok=True)
        os.makedirs(self.locks_dir, exist_ok=True)
        
        self._cleanup_orphaned_locks()
        
        ts.set_token(TUSHARE_TOKEN)
        self.pro = ts.pro_api()

    def _cleanup_orphaned_locks(self):
        lock_files = glob.glob(os.path.join(self.locks_dir, "*.lock"))
        count = 0
        for f in lock_files:
            try:
                os.remove(f)
                count += 1
            except OSError:
                pass
        if count > 0:
            logger.info(f"🧹 Cleared {count} orphaned lock files.")

    def _get_lock_path(self, ts_code: str) -> str:
        return os.path.join(self.locks_dir, f"{ts_code}.lock")

    @smart_safe_api_call()
    def _fetch_daily(self, ts_code, start_date, end_date):
        return self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

    @smart_safe_api_call()
    def _fetch_adj_factor(self, ts_code, start_date, end_date):
        return self.pro.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)

    def _atomic_save(self, df: pd.DataFrame, final_path: str):
        tmp_path = final_path + ".tmp"
        df.to_parquet(tmp_path, engine='pyarrow', index=False)
        os.replace(tmp_path, final_path)

    def process_single_stock(self, stock_info: dict) -> dict:
        ts_code = stock_info['ts_code']
        list_date = str(stock_info['list_date'])
        delist_date = stock_info['delist_date']
        
        # 统一格式化限制日期
        end_date_limit = str(delist_date) if pd.notna(delist_date) else datetime.now().strftime('%Y%m%d')
        save_path = os.path.join(self.quotes_dir, f"{ts_code}.parquet")
        
        try:
            with FileLock(self._get_lock_path(ts_code), timeout=10):
                fetch_start = list_date
                df_exist = pd.DataFrame()
                
                if os.path.exists(save_path):
                    try:
                        df_dates = pd.read_parquet(save_path, columns=['trade_date'])
                        last_date_val = df_dates['trade_date'].max()
                        
                        # ==========================================
                        # 审计修复 2：根除 pd.to_datetime(int) 的纳秒陷阱
                        # 强制转换为 str，并指定 format='%Y%m%d'
                        # ==========================================
                        last_date_dt = pd.to_datetime(str(last_date_val), format='%Y%m%d')
                        limit_dt = pd.to_datetime(end_date_limit, format='%Y%m%d')
                        yesterday_dt = datetime.now() - timedelta(days=1)
                        
                        if last_date_dt >= limit_dt or last_date_dt.date() >= yesterday_dt.date():
                            return {'ts_code': ts_code, 'status': 'skip', 'msg': 'Up-to-date'}
                        
                        # ==========================================
                        # 审计修复 1：边界防穿透（回退不得早于上市日）
                        # ==========================================
                        lookback_dt = last_date_dt - timedelta(days=5)
                        list_dt = pd.to_datetime(list_date, format='%Y%m%d')
                        actual_start_dt = max(lookback_dt, list_dt)
                        fetch_start = actual_start_dt.strftime('%Y%m%d')
                        
                        # ==========================================
                        # 审计修复 3：内存碎片与性能极客优化
                        # 读取全量后，立刻截断 fetch_start 及之后的老数据，化去重为纯净追加
                        # ==========================================
                        df_exist_full = pd.read_parquet(save_path)
                        # 确保 df_exist 中的 trade_date 也是统一的字符串用于安全对比
                        df_exist_full['trade_date'] = df_exist_full['trade_date'].astype(str)
                        df_exist = df_exist_full[df_exist_full['trade_date'] < fetch_start].copy()
                        del df_exist_full # 主动释放全量内存
                        
                    except Exception:
                        fetch_start = list_date 
                        df_exist = pd.DataFrame()

                # API 拉取
                df_daily = self._fetch_daily(ts_code, fetch_start, end_date_limit)
                df_adj = self._fetch_adj_factor(ts_code, fetch_start, end_date_limit)
                
                if df_daily is None or df_daily.empty:
                    return {'ts_code': ts_code, 'status': 'skip', 'msg': 'No new data from API'}
                if df_adj is None or df_adj.empty:
                    # 如果日线有数据但复权因子没出，为了数据一致性，直接跳过等明天修复
                    return {'ts_code': ts_code, 'status': 'error', 'msg': 'API returned quotes but missing adj_factor'}

                df_merged = pd.merge(df_daily, df_adj, on=['ts_code', 'trade_date'], how='left')
                
                # 强类型与安全约束
                df_merged['trade_date'] = df_merged['trade_date'].astype(str)
                df_merged = df_merged.sort_values('trade_date')
                df_merged['adj_factor'] = df_merged['adj_factor'].ffill().fillna(1.0).astype(float)
                
                # 由于我们已经截断了 df_exist 中重叠的部分，这里直接纯净追加 (Append)
                if not df_exist.empty:
                    df_final = pd.concat([df_exist, df_merged], ignore_index=True)
                else:
                    df_final = df_merged

                df_final = df_final.sort_values('trade_date').reset_index(drop=True)
                
                # 终极保险：物理文件原子写入
                self._atomic_save(df_final, save_path)
                
                return {'ts_code': ts_code, 'status': 'success', 'msg': f'Appended {len(df_merged)} rows'}

        except Timeout:
            return {'ts_code': ts_code, 'status': 'error', 'msg': 'FileLock timeout'}
        except Exception as e:
            return {'ts_code': ts_code, 'status': 'error', 'msg': str(e)}

    def run_concurrent_pipeline(self, max_workers=8):
        basics_path = os.path.join(self.data_dir, "stock_basics.parquet")
        if not os.path.exists(basics_path):
            logger.error("stock_basics.parquet not found! Run Layer 1 first.")
            return

        df_basics = pd.read_parquet(basics_path)
        stocks_to_process =[]
        for _, row in df_basics.iterrows():
            stocks_to_process.append({
                'ts_code': row['ts_code'],
                'list_date': row['list_date'].strftime('%Y%m%d') if pd.notna(row['list_date']) else '20100101',
                'delist_date': row['delist_date'].strftime('%Y%m%d') if pd.notna(row['delist_date']) else None
            })

        logger.info(f"🚀 Igniting V8 Pinnacle Engine. Target: {len(stocks_to_process)} stocks.")
        
        success_count, skip_count = 0, 0
        failed_stocks =[]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_single_stock, stock): stock['ts_code'] for stock in stocks_to_process}
            
            with tqdm(total=len(stocks_to_process), desc="Downloading Quotes", unit="stock") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    ts_code = futures[future]
                    try:
                        result = future.result()
                        if result['status'] == 'success':
                            success_count += 1
                        elif result['status'] == 'skip':
                            skip_count += 1
                        else:
                            failed_stocks.append(f"{ts_code} | {result['msg']}")
                    except Exception as e:
                        failed_stocks.append(f"{ts_code} | Unhandled Exception: {str(e)}")
                    finally:
                        pbar.update(1)

        # 审计报告
        logger.info("="*50)
        logger.info("📊 V8 PIPELINE AUDIT REPORT")
        logger.info(f"Success/Updated : {success_count}")
        logger.info(f"Skipped(Fast-IO) : {skip_count}")
        logger.info(f"Failed/Errors    : {len(failed_stocks)}")
        logger.info("="*50)

        if failed_stocks:
            fail_rate = len(failed_stocks) / len(stocks_to_process)
            for fail_msg in failed_stocks[:20]: 
                logger.warning(f"  - {fail_msg}")
            if len(failed_stocks) > 20:
                logger.warning(f"  ... and {len(failed_stocks) - 20} more.")
            
            if fail_rate > 0.05:
                logger.critical(f"🛑 CRITICAL: Failure rate {fail_rate:.1%} exceeds 5% threshold!")
                sys.exit(1)
        else:
            logger.success("🏁 V8 Pipeline Finished flawlessly! 100% Data Integrity.")

if __name__ == "__main__":
    logger.add("download_pipeline.log", rotation="50 MB", enqueue=True)
    engine = DailyQuotesEngine()
    engine.run_concurrent_pipeline(max_workers=8)