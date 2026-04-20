# 01_build_universe_v5_final.py

import sys
try:
    import os
    import time
    from datetime import datetime, timedelta
    import tushare as ts
    import pandas as pd
    from loguru import logger
    from dotenv import load_dotenv
    from filelock import FileLock, Timeout
except ImportError as e:
    print(f"CRITICAL BOOT ERROR: Missing core dependency. Details: {e}")
    sys.exit(1)

load_dotenv()
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
if not TUSHARE_TOKEN or TUSHARE_TOKEN.strip() == "" or TUSHARE_TOKEN == "YOUR_TUSHARE_TOKEN_HERE":
    logger.critical("SECURITY HALT: TUSHARE_TOKEN is missing or invalid in environment/.env file!")
    sys.exit(1)

# ==========================================
# 工业级智能退避装饰器
# ==========================================
def smart_safe_api_call(max_retries=3, base_delay=1.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    time.sleep(base_delay) 
                    return result
                except Exception as e:
                    err_msg = str(e)
                    if any(kw in err_msg for kw in ["权限", "未授权", "token", "Token"]):
                        logger.critical(f"FATAL Auth Error in {func.__name__}: {err_msg}")
                        raise e
                    if isinstance(e, (TypeError, ValueError, KeyError, AttributeError)):
                        logger.critical(f"FATAL Code Error in {func.__name__}: {err_msg}")
                        raise e
                    
                    sleep_time = base_delay * (2 ** attempt)
                    logger.warning(f"Network Warning: {func.__name__} | Retrying in {sleep_time}s ({attempt+1}/{max_retries})")
                    time.sleep(sleep_time)
            logger.error(f"FATAL: {func.__name__} exhausted retries.")
            raise ConnectionError(f"API exhausted retries for {func.__name__}")
        return wrapper
    return decorator


class UltimateUniverseBuilder:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # 魔法武器：获取当前运行脚本的绝对路径的所在目录
            # 因为你把脚本放到了 /quant_project/data_lake/ 下，所以 script_dir 就是 data_lake 目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 我们把数据存放在与脚本同级的目录下（或者你也可以往上一级指定）
            self.data_dir = script_dir
        else:
            self.data_dir = data_dir
            
        # 优化点 3：独立的锁目录，规避跨平台路径问题
        self.lock_dir = os.path.join(self.data_dir, ".locks")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.lock_dir, exist_ok=True)
        
        ts.set_token(TUSHARE_TOKEN)
        self.pro = ts.pro_api()
        
        self.start_date = '20100101'
        self.end_date = str(datetime.now().year + 1) + '1231'

    def _get_lock_path(self, filename: str) -> str:
        """生成安全的锁文件路径"""
        return os.path.join(self.lock_dir, f"{os.path.basename(filename)}.lock")

    def _atomic_save(self, df: pd.DataFrame, final_path: str):
        tmp_path = final_path + ".tmp"
        df.to_parquet(tmp_path, engine='pyarrow', index=False)
        os.replace(tmp_path, final_path)

    @smart_safe_api_call(max_retries=3, base_delay=0.6)
    def _fetch_calendar(self, exchange, start_date, end_date):
        return self.pro.trade_cal(exchange=exchange, start_date=start_date, end_date=end_date)

    @smart_safe_api_call(max_retries=3, base_delay=0.6)
    def _fetch_stock_basic(self, list_status):
        fields = 'ts_code,symbol,name,area,industry,market,list_status,list_date,delist_date'
        return self.pro.stock_basic(exchange='', list_status=list_status, fields=fields)
        
    @smart_safe_api_call(max_retries=3, base_delay=1.0)
    def _fetch_namechange(self):
        return self.pro.namechange(exchange='')

    def build_trade_calendar(self):
        save_path = os.path.join(self.data_dir, "trade_calendar.parquet")
        
        try:
            with FileLock(self._get_lock_path(save_path), timeout=60):
                fetch_start = self.start_date
                df_exist = pd.DataFrame()

                if os.path.exists(save_path):
                    try:
                        df_exist = pd.read_parquet(save_path)
                        last_date = df_exist['cal_date'].max()
                        if (datetime.now() - last_date).days > 90:
                            fetch_start = self.start_date
                        elif last_date.date() > datetime.now().date() + timedelta(days=30):
                            logger.info("Trade calendar is up-to-date. Skipping.")
                            return
                        else:
                            fetch_start = (last_date - timedelta(days=7)).strftime('%Y%m%d')
                    except Exception:
                        fetch_start = self.start_date

                df_sse = self._fetch_calendar('SSE', fetch_start, self.end_date)
                df_szse = self._fetch_calendar('SZSE', fetch_start, self.end_date)
                
                df_sse_open = df_sse[df_sse['is_open'] == 1][['cal_date']]
                df_szse_open = df_szse[df_szse['is_open'] == 1][['cal_date']]
                
                df_delta = pd.merge(df_sse_open, df_szse_open, on='cal_date', how='inner')
                df_delta['cal_date'] = pd.to_datetime(df_delta['cal_date'])
                
                if df_delta.empty:
                    return

                if not df_exist.empty and fetch_start != self.start_date:
                    df_final = pd.concat([df_exist, df_delta]).drop_duplicates('cal_date').sort_values('cal_date')
                else:
                    df_final = df_delta.sort_values('cal_date')
                    
                df_final.reset_index(drop=True, inplace=True)
                self._atomic_save(df_final, save_path)
                logger.success(f"Trade calendar synced. Total days: {len(df_final)}")
                
        except Timeout:
            logger.error("Lock acquisition failed for trade calendar.")

    def build_unified_stock_universe(self):
        save_basics_path = os.path.join(self.data_dir, "stock_basics.parquet")
        save_names_path = os.path.join(self.data_dir, "unified_name_history.parquet")
        
        try:
            with FileLock(self._get_lock_path(save_basics_path), timeout=120):
                if os.path.exists(save_basics_path) and os.path.exists(save_names_path):
                    if datetime.fromtimestamp(os.path.getmtime(save_basics_path)).date() == datetime.now().date():
                        logger.info("Universe already updated today. Skipping.")
                        return

                logger.info("Fetching stock basics & syncing history...")
                
                df_L = self._fetch_stock_basic('L')
                df_D = self._fetch_stock_basic('D')
                df_P = self._fetch_stock_basic('P')
                
                df_basics = pd.concat([df_L, df_D, df_P], ignore_index=True)
                df_basics['list_date'] = pd.to_datetime(df_basics['list_date'], errors='coerce')
                df_basics['delist_date'] = pd.to_datetime(df_basics['delist_date'], errors='coerce')
                self._atomic_save(df_basics, save_basics_path)
                
                # ==========================================
                # 优化点 1：边界风险应对，强制 Schema 约束
                # ==========================================
                expected_cols =['ts_code', 'name', 'start_date', 'end_date', 'ann_date', 'change_reason']
                df_namechange = self._fetch_namechange()
                
                if df_namechange is None or df_namechange.empty:
                    df_namechange = pd.DataFrame(columns=expected_cols)

                df_namechange['start_date'] = pd.to_datetime(df_namechange['start_date'], errors='coerce')
                df_namechange['end_date'] = pd.to_datetime(df_namechange['end_date'], errors='coerce')
                df_namechange['ann_date'] = pd.to_datetime(df_namechange['ann_date'], errors='coerce')
                
                changed_ts_codes = set(df_namechange['ts_code'].dropna().unique())
                never_changed = df_basics[~df_basics['ts_code'].isin(changed_ts_codes)].copy()
                
                if not never_changed.empty:
                    # ==========================================
                    # 优化点 2：退市股票生命周期终点自洽修正
                    # ==========================================
                    virtual_history = pd.DataFrame({
                        'ts_code': never_changed['ts_code'],
                        'name': never_changed['name'],
                        'start_date': never_changed['list_date'],
                        # 巧妙应用 fillna：若退市则取退市日，若未退市则取 2099 年
                        'end_date': never_changed['delist_date'].fillna(pd.to_datetime('2099-12-31')),
                        'ann_date': pd.NaT,
                        'change_reason': 'Never Changed'
                    })
                    df_unified_names = pd.concat([df_namechange, virtual_history], ignore_index=True)
                else:
                    df_unified_names = df_namechange.copy()
                
                df_unified_names = df_unified_names.sort_values(by=['ts_code', 'start_date']).reset_index(drop=True)
                self._atomic_save(df_unified_names, save_names_path)
                
                logger.success(f"Unified Universe synced! Basics: {len(df_basics)}, Name History: {len(df_unified_names)}")

        except Timeout:
            logger.error("Lock acquisition failed for universe.")

if __name__ == "__main__":
    builder = UltimateUniverseBuilder()
    builder.build_trade_calendar()
    builder.build_unified_stock_universe()