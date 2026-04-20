# env_check.py
import sys
import platform
import importlib

def check_environment():
    print("="*50)
    print(" A-Share Quantamental Framework - Environment Probe ")
    print("="*50)
    
    # 1. 检查底层系统与 Python 环境
    print("\n[1. System & Python Info]")
    print(f"OS Platform    : {platform.system()} {platform.release()}")
    print(f"Python Version : {sys.version.split()[0]}")
    print(f"Architecture   : {platform.machine()}")

    # 2. 检查核心依赖包 (列式存储、数据库、日志、进度条)
    print("\n[2. Core Dependencies Check]")
    core_packages =['pandas', 'pyarrow', 'fastparquet', 'duckdb', 'loguru', 'tqdm', 'tushare']
    
    missing_pkgs =[]
    for pkg in core_packages:
        try:
            lib = importlib.import_module(pkg)
            version = getattr(lib, '__version__', 'unknown')
            print(f"  [√] {pkg:<12} : Installed (v{version})")
        except ImportError:
            print(f"  [X] {pkg:<12} : NOT INSTALLED!")
            missing_pkgs.append(pkg)

    # 3. 检查 Tushare API 连通性与权限
    print("\n[3. Tushare API & Network Check]")
    try:
        import tushare as ts
        # 【重要】请在这里填入你的 Tushare Token
        TOKEN = '6ddbbb8ecd27d50217381a8cb4715a48198f9faf6fd998b9237fb864' 
        
        if TOKEN == '6ddbbb8ecd27d50217381a8cb4715a48198f9faf6fd998b9237fb864':
            print("  [!] Please replace the placeholder with your real Tushare Token in the script.")
        else:
            ts.set_token(TOKEN)
            pro = ts.pro_api()
            
            # 尝试拉取上交所 2024 年前 5 个交易日的日历，测试网络与积分权限
            df_cal = pro.trade_cal(exchange='SSE', start_date='20240101', end_date='20240110')
            
            if not df_cal.empty:
                print(f"  [√] Tushare Connected! Successfully fetched {len(df_cal)} rows of calendar data.")
                print(f"      Sample data:\n{df_cal.head(2).to_string()}")
            else:
                print("  [X] Tushare Connected, but returned empty data. Check permissions.")
                
    except Exception as e:
        print(f"  [X] Tushare connection failed. Error: {str(e)}")

    print("\n" + "="*50)
    if missing_pkgs:
        print(f"WARNING: Please pip/conda install the missing packages: {', '.join(missing_pkgs)}")
    else:
        print("ALL SYSTEMS GO! The environment is ready for Layer 1 engineering.")
    print("="*50)

if __name__ == "__main__":
    check_environment()