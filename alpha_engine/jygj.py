import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.api import OLS
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class CointegrationAnalyzer:
    """
    协整关系分析器：找出与目标股票在特定时间段内有协整关系的股票
    """
    
    def __init__(self, target_ticker='603871.SS', start_date='2026-03-01', end_date='2026-04-07'):
        self.target_ticker = target_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.target_data = None
        self.candidates = {
            # 景顺长城系重仓股
            '000630.SZ': '铜陵有色',  # 景盛双息重仓
            '603281.SS': '江瀚新材',  # 能源基建混合
            '601899.SS': '紫金矿业',  # 景顺系第一大重仓
            '600362.SS': '江西铜业',  # 同属铜板块
            
            # 中证1000成分股+周期股（3月13日同步暴跌）
            '601600.SS': '中国铝业',
            '002155.SZ': '湖南黄金',
            '600489.SS': '中金黄金',
            '000878.SZ': '云南铜业',
            
            # 跨境物流/贸易
            '603162.SS': '海通发展',
            
            # 指数基准
            '000852.SS': '中证1000指数',
            '000300.SS': '沪深300指数',
        }
        
    def fetch_data(self, ticker):
        """获取股票数据"""
        try:
            data = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
            if len(data) > 0:
                return data['Close']
            else:
                return None
        except:
            return None
    
    def load_all_data(self):
        """加载所有候选股票数据"""
        print(f"正在获取数据：{self.start_date} 至 {self.end_date}")
        
        # 获取目标股票数据
        self.target_data = self.fetch_data(self.target_ticker)
        if self.target_data is None:
            print(f"错误：无法获取目标股票 {self.target_ticker}")
            return False
            
        print(f"✓ 目标股票 {self.target_ticker} 获取成功，共 {len(self.target_data)} 个交易日")
        
        # 获取候选股票数据
        self.candidate_data = {}
        for ticker, name in self.candidates.items():
            data = self.fetch_data(ticker)
            if data is not None and len(data) > 10:  # 至少10个交易日数据
                self.candidate_data[ticker] = {
                    'name': name,
                    'data': data,
                    'returns': np.log(data / data.shift(1)).dropna()
                }
                print(f"✓ {name} ({ticker}) 获取成功")
            else:
                print(f"✗ {name} ({ticker}) 数据不足或获取失败")
                
        return True
    
    def check_stationarity(self, series, name="Series"):
        """ADF平稳性检验"""
        result = adfuller(series.dropna())
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05,
            'critical_values': result[4]
        }
    
    def engle_granger_test(self, y, x, names=('Y', 'X')):
        """
        Engle-Granger两步法协整检验
        返回：协整统计量、p值、协整系数、残差序列
        """
        # 第一步：OLS回归
        x_const = sm.add_constant(x)
        model = OLS(y, x_const).fit()
        residuals = model.resid
        
        # 第二步：残差ADF检验
        adf_result = adfuller(residuals)
        
        return {
            'coint_statistic': adf_result[0],
            'p_value': adf_result[1],
            'is_cointegrated': adf_result[1] < 0.05,
            'beta': model.params[1],  # 协整系数
            'alpha': model.params[0],  # 截距
            'residuals': residuals,
            'r_squared': model.rsquared,
            'model': model
        }
    
    def analyze_all(self):
        """分析所有候选股票与目标股票的协整关系"""
        if not hasattr(self, 'candidate_data'):
            print("错误：请先调用load_all_data()加载数据")
            return
            
        results = []
        target_prices = self.target_data.dropna()
        target_log = np.log(target_prices)
        
        print(f"\n{'='*60}")
        print(f"协整关系分析结果：嘉友国际 vs 候选股票")
        print(f"{'='*60}")
        print(f"{'股票名称':<12} {'代码':<12} {'协整P值':<10} {'协整关系':<8} {'β系数':<8} {'R²':<6}")
        print(f"{'-'*60}")
        
        for ticker, info in self.candidate_data.items():
            try:
                # 对齐数据
                common_idx = target_prices.index.intersection(info['data'].index)
                if len(common_idx) < 10:
                    continue
                    
                y = np.log(target_prices.loc[common_idx])
                x = np.log(info['data'].loc[common_idx])
                
                # Engle-Granger协整检验
                eg_result = self.engle_granger_test(y, x, ('嘉友国际', info['name']))
                
                # 使用statsmodels的coint函数验证
                score, pvalue, _ = coint(y, x)
                
                result = {
                    'ticker': ticker,
                    'name': info['name'],
                    'eg_pvalue': eg_result['p_value'],
                    'coint_pvalue': pvalue,
                    'beta': eg_result['beta'],
                    'r_squared': eg_result['r_squared'],
                    'is_cointegrated': (eg_result['p_value'] < 0.05) and (pvalue < 0.05),
                    'residuals': eg_result['residuals']
                }
                results.append(result)
                
                status = "✓ 协整" if result['is_cointegrated'] else "✗ 无"
                print(f"{info['name']:<12} {ticker:<12} {min(eg_result['p_value'], pvalue):<10.4f} "
                      f"{status:<8} {eg_result['beta']:<8.2f} {eg_result['r_squared']:<6.2f}")
                      
            except Exception as e:
                print(f"{info['name']:<12} {ticker:<12} 分析失败: {str(e)[:20]}")
        
        self.results_df = pd.DataFrame(results)
        return results
    
    def plot_cointegrated_pairs(self, top_n=3):
        """绘制协整关系最强的股票对"""
        if not hasattr(self, 'results_df') or len(self.results_df) == 0:
            print("无协整分析结果")
            return
            
        # 筛选有协整关系的
        cointegrated = self.results_df[self.results_df['is_cointegrated'] == True]
        if len(cointegrated) == 0:
            print("未发现协整关系")
            return
            
        # 按P值排序
        top_pairs = cointegrated.nsmallest(top_n, 'coint_pvalue')
        
        fig, axes = plt.subplots(2, len(top_pairs), figsize=(15, 8))
        if len(top_pairs) == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (_, row) in enumerate(top_pairs.iterrows()):
            ticker = row['ticker']
            name = row['name']
            
            # 获取对齐数据
            target_prices = self.target_data
            candidate_prices = self.candidate_data[ticker]['data']
            common_idx = target_prices.index.intersection(candidate_prices.index)
            
            y = np.log(target_prices.loc[common_idx])
            x = np.log(candidate_prices.loc[common_idx])
            
            # 绘制价格走势（对数坐标）
            ax1 = axes[0, idx]
            ax1.plot(y.index, y.values, label='嘉友国际', color='red', linewidth=2)
            ax1.plot(x.index, x.values, label=name, color='blue', alpha=0.7)
            ax1.set_title(f'{name} ({ticker})\nP值: {row["coint_pvalue"]:.4f}, β: {row["beta"]:.2f}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 绘制残差（均衡偏离）
            ax2 = axes[1, idx]
            residuals = y - row['beta'] * x - np.mean(y - row['beta'] * x)
            ax2.plot(residuals.index, residuals.values, color='green', linewidth=1.5)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.fill_between(residuals.index, residuals.values, 0, alpha=0.3)
            ax2.set_title(f'残差序列（均衡偏离）')
            ax2.grid(True, alpha=0.3)
            
            # 添加统计信息
            std_resid = np.std(residuals)
            current_deviation = residuals.iloc[-1] / std_resid
            ax2.text(0.05, 0.95, f'当前偏离: {current_deviation:.2f}σ', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('cointegration_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\n图表已保存至: cointegration_analysis.png")
        plt.show()
    
    def generate_trading_signals(self):
        """基于协整残差生成交易信号"""
        if not hasattr(self, 'results_df'):
            return
            
        print(f"\n{'='*60}")
        print("基于协整关系的交易信号")
        print(f"{'='*60}")
        
        for _, row in self.results_df.iterrows():
            if not row['is_cointegrated']:
                continue
                
            ticker = row['ticker']
            name = row['name']
            
            # 获取最新残差
            residuals = row['residuals']
            if len(residuals) == 0:
                continue
                
            latest_residual = residuals.iloc[-1]
            std_residual = np.std(residuals)
            z_score = latest_residual / std_residual
            
            # 交易信号
            if z_score < -2.0:
                signal = "🟢 强烈买入（均值回归）"
            elif z_score < -1.5:
                signal = "🟡 考虑买入"
            elif z_score > 2.0:
                signal = "🔴 强烈卖出（偏离过高）"
            elif z_score > 1.5:
                signal = "🟠 考虑卖出"
            else:
                signal = "⚪ 持有/观望"
            
            print(f"{name:<12} 偏离度: {z_score:>6.2f}σ  {signal}")

# 使用示例
if __name__ == "__main__":
    import statsmodels.api as sm
    
    # 初始化分析器
    analyzer = CointegrationAnalyzer(
        target_ticker='603871.SS',  # 嘉友国际
        start_date='2026-03-01',     # 3月1日开始（覆盖3月13日前后）
        end_date='2026-04-07'        # 4月7日（当前时点）
    )
    
    # 步骤1：加载数据
    if analyzer.load_all_data():
        # 步骤2：协整分析
        results = analyzer.analyze_all()
        
        # 步骤3：可视化
        analyzer.plot_cointegrated_pairs(top_n=3)
        
        # 步骤4：交易信号
        analyzer.generate_trading_signals()
        
        # 步骤5：输出详细报告
        print(f"\n{'='*60}")
        print("详细分析报告")
        print(f"{'='*60}")
        
        # 找出协整关系最强的股票
        if len(analyzer.results_df) > 0:
            cointegrated = analyzer.results_df[analyzer.results_df['is_cointegrated'] == True]
            if len(cointegrated) > 0:
                best_match = cointegrated.loc[cointegrated['coint_pvalue'].idxmin()]
                print(f"\n✓ 最强协整关系: {best_match['name']} ({best_match['ticker']})")
                print(f"  - 协整P值: {best_match['coint_pvalue']:.4f} (<0.05显著)")
                print(f"  - β系数: {best_match['beta']:.2f} (1单位{x}变动，嘉友变动{best_match['beta']:.2f}单位)")
                print(f"  - 解释力R²: {best_match['r_squared']:.2%}")
                print(f"\n  解读: 两只股票存在长期均衡关系，短期偏离后会均值回归。")
                print(f"       当前可构建配对交易策略（Pair Trading）。")
            else:
                print("\n✗ 未发现显著协整关系（P值<0.05）")
                print("  可能原因：")
                print("  1. 考察期过短（协整需要长期均衡关系）")
                print("  2. 期间发生结构性断点（如3月13日流动性危机）")
                print("  3. 候选股票与嘉友国际无内在经济联系")
