# alpha_engine/01_build_meta_crowding.py

import os
import sys
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from functools import reduce

class FactorCrowdingMonitor:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.factors_dir = os.path.join(self.base_dir, "factors")
        
        # 终极拥挤度元特征输出路径
        self.output_path = os.path.join(self.factors_dir, "meta_crowding_features.parquet")
        
        # 需要监控拥挤度的目标因子 (不包含本身就是估值类的因子如 qarp, pure_bp)
        self.target_factors =['pead_alpha', 'reversal_alpha', 'max_ret_20d', 'amihud_20d', 'sales_yoy']
        
        # 我们的“估值锚”：用 Layer 2 算好的纯净 BP (市净率倒数)
        # BP 越大越便宜，BP 越小越贵
        self.valuation_anchor = 'pure_bp'

    def load_and_merge_factors(self) -> pd.DataFrame:
        logger.info("📡 1/3 Loading Factor Armory for Crowding Detection...")
        
        # 加载基础因子
        df_pead = pd.read_parquet(os.path.join(self.factors_dir, "pead_alpha.parquet"))
        df_rev = pd.read_parquet(os.path.join(self.factors_dir, "reversal_alpha.parquet"))
        df_micro = pd.read_parquet(os.path.join(self.factors_dir, "micro_features.parquet"))
        df_fund = pd.read_parquet(os.path.join(self.factors_dir, "pure_fundamentals.parquet"))
        
        # 合并为一个巨大特征宽表
        data_frames =[df_pead, df_rev, df_micro, df_fund]
        df_master = reduce(lambda left, right: pd.merge(left, right, on=['trade_date', 'ts_code'], how='inner'), data_frames)
        
        return df_master.dropna()

    def calculate_crowding_spread(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("🧮 2/3 Calculating AQR Valuation Spreads (The Crowding Sensors)...")
        
        
        
        # 存储每日的大盘级“拥挤度特征”
        crowding_records =[]
        
        # 按天计算每个因子的多空估值价差
        for trade_date, group in tqdm(df.groupby('trade_date'), desc="Measuring Crowding"):
            daily_record = {'trade_date': trade_date}
            
            for factor in self.target_factors:
                # 找到该因子得分最高 20% (多头拥挤区) 和最低 20% (空头区)
                q_top = group[factor].quantile(0.80)
                q_bot = group[factor].quantile(0.20)
                
                top_group = group[group[factor] >= q_top]
                bot_group = group[group[factor] <= q_bot]
                
                if len(top_group) == 0 or len(bot_group) == 0:
                    daily_record[f"{factor}_spread"] = np.nan
                    continue
                
                # 计算估值价差 Spread
                # 因为 BP 越大代表越便宜。如果 Top 组被买得极度昂贵，它的 BP 会极低。
                # Spread = Top组平均 BP - Bottom组平均 BP
                # 结论：Spread 越负 (越小)，说明该因子越拥挤、越昂贵、随时崩盘！
                spread = top_group[self.valuation_anchor].median() - bot_group[self.valuation_anchor].median()
                daily_record[f"{factor}_spread"] = spread
                
            crowding_records.append(daily_record)
            
        df_spreads = pd.DataFrame(crowding_records)
        df_spreads = df_spreads.set_index('trade_date').sort_index()
        
        # ==========================================
        # 核心防线：滚动 Z-Score (识别历史极值)
        # ==========================================
        logger.info("🌊 Smoothing Spreads and Z-Scoring against 3-Year History...")
        df_crowding_z = pd.DataFrame(index=df_spreads.index)
        
        for factor in self.target_factors:
            col_name = f"{factor}_spread"
            
            # 使用 20日 EMA 平滑日常波动，防止单日异动
            smoothed_spread = df_spreads[col_name].ewm(span=20, min_periods=1).mean()
            
            # 回看过去 750 个交易日 (3年) 的均值和标准差，计算当前拥挤度在历史上的极值水平
            # 加负号 (-1.0)：因为 Spread 越小代表越拥挤。乘以负号后，Z-Score 越大 (> 2.0) 代表越拥挤！
            roll_mean = smoothed_spread.rolling(750, min_periods=252).mean()
            roll_std = smoothed_spread.rolling(750, min_periods=252).std()
            
            z_score = -1.0 * (smoothed_spread - roll_mean) / (roll_std + 1e-8)
            
            # 命名为专属的 crowding 特征
            df_crowding_z[f"{factor}_crowding"] = z_score
            
        # 填充前期的 NaN (用 0 代表不拥挤)
        df_crowding_z = df_crowding_z.fillna(0.0).reset_index()
        
        return df_crowding_z

    def run_pipeline(self):
        df_master = self.load_and_merge_factors()
        df_crowding = self.calculate_crowding_spread(df_master)
        
        logger.info("🔗 3/3 Broadcasting Macro Crowding Features to Individual Stocks...")
        # 将每日的大盘拥挤度特征，广播 (Broadcast) 贴到每一只股票上！
        # 这样 LightGBM 在看某只股票的 PEAD 得分时，同时也能看到今天全市场 PEAD 的拥挤度！
        df_final = pd.merge(df_master[['trade_date', 'ts_code']], df_crowding, on='trade_date', how='left')
        
        df_final.to_parquet(self.output_path, engine='pyarrow', index=False)
        
        logger.success("="*60)
        logger.success("✅ FACTOR CROWDING METRICS BUILT SUCCESSFULLY!")
        logger.success(f"💾 Saved {len(self.target_factors)} Meta-Features to: {self.output_path}")
        logger.success("="*60)

if __name__ == "__main__":
    logger.add("meta_crowding.log", rotation="50 MB", enqueue=True)
    FactorCrowdingMonitor().run_pipeline()