# src/strategies/high_frequency_strategy.py

"""
高频交易策略 - High Frequency Mean Reversion

核心改进：
1. 在布林带内部也交易（捕捉小波动）
2. 使用多个阈值级别
3. 快速进出，积累小利润

适合：高波动、震荡市场
"""

from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np


class HighFrequencyStrategy:
    """
    高频均值回归策略
    
    交易规则（3级阈值）：
    
    开仓：
    - 价格 > 90% → 强力做空
    - 价格 > 75% → 温和做空
    - 价格 < 10% → 强力做多
    - 价格 < 25% → 温和做多
    
    平仓：
    - 多仓：价格回到 35% 就卖出
    - 空仓：价格回落到 65% 就平仓
    
    特点：更快进出，捕捉小波动
    """
    
    def __init__(self,
                 bb_period: int = 20,
                 bb_std_dev: float = 2.0,
                 # 开仓阈值（3级）
                 strong_entry: float = 0.90,   # 强力开仓
                 mild_entry: float = 0.75,     # 温和开仓
                 # 平仓阈值
                 exit_threshold: float = 0.65,  # 空仓平仓点
                 stop_loss_threshold: float = 0.08,  # 更紧的止损
                 monitor_interval_seconds: int = 60,
                 max_history_bars: int = 500):
        """
        参数说明：
            strong_entry: 强力开仓阈值（0.90 = 90%）
            mild_entry: 温和开仓阈值（0.75 = 75%）
            exit_threshold: 平仓阈值（0.65 = 65%）
            stop_loss_threshold: 止损（0.08 = 8%，更紧）
        """
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.strong_entry = strong_entry
        self.mild_entry = mild_entry
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.monitor_interval_seconds = monitor_interval_seconds
        self.max_history_bars = max_history_bars
        
        self._history_data: Dict[str, pd.DataFrame] = {}
        
        print(f"🚀 高频策略初始化:")
        print(f"   强力开仓: {strong_entry*100:.0f}%")
        print(f"   温和开仓: {mild_entry*100:.0f}%")
        print(f"   平仓点: {exit_threshold*100:.0f}%")
        print(f"   止损: {stop_loss_threshold*100:.0f}%")
    
    # ==================== 数据管理 ====================
    
    def _merge_data(self, ticker: str, new_df: pd.DataFrame) -> pd.DataFrame:
        """合并新数据与历史数据"""
        if new_df.empty:
            return self._history_data.get(ticker, pd.DataFrame())
        
        if ticker not in self._history_data or self._history_data[ticker].empty:
            merged_df = new_df.copy()
        else:
            history_df = self._history_data[ticker]
            merged_df = pd.concat([history_df, new_df])
            merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
            merged_df = merged_df.sort_index()
        
        if len(merged_df) > self.max_history_bars:
            merged_df = merged_df.iloc[-self.max_history_bars:]
        
        return merged_df
    
    def get_history_data(self, ticker: str) -> pd.DataFrame:
        """获取历史数据"""
        if ticker in self._history_data:
            return self._history_data[ticker].copy()
        return pd.DataFrame()
    
    # ==================== 技术指标 ====================
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()
        
        # 布林带
        df['SMA'] = df['close'].rolling(window=self.bb_period).mean()
        df['STD'] = df['close'].rolling(window=self.bb_period).std()
        df['BB_UPPER'] = df['SMA'] + (df['STD'] * self.bb_std_dev)
        df['BB_LOWER'] = df['SMA'] - (df['STD'] * self.bb_std_dev)
        
        # 布林带位置
        df['BB_WIDTH'] = df['BB_UPPER'] - df['BB_LOWER']
        df['BB_POSITION'] = (df['close'] - df['BB_LOWER']) / df['BB_WIDTH']
        
        # 🔥 新增：价格动量（检测快速变化）
        df['PRICE_CHANGE'] = df['close'].pct_change()
        df['MOMENTUM'] = df['close'].diff(3)  # 3根K线的价格变化
        
        return df
    
    # ==================== 信号生成 ====================
    
    def _generate_signal(self,
                        price: float,
                        bb_position: float,
                        momentum: float,
                        current_position: float = 0.0,
                        avg_cost: float = 0.0) -> Tuple[str, int, str]:
        """高频信号生成"""
        
        if pd.isna([price, bb_position]).any():
            return "HOLD", 0, "数据不足"
        
        # ===== 止损（更紧）=====
        if current_position != 0 and avg_cost > 0:
            if current_position > 0:
                loss_pct = (avg_cost - price) / avg_cost
                if loss_pct >= self.stop_loss_threshold:
                    return "SELL", 10, f"⚠️ 止损 {loss_pct*100:.2f}%"
            else:
                loss_pct = (price - avg_cost) / avg_cost
                if loss_pct >= self.stop_loss_threshold:
                    return "COVER", 10, f"⚠️ 止损 {loss_pct*100:.2f}%"
        
        # ===== 高频交易逻辑 =====
        
        # 无仓位时
        if current_position == 0:
            # 强力做空信号
            if bb_position > self.strong_entry:
                return "SHORT", 10, f"🔥 强力做空！位置 {bb_position*100:.0f}%"
            
            # 温和做空信号
            elif bb_position > self.mild_entry:
                return "SHORT", 7, f"📉 温和做空 位置 {bb_position*100:.0f}%"
            
            # 强力做多信号
            elif bb_position < (1 - self.strong_entry):
                return "BUY", 10, f"🔥 强力做多！位置 {bb_position*100:.0f}%"
            
            # 温和做多信号
            elif bb_position < (1 - self.mild_entry):
                return "BUY", 7, f"📈 温和做多 位置 {bb_position*100:.0f}%"
        
        # 持有多仓
        elif current_position > 0:
            # 快速平仓（回到 35%）
            if bb_position > (1 - self.exit_threshold):
                pnl_pct = (price - avg_cost) / avg_cost * 100
                return "SELL", 8, f"💰 多仓平仓 盈亏{pnl_pct:+.2f}% 位置{bb_position*100:.0f}%"
            
            # 反转做空（价格冲到上部）
            elif bb_position > self.strong_entry:
                return "SELL", 9, f"🔄 反转！多→空 位置{bb_position*100:.0f}%"
        
        # 持有空仓
        elif current_position < 0:
            # 快速平仓（回落到 65%）
            if bb_position < self.exit_threshold:
                pnl_pct = (avg_cost - price) / avg_cost * 100
                return "COVER", 8, f"💰 空仓平仓 盈亏{pnl_pct:+.2f}% 位置{bb_position*100:.0f}%"
            
            # 反转做多（价格跌到下部）
            elif bb_position < (1 - self.strong_entry):
                return "COVER", 9, f"🔄 反转！空→多 位置{bb_position*100:.0f}%"
        
        return "HOLD", 2, f"持仓中 位置{bb_position*100:.0f}%"
    
    # ==================== 主接口 ====================
    
    def get_signal(self,
                   ticker: str,
                   new_data: pd.DataFrame,
                   current_position: float = 0.0,
                   avg_cost: float = 0.0,
                   verbose: bool = True) -> Tuple[Dict, float]:
        """获取交易信号"""
        # 1. 合并数据
        df = self._merge_data(ticker, new_data)
        
        if verbose:
            pos_str = f"多{current_position:.0f}" if current_position > 0 else \
                     f"空{abs(current_position):.0f}" if current_position < 0 else "无"
            print(f"🚀 {ticker}: {len(df)} K线 | {pos_str}")
        
        if df.empty:
            return {"signal": "HOLD", "confidence_score": 0, "reason": "No data"}, 0.0
        
        # 2. 计算指标
        df = self._calculate_indicators(df)
        
        # 3. 更新历史
        self._history_data[ticker] = df.copy()
        
        # 4. 获取有效数据
        df_valid = df.dropna()
        
        if df_valid.empty or len(df_valid) < self.bb_period:
            return {"signal": "HOLD", "confidence_score": 0, "reason": "数据不足"}, 0.0
        
        # 5. 最新数据
        latest = df_valid.iloc[-1]
        price = latest['close']
        
        # 6. 生成信号
        signal, confidence, reason = self._generate_signal(
            price,
            latest['BB_POSITION'],
            latest.get('MOMENTUM', 0),
            current_position,
            avg_cost
        )
        
        # 7. 打印
        if verbose:
            print(f"   ${price:.2f} | BB {latest['BB_POSITION']*100:.0f}% | "
                  f"{signal} - {reason}")
        
        return {
            "signal": signal,
            "confidence_score": confidence,
            "reason": reason
        }, price