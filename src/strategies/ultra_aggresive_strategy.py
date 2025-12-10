# src/strategies/ultra_aggressive_strategy.py

"""
超激进动态策略 - Ultra Aggressive Dynamic Strategy

核心特点：
1. 动态调整阈值（根据波动率）
2. 允许频繁切换方向
3. 多级加仓/减仓
4. 捕捉所有可能的机会

⚠️ 警告：高风险！适合模拟盘测试
"""

from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np


class UltraAggressiveStrategy:
    """
    超激进动态策略
    
    特点：
    - 波动大时放宽阈值，波动小时收紧阈值
    - 价格在布林带任何位置都可能交易
    - 快速止盈止损
    - 最大化捕捉小波动
    
    示例（以你的18:20数据）：
    - 波动率高 → 70% 位置就做空
    - 波动率低 → 85% 位置才做空
    """
    
    def __init__(self,
                 bb_period: int = 20,
                 bb_std_dev: float = 2.0,
                 # 动态阈值范围
                 min_entry_threshold: float = 0.70,  # 最激进
                 max_entry_threshold: float = 0.90,  # 最保守
                 quick_exit_threshold: float = 0.55, # 快速平仓
                 stop_loss_threshold: float = 0.06,  # 6% 止损
                 take_profit_threshold: float = 0.03, # 3% 止盈
                 monitor_interval_seconds: int = 60,
                 max_history_bars: int = 500):
        """
        参数说明：
            min_entry_threshold: 高波动时的开仓阈值（0.70 = 70%）
            max_entry_threshold: 低波动时的开仓阈值（0.90 = 90%）
            quick_exit_threshold: 快速平仓点（0.55 = 55%）
            take_profit_threshold: 止盈阈值（0.03 = 3%）
        """
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.min_entry = min_entry_threshold
        self.max_entry = max_entry_threshold
        self.quick_exit = quick_exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.take_profit_threshold = take_profit_threshold
        self.monitor_interval_seconds = monitor_interval_seconds
        self.max_history_bars = max_history_bars
        
        self._history_data: Dict[str, pd.DataFrame] = {}
        
        print(f"⚡ 超激进策略初始化:")
        print(f"   动态开仓: {min_entry_threshold*100:.0f}%-{max_entry_threshold*100:.0f}%")
        print(f"   快速平仓: {quick_exit_threshold*100:.0f}%")
        print(f"   止盈/止损: {take_profit_threshold*100:.0f}%/{stop_loss_threshold*100:.0f}%")
    
    # ==================== 数据管理 ====================
    
    def _merge_data(self, ticker: str, new_df: pd.DataFrame) -> pd.DataFrame:
        """合并数据"""
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
        
        # 🔥 波动率（用于动态调整阈值）
        df['VOLATILITY'] = df['close'].rolling(10).std() / df['close'].rolling(10).mean()
        
        # 🔥 价格动量
        df['MOMENTUM_3'] = df['close'].diff(3)
        df['MOMENTUM_5'] = df['close'].diff(5)
        
        # 🔥 布林带宽度变化（挤压/扩张）
        df['BB_WIDTH_CHANGE'] = df['BB_WIDTH'].pct_change(5)
        
        return df
    
    def _get_dynamic_threshold(self, volatility: float) -> float:
        """
        根据波动率动态计算开仓阈值
        
        波动率高 → 阈值低（更早开仓）
        波动率低 → 阈值高（更晚开仓）
        """
        if pd.isna(volatility):
            return 0.80  # 默认值
        
        # 标准化波动率（假设正常范围 0.01-0.05）
        normalized_vol = np.clip((volatility - 0.01) / 0.04, 0, 1)
        
        # 波动率越高，阈值越低
        threshold = self.max_entry - (normalized_vol * (self.max_entry - self.min_entry))
        
        return threshold
    
    # ==================== 信号生成 ====================
    
    def _generate_signal(self,
                        price: float,
                        bb_position: float,
                        volatility: float,
                        momentum_3: float,
                        current_position: float = 0.0,
                        avg_cost: float = 0.0) -> Tuple[str, int, str]:
        """超激进信号生成"""
        
        if pd.isna([price, bb_position]).any():
            return "HOLD", 0, "数据不足"
        
        # 动态阈值
        dynamic_threshold = self._get_dynamic_threshold(volatility)
        
        # ===== 快速止盈/止损 =====
        if current_position != 0 and avg_cost > 0:
            if current_position > 0:
                pnl_pct = (price - avg_cost) / avg_cost
                # 止盈
                if pnl_pct >= self.take_profit_threshold:
                    return "SELL", 10, f"💰 止盈 {pnl_pct*100:.2f}%"
                # 止损
                if pnl_pct <= -self.stop_loss_threshold:
                    return "SELL", 10, f"⚠️ 止损 {pnl_pct*100:.2f}%"
            else:
                pnl_pct = (avg_cost - price) / avg_cost
                # 止盈
                if pnl_pct >= self.take_profit_threshold:
                    return "COVER", 10, f"💰 止盈 {pnl_pct*100:.2f}%"
                # 止损
                if pnl_pct <= -self.stop_loss_threshold:
                    return "COVER", 10, f"⚠️ 止损 {pnl_pct*100:.2f}%"
        
        # ===== 超激进交易逻辑 =====
        
        # 无仓位
        if current_position == 0:
            # 做空信号（动态阈值）
            if bb_position > dynamic_threshold:
                confidence = 10 if bb_position > 0.95 else 8
                return "SHORT", confidence, (f"⚡ 做空 位置{bb_position*100:.0f}% "
                                            f"阈值{dynamic_threshold*100:.0f}%")
            
            # 做多信号（动态阈值）
            elif bb_position < (1 - dynamic_threshold):
                confidence = 10 if bb_position < 0.05 else 8
                return "BUY", confidence, (f"⚡ 做多 位置{bb_position*100:.0f}% "
                                          f"阈值{(1-dynamic_threshold)*100:.0f}%")
        
        # 持有多仓
        elif current_position > 0:
            # 快速平仓
            if bb_position > self.quick_exit:
                return "SELL", 8, f"🔄 快速平多 位置{bb_position*100:.0f}%"
            
            # 反转做空
            elif bb_position > 0.95:
                return "SELL", 10, f"🔄 反转做空 位置{bb_position*100:.0f}%"
            
            # 动量反转
            elif momentum_3 < 0 and bb_position > 0.60:
                return "SELL", 7, f"📉 动量反转 位置{bb_position*100:.0f}%"
        
        # 持有空仓
        elif current_position < 0:
            # 快速平仓
            if bb_position < (1 - self.quick_exit):
                return "COVER", 8, f"🔄 快速平空 位置{bb_position*100:.0f}%"
            
            # 反转做多
            elif bb_position < 0.05:
                return "COVER", 10, f"🔄 反转做多 位置{bb_position*100:.0f}%"
            
            # 动量反转
            elif momentum_3 > 0 and bb_position < 0.40:
                return "COVER", 7, f"📈 动量反转 位置{bb_position*100:.0f}%"
        
        return "HOLD", 1, f"观望 位置{bb_position*100:.0f}%"
    
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
            print(f"⚡ {ticker}: {len(df)} K | {pos_str}")
        
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
            latest.get('VOLATILITY', 0.02),
            latest.get('MOMENTUM_3', 0),
            current_position,
            avg_cost
        )
        
        # 7. 打印
        if verbose:
            vol_str = f"波动{latest.get('VOLATILITY', 0)*100:.2f}%"
            print(f"   ${price:.2f} | BB{latest['BB_POSITION']*100:.0f}% | {vol_str}")
            print(f"   {signal} - {reason}")
        
        return {
            "signal": signal,
            "confidence_score": confidence,
            "reason": reason
        }, price