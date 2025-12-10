# src/strategies/moderate_aggressive_dynamic_strategy.py

"""
动态阈值温和进取策略 - Moderate Aggressive with Dynamic Thresholds

核心改进：
1. 根据波动率动态调整阈值
2. 低波动时 → 降低阈值，捕捉小波动
3. 高波动时 → 提高阈值，避免假信号

解决问题：14:30-15:55 横盘期间也能交易
"""

from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np


class ModerateAggressiveDynamicStrategy:
    """
    动态阈值温和进取策略
    
    根据布林带宽度（波动率）动态调整交易阈值：
    - 布林带宽 > 2% → 高波动，使用标准阈值（0.85）
    - 布林带宽 < 1% → 低波动，降低阈值到 0.70（更激进）
    - 1%-2% 之间 → 线性插值
    
    示例：
    正常波动：price > 85% 才做空
    横盘期间：price > 70% 就做空（更早捕捉）
    """
    
    def __init__(self,
                 bb_period: int = 20,
                 bb_std_dev: float = 2.0,
                 base_entry_threshold: float = 0.85,    # 基础开仓阈值
                 aggressive_entry_threshold: float = 0.70,  # 横盘期激进阈值
                 exit_threshold: float = 0.60,
                 stop_loss_threshold: float = 0.10,
                 high_volatility_threshold: float = 0.02,  # 2% 布林带宽度
                 low_volatility_threshold: float = 0.01,   # 1% 布林带宽度
                 monitor_interval_seconds: int = 60,
                 max_history_bars: int = 500):
        """
        参数说明：
            base_entry_threshold: 正常波动时的开仓阈值（0.85 = 85%）
            aggressive_entry_threshold: 低波动时的开仓阈值（0.70 = 70%）
            high_volatility_threshold: 高波动判定（布林带宽度 > 2%）
            low_volatility_threshold: 低波动判定（布林带宽度 < 1%）
        """
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.base_entry_threshold = base_entry_threshold
        self.aggressive_entry_threshold = aggressive_entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.high_volatility_threshold = high_volatility_threshold
        self.low_volatility_threshold = low_volatility_threshold
        self.monitor_interval_seconds = monitor_interval_seconds
        self.max_history_bars = max_history_bars
        
        self._history_data: Dict[str, pd.DataFrame] = {}
        
        print(f"📊 动态阈值温和进取策略初始化:")
        print(f"   基础开仓阈值: {base_entry_threshold*100:.0f}%")
        print(f"   横盘开仓阈值: {aggressive_entry_threshold*100:.0f}%")
        print(f"   平仓阈值: {exit_threshold*100:.0f}%")
        print(f"   高波动阈值: {high_volatility_threshold*100:.1f}%")
        print(f"   低波动阈值: {low_volatility_threshold*100:.1f}%")
    
    # ==================== 数据管理 ====================
    
    def _merge_data(self, ticker: str, new_df: pd.DataFrame) -> pd.DataFrame:
        """合并新数据与历史数据"""
        if new_df.empty:
            return self._history_data.get(ticker, pd.DataFrame())
        
        if ticker not in self._history_data or self._history_data[ticker].empty:
            merged_df = new_df.copy()
        else:
            merged_df = pd.concat([self._history_data[ticker], new_df])
            merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
            merged_df = merged_df.sort_index()
        
        if len(merged_df) > self.max_history_bars:
            merged_df = merged_df.iloc[-self.max_history_bars:]
        
        return merged_df
    
    def get_history_data(self, ticker: str) -> pd.DataFrame:
        """获取历史数据"""
        return self._history_data.get(ticker, pd.DataFrame()).copy()
    
    # ==================== 技术指标 ====================
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算布林带和波动率指标"""
        df = df.copy()
        
        # 布林带
        df['SMA'] = df['close'].rolling(window=self.bb_period).mean()
        df['STD'] = df['close'].rolling(window=self.bb_period).std()
        df['BB_UPPER'] = df['SMA'] + (df['STD'] * self.bb_std_dev)
        df['BB_LOWER'] = df['SMA'] - (df['STD'] * self.bb_std_dev)
        
        # 🆕 布林带宽度（衡量波动率）
        df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['SMA']
        
        # 🆕 布林带位置
        df['BB_POSITION'] = (df['close'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'])
        
        return df
    
    # ==================== 动态阈值计算 ====================
    
    def _calculate_dynamic_threshold(self, bb_width: float) -> float:
        """
        根据布林带宽度动态计算开仓阈值
        
        Args:
            bb_width: 布林带宽度（百分比）
        
        Returns:
            动态阈值（0.70-0.85之间）
        """
        if bb_width >= self.high_volatility_threshold:
            # 高波动 → 使用标准阈值
            return self.base_entry_threshold
        elif bb_width <= self.low_volatility_threshold:
            # 低波动 → 使用激进阈值
            return self.aggressive_entry_threshold
        else:
            # 中等波动 → 线性插值
            ratio = (bb_width - self.low_volatility_threshold) / \
                   (self.high_volatility_threshold - self.low_volatility_threshold)
            return self.aggressive_entry_threshold + \
                   ratio * (self.base_entry_threshold - self.aggressive_entry_threshold)
    
    # ==================== 信号生成 ====================
    
    def _generate_signal(self,
                        price: float,
                        bb_upper: float,
                        bb_lower: float,
                        sma: float,
                        bb_position: float,
                        bb_width: float,
                        current_position: float = 0.0,
                        avg_cost: float = 0.0) -> Tuple[str, int, str]:
        """
        根据布林带位置和波动率生成信号
        """
        if pd.isna([price, bb_upper, bb_lower, sma, bb_position, bb_width]).any():
            return "HOLD", 0, "数据不足"
        
        # 🆕 计算动态阈值
        dynamic_entry_threshold = self._calculate_dynamic_threshold(bb_width)
        
        # 止损检查
        if current_position != 0 and avg_cost > 0:
            if current_position > 0:
                loss_pct = (avg_cost - price) / avg_cost
                if loss_pct >= self.stop_loss_threshold:
                    return "SELL", 10, f"⚠️ 止损！多仓亏损 {loss_pct*100:.2f}%"
            elif current_position < 0:
                loss_pct = (price - avg_cost) / avg_cost
                if loss_pct >= self.stop_loss_threshold:
                    return "COVER", 10, f"⚠️ 止损！空仓亏损 {loss_pct*100:.2f}%"
        
        # 🔥 接近上轨 → 做空（动态阈值）
        if bb_position > dynamic_entry_threshold:
            if current_position <= 0:
                volatility_label = "低波动" if bb_width < self.low_volatility_threshold else \
                                 "高波动" if bb_width > self.high_volatility_threshold else "中波动"
                return "SHORT", 8, (f"价格接近上轨！位置 {bb_position*100:.1f}% "
                                   f"(动态阈值 {dynamic_entry_threshold*100:.0f}%, {volatility_label})")
        
        # 🔥 空仓回调 → 平空
        if current_position < 0:
            if bb_position < self.exit_threshold:
                return "COVER", 7, f"空仓获利平仓！位置回到 {bb_position*100:.1f}%"
        
        # 🔥 接近下轨 → 做多（动态阈值）
        if bb_position < (1 - dynamic_entry_threshold):
            if current_position >= 0:
                volatility_label = "低波动" if bb_width < self.low_volatility_threshold else \
                                 "高波动" if bb_width > self.high_volatility_threshold else "中波动"
                return "BUY", 8, (f"价格接近下轨！位置 {bb_position*100:.1f}% "
                                 f"(动态阈值 {(1-dynamic_entry_threshold)*100:.0f}%, {volatility_label})")
        
        # 🔥 多仓回调 → 平多
        if current_position > 0:
            if bb_position > (1 - self.exit_threshold):
                return "SELL", 7, f"多仓获利平仓！位置回到 {bb_position*100:.1f}%"
        
        # 持有
        return "HOLD", 3, f"价格在区间内 {bb_position*100:.1f}% (动态阈值 {dynamic_entry_threshold*100:.0f}%)"
    
    # ==================== 主接口 ====================
    
    def get_signal(self,
                   ticker: str,
                   new_data: pd.DataFrame,
                   current_position: float = 0.0,
                   avg_cost: float = 0.0,
                   verbose: bool = True,
                   is_market_close: bool = False,
                   current_time_et: pd.Timestamp = None) -> Tuple[Dict, float]:
        """获取交易信号"""
        
        # 收盘强制平仓
        if is_market_close and current_position != 0:
            close_signal = 'SELL' if current_position > 0 else 'COVER'
            reason = f"🔔 市场收盘 - 强制平仓！持仓: {current_position:.0f} 股"
            
            if verbose:
                print(f"⚠️ 收盘平仓: {close_signal} | {reason}")
            
            return {
                "signal": close_signal,
                "confidence_score": 10,
                "reason": reason
            }, 0.0
        
        # 15:50后禁止新开仓
        if current_time_et is not None:
            if current_time_et.hour == 15 and current_time_et.minute >= 50:
                if current_position == 0:
                    return {
                        "signal": "HOLD",
                        "confidence_score": 0,
                        "reason": "⏰ 接近收盘，禁止新开仓"
                    }, 0.0
        
        # 合并数据
        df = self._merge_data(ticker, new_data)
        
        if verbose:
            pos_str = f"多{current_position:.0f}股" if current_position > 0 else \
                     f"空{abs(current_position):.0f}股" if current_position < 0 else "无仓"
            print(f"📊 {ticker}: {len(df)} 条K线 | {pos_str}")
        
        if df.empty:
            return {"signal": "HOLD", "confidence_score": 0, "reason": "No data"}, 0.0
        
        # 计算指标
        df = self._calculate_bollinger_bands(df)
        
        # 更新历史数据
        self._history_data[ticker] = df.copy()
        
        # 获取有效数据
        df_valid = df.dropna()
        
        if df_valid.empty or len(df_valid) < self.bb_period:
            if verbose:
                print(f"❌ 数据不足（需要 {self.bb_period} 条）")
            return {"signal": "HOLD", "confidence_score": 0, "reason": "数据不足"}, 0.0
        
        # 获取最新数据
        latest = df_valid.iloc[-1]
        price = latest['close']
        
        # 生成信号
        signal, confidence, reason = self._generate_signal(
            price,
            latest['BB_UPPER'],
            latest['BB_LOWER'],
            latest['SMA'],
            latest['BB_POSITION'],
            latest['BB_WIDTH'],  # 🆕 传入布林带宽度
            current_position,
            avg_cost
        )
        
        if verbose and signal != 'HOLD':
            print(f"💡 信号: {signal} | 置信度: {confidence} | {reason}")
        
        return {
            "signal": signal,
            "confidence_score": confidence,
            "reason": reason
        }, price