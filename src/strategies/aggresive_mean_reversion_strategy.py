# src/strategies/aggressive_mean_reversion_strategy.py

import pandas as pd
import numpy as np
from typing import Literal, Tuple, Dict, Optional
from datetime import datetime, timezone

from src.strategies.base_strategy import BaseStrategy


class AggressiveMeanReversionStrategy(BaseStrategy):
    """
    激进均值回归策略 - 基于布林带的高频交易策略
    
    **核心交易逻辑：**
    1. 价格突破上轨 → SHORT（做空）
    2. 空仓时价格回到中线/下方 → COVER（平空）
    3. 价格跌破下轨 → BUY（做多）
    4. 多仓时价格回到中线/上方 → SELL（平多）
    
    **止损机制：**
    - 单笔持仓亏损达到阈值（默认10%）时强制平仓
    
    **特点：**
    - 支持高频监控（默认1分钟检查）
    - 使用5分钟K线计算技术指标
    - 维护历史数据，累积计算
    - 可配置的止损阈值
    """
    
    # 默认参数
    DEFAULT_BB_PERIOD = 20              # 布林带周期
    DEFAULT_BB_STD_DEV = 2              # 布林带标准差倍数
    DEFAULT_MAX_HISTORY_BARS = 500      # 最大历史K线数量
    DEFAULT_STOP_LOSS_THRESHOLD = 0.10  # 止损阈值（10%）
    DEFAULT_MONITOR_INTERVAL_SECONDS = 60  # 监控间隔（秒）
    
    def __init__(self, 
                 bb_period: int = DEFAULT_BB_PERIOD,
                 bb_std_dev: float = DEFAULT_BB_STD_DEV,
                 max_history_bars: int = DEFAULT_MAX_HISTORY_BARS,
                 stop_loss_threshold: float = DEFAULT_STOP_LOSS_THRESHOLD,
                 monitor_interval_seconds: int = DEFAULT_MONITOR_INTERVAL_SECONDS):
        """
        初始化激进均值回归策略。
        
        Args:
            bb_period: 布林带计算周期
            bb_std_dev: 布林带标准差倍数
            max_history_bars: 最大保留的历史K线数量
            stop_loss_threshold: 止损阈值（例如 0.10 表示亏损10%时止损）
            monitor_interval_seconds: 监控间隔（秒）
        """
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.max_history_bars = max_history_bars
        self.stop_loss_threshold = stop_loss_threshold
        self.monitor_interval_seconds = monitor_interval_seconds
        
        # 历史数据存储：按 ticker 分别存储
        self._history_data: Dict[str, pd.DataFrame] = {}
        
        print(f"💹 AggressiveMeanReversionStrategy 初始化:")
        print(f"   布林带: 周期={bb_period}, 标准差={bb_std_dev}σ")
        print(f"   止损阈值: {stop_loss_threshold*100:.1f}%")
        print(f"   监控间隔: {monitor_interval_seconds}秒")
    
    # ==================== 历史数据管理 ====================
    
    def _merge_data(self, ticker: str, new_df: pd.DataFrame) -> pd.DataFrame:
        """
        将新数据与历史数据合并，去除重复项并按时间排序。
        
        Args:
            ticker: 股票代码
            new_df: 新获取的 OHLCV DataFrame
            
        Returns:
            pd.DataFrame: 合并后的 DataFrame
        """
        if new_df.empty:
            return self._history_data.get(ticker, pd.DataFrame())
        
        if ticker not in self._history_data or self._history_data[ticker].empty:
            merged_df = new_df.copy()
        else:
            history_df = self._history_data[ticker]
            merged_df = pd.concat([history_df, new_df])
            # 去重，保留最新数据
            merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
            merged_df = merged_df.sort_index()
        
        # 限制历史数据大小
        if len(merged_df) > self.max_history_bars:
            merged_df = merged_df.iloc[-self.max_history_bars:]
        
        # 更新历史数据存储
        self._history_data[ticker] = merged_df
        
        return merged_df
    
    def get_history_data(self, ticker: str) -> pd.DataFrame:
        """获取指定 ticker 的历史数据副本。"""
        if ticker in self._history_data:
            return self._history_data[ticker].copy()
        return pd.DataFrame()
    
    def clear_history(self, ticker: Optional[str] = None):
        """清除历史数据。如果 ticker 为 None，清除所有。"""
        if ticker is None:
            self._history_data.clear()
            print("🗑️ 已清除所有历史数据。")
        elif ticker in self._history_data:
            del self._history_data[ticker]
            print(f"🗑️ 已清除 {ticker} 的历史数据。")
    
    def get_history_size(self, ticker: str) -> int:
        """获取指定 ticker 的历史数据条数。"""
        return len(self._history_data.get(ticker, []))
    
    # ==================== 技术指标计算 ====================
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算布林带指标。"""
        df = df.copy()
        df['SMA'] = df['close'].rolling(window=self.bb_period).mean()
        df['STD'] = df['close'].rolling(window=self.bb_period).std()
        df['BB_UPPER'] = df['SMA'] + (df['STD'] * self.bb_std_dev)
        df['BB_LOWER'] = df['SMA'] - (df['STD'] * self.bb_std_dev)
        return df
    
    # ==================== 信号生成逻辑 ====================
    
    def _generate_signal_from_indicators(self, 
                                        latest_close: float,
                                        bb_upper: float,
                                        bb_lower: float,
                                        sma: float,
                                        current_position: float = 0.0,
                                        avg_cost: float = 0.0) -> Tuple[str, int, str]:
        """
        根据技术指标和当前持仓状态生成交易信号。
        
        **新的交易规则：**
        1. 价格突破上轨 → SHORT（无论当前状态）
        2. 持有空仓 + 价格回到中线或以下 → COVER
        3. 价格跌破下轨 → BUY（无论当前状态）
        4. 持有多仓 + 价格回到中线或以上 → SELL
        
        **止损检查：**
        - 如果当前持仓亏损 >= 止损阈值 → 强制平仓
        
        Args:
            latest_close: 最新价格
            bb_upper: 布林带上轨
            bb_lower: 布林带下轨
            sma: 布林带中线（移动平均线）
            current_position: 当前持仓（正数=多仓，负数=空仓，0=无仓位）
            avg_cost: 持仓平均成本
            
        Returns:
            (signal, confidence, reason)
        """
        if pd.isna([latest_close, bb_upper, bb_lower, sma]).any():
            return "HOLD", 0, "技术指标数据不足"
        
        # ===== 止损检查（优先级最高）=====
        if current_position != 0 and avg_cost > 0:
            if current_position > 0:  # 多仓
                loss_pct = (avg_cost - latest_close) / avg_cost
                if loss_pct >= self.stop_loss_threshold:
                    return "SELL", 10, (f"⚠️ 止损触发！多仓亏损 {loss_pct*100:.2f}% "
                                       f"(成本${avg_cost:.2f} → 现价${latest_close:.2f})")
            elif current_position < 0:  # 空仓
                loss_pct = (latest_close - avg_cost) / avg_cost
                if loss_pct >= self.stop_loss_threshold:
                    return "COVER", 10, (f"⚠️ 止损触发！空仓亏损 {loss_pct*100:.2f}% "
                                        f"(成本${avg_cost:.2f} → 现价${latest_close:.2f})")
        
        # ===== 根据价格位置和持仓状态决定信号 =====
        
        # 1. 价格突破上轨 → 做空（或平多后做空）
        if latest_close > bb_upper:
            if current_position > 0:
                # 先平多仓
                return "SELL", 8, (f"价格突破上轨 ${latest_close:.2f} > ${bb_upper:.2f}，"
                                  f"先平多仓，准备做空")
            elif current_position == 0:
                # 开空仓
                return "SHORT", 9, (f"价格突破上轨 ${latest_close:.2f} > ${bb_upper:.2f}，"
                                   f"开空仓")
            else:
                # 已有空仓，继续持有
                return "HOLD", 7, f"已持有空仓，价格在上轨上方 (${latest_close:.2f})"
        
        # 2. 价格跌破下轨 → 做多（或平空后做多）
        elif latest_close < bb_lower:
            if current_position < 0:
                # 先平空仓
                return "COVER", 8, (f"价格跌破下轨 ${latest_close:.2f} < ${bb_lower:.2f}，"
                                   f"先平空仓，准备做多")
            elif current_position == 0:
                # 开多仓
                return "BUY", 9, (f"价格跌破下轨 ${latest_close:.2f} < ${bb_lower:.2f}，"
                                 f"开多仓")
            else:
                # 已有多仓，继续持有
                return "HOLD", 7, f"已持有多仓，价格在下轨下方 (${latest_close:.2f})"
        
        # 3. 价格回到中线附近 → 考虑平仓
        else:
            # 持有空仓 + 价格回到中线或以下 → 平空
            if current_position < 0 and latest_close <= sma:
                profit_pct = (avg_cost - latest_close) / avg_cost
                return "COVER", 8, (f"空仓回归中线，平仓获利 {profit_pct*100:.2f}% "
                                   f"(成本${avg_cost:.2f} → 现价${latest_close:.2f})")
            
            # 持有多仓 + 价格回到中线或以上 → 平多
            elif current_position > 0 and latest_close >= sma:
                profit_pct = (latest_close - avg_cost) / avg_cost
                return "SELL", 8, (f"多仓回归中线，平仓获利 {profit_pct*100:.2f}% "
                                  f"(成本${avg_cost:.2f} → 现价${latest_close:.2f})")
            
            # 价格在布林带中间区域 → 继续持有
            else:
                if current_position > 0:
                    return "HOLD", 5, f"多仓持有中，价格在区间内 (${latest_close:.2f})"
                elif current_position < 0:
                    return "HOLD", 5, f"空仓持有中，价格在区间内 (${latest_close:.2f})"
                else:
                    return "HOLD", 5, (f"无仓位，价格在区间内 "
                                      f"[${bb_lower:.2f}, ${bb_upper:.2f}]")
    
    # ==================== 主接口 ====================
    
    def get_signal(self, 
                   ticker: str,
                   new_data: pd.DataFrame,
                   current_position: float = 0.0,
                   avg_cost: float = 0.0,
                   verbose: bool = True) -> Tuple[Dict, float]:
        """
        获取交易信号。
        
        数据会与历史数据合并后再计算指标，确保有足够的数据点。
        
        Args:
            ticker: 股票代码
            new_data: 新的 OHLCV DataFrame（5分钟K线），索引为时间戳
            current_position: 当前持仓（正数=多仓，负数=空仓，0=无仓位）
            avg_cost: 持仓平均成本
            verbose: 是否打印详细信息
            
        Returns:
            Tuple[signal_dict, current_price]:
                - signal_dict: {'signal': str, 'confidence_score': int, 'reason': str}
                - current_price: 最新价格
        """
        # 1. 合并历史数据和新数据
        df = self._merge_data(ticker, new_data)
        
        if verbose:
            pos_str = f"多仓{current_position:.0f}股" if current_position > 0 else \
                     f"空仓{abs(current_position):.0f}股" if current_position < 0 else "无仓位"
            print(f"📊 {ticker} 数据: {len(df)} 条K线 (新增: {len(new_data)}) | 当前: {pos_str}")
        
        if df.empty:
            return {"signal": "HOLD", "confidence_score": 0, "reason": "No data"}, 0.0
        
        # 2. 计算布林带指标
        df = self._calculate_bollinger_bands(df)
        
        # 🔧 关键修复：立即更新历史数据（包含布林带指标）
        self._history_data[ticker] = df.copy()
        
        # 3. 获取有效数据（去除 NaN）
        df_valid = df.dropna()
        
        min_required = self.bb_period
        if df_valid.empty:
            if verbose:
                print(f"❌ 数据不足，需要至少 {min_required} 条有效数据")
            return {"signal": "HOLD", "confidence_score": 0, 
                    "reason": f"Insufficient data for BB (need {min_required})"}, 0.0
        
        # 4. 获取最新数据点
        latest = df_valid.iloc[-1]
        current_price = latest['close']
        
        # 5. 生成信号（传入持仓信息和平均成本）
        signal, confidence, reason = self._generate_signal_from_indicators(
            current_price, 
            latest['BB_UPPER'], 
            latest['BB_LOWER'], 
            latest['SMA'],
            current_position,
            avg_cost
        )
        
        # 6. 验证信号
        signal = self._validate_signal(signal)
        
        # 7. 打印信息
        if verbose:
            timestamp_str = df_valid.index[-1].strftime('%Y-%m-%d %H:%M') if hasattr(df_valid.index[-1], 'strftime') else str(df_valid.index[-1])
            print(f"   [{timestamp_str}] 价格: ${current_price:.2f} | "
                  f"BB: [${latest['BB_LOWER']:.2f}, ${latest['SMA']:.2f}, ${latest['BB_UPPER']:.2f}]")
            
            # 显示盈亏情况
            if current_position != 0 and avg_cost > 0:
                if current_position > 0:
                    pnl_pct = (current_price - avg_cost) / avg_cost * 100
                    pnl_emoji = "📈" if pnl_pct > 0 else "📉"
                    print(f"   {pnl_emoji} 多仓盈亏: {pnl_pct:+.2f}% (成本: ${avg_cost:.2f})")
                else:
                    pnl_pct = (avg_cost - current_price) / avg_cost * 100
                    pnl_emoji = "📈" if pnl_pct > 0 else "📉"
                    print(f"   {pnl_emoji} 空仓盈亏: {pnl_pct:+.2f}% (成本: ${avg_cost:.2f})")
            
            signal_emoji = {"BUY": "🟢", "SELL": "🔴", "SHORT": "🔻", "COVER": "🔺", "HOLD": "⚪"}.get(signal, "⚪")
            print(f"   {signal_emoji} 信号: {signal} (置信度: {confidence}/10) - {reason}")
        
        return {
            "signal": signal,
            "confidence_score": confidence,
            "reason": reason
        }, current_price
    
    def __str__(self):
        return (f"AggressiveMeanReversionStrategy(BB={self.bb_period}, "
                f"StopLoss={self.stop_loss_threshold*100:.0f}%, "
                f"Monitor={self.monitor_interval_seconds}s)")


# ==================== 配置示例 ====================

# 在 runner 中使用示例：
"""
from src.strategies.aggressive_mean_reversion_strategy import AggressiveMeanReversionStrategy

strategy = AggressiveMeanReversionStrategy(
    bb_period=20,              # 布林带周期
    bb_std_dev=2,              # 标准差倍数
    stop_loss_threshold=0.10,  # 10% 止损
    monitor_interval_seconds=60  # 每分钟检查
)

# 在 LiveEngine 中设置
engine = LiveEngine(
    ticker=TICKER,
    strategy=strategy,
    interval_seconds=60,  # 每60秒运行一次策略
    timeframe=TimeFrame(5, TimeFrameUnit.Minute),  # 使用5分钟K线
    ...
)
"""