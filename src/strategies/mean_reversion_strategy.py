# src/strategies/mean_reversion_strategy.py

import pandas as pd
import numpy as np
from typing import Literal, Tuple, Dict, Optional
from datetime import datetime, timezone

# 导入基类
from src.strategies.base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    均值回归策略类 - 使用纯数学计算，基于布林带和 RSI 指标。
    
    特点：
    1. 不依赖 data_fetcher，数据通过参数传入
    2. 维护历史数据，每次调用时合并新数据
    3. 纯粹的信号生成器：数据输入 → 信号输出
    
    交易规则：
    1. BUY (买入)：当价格跌破布林带下轨 AND RSI < 30 (超卖)
    2. SELL (卖出)：当价格突破布林带上轨 OR RSI > 70 (超买)
    3. HOLD (观望)：其他情况
    """
    
    # 默认参数
    DEFAULT_BB_PERIOD = 20      # 布林带周期
    DEFAULT_BB_STD_DEV = 2      # 布林带标准差倍数
    DEFAULT_RSI_WINDOW = 14     # RSI 窗口期
    DEFAULT_RSI_OVERSOLD = 30   # RSI 超卖阈值
    DEFAULT_RSI_OVERBOUGHT = 70 # RSI 超买阈值
    DEFAULT_MAX_HISTORY_BARS = 500  # 最大保留历史K线数量
    
    def __init__(self, 
                 bb_period: int = DEFAULT_BB_PERIOD,
                 bb_std_dev: float = DEFAULT_BB_STD_DEV,
                 rsi_window: int = DEFAULT_RSI_WINDOW,
                 rsi_oversold: float = DEFAULT_RSI_OVERSOLD,
                 rsi_overbought: float = DEFAULT_RSI_OVERBOUGHT,
                 max_history_bars: int = DEFAULT_MAX_HISTORY_BARS):
        """
        初始化均值回归策略。
        
        Args:
            bb_period: 布林带计算周期
            bb_std_dev: 布林带标准差倍数
            rsi_window: RSI 计算窗口
            rsi_oversold: RSI 超卖阈值
            rsi_overbought: RSI 超买阈值
            max_history_bars: 最大保留的历史K线数量
        """
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.max_history_bars = max_history_bars
        
        # 历史数据存储：按 ticker 分别存储
        self._history_data: Dict[str, pd.DataFrame] = {}
        
        print(f"📊 MeanReversionStrategy 初始化: BB({bb_period}, {bb_std_dev}σ), RSI({rsi_window}), "
              f"超卖<{rsi_oversold}, 超买>{rsi_overbought}, 最大历史={max_history_bars}")
    
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
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算布林带指标。"""
        df = df.copy()
        df['SMA'] = df['close'].rolling(window=self.bb_period).mean()
        df['STD'] = df['close'].rolling(window=self.bb_period).std()
        df['BB_UPPER'] = df['SMA'] + (df['STD'] * self.bb_std_dev)
        df['BB_LOWER'] = df['SMA'] - (df['STD'] * self.bb_std_dev)
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 RSI 指标。"""
        df = df.copy()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        RS = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + RS))
        return df
    
    def _generate_signal_from_indicators(self, 
                                        latest_close: float,
                                        bb_upper: float,
                                        bb_lower: float,
                                        sma: float,
                                        rsi: float) -> Tuple[Literal["BUY", "SELL", "HOLD"], int, str]:
        """根据技术指标生成交易信号。"""
        if pd.isna([latest_close, bb_upper, bb_lower, sma, rsi]).any():
            return "HOLD", 0, "技术指标数据不足，无法计算信号"
        
        # BUY: 价格跌破下轨 AND RSI 超卖
        if latest_close < bb_lower and rsi < self.rsi_oversold:
            return "BUY", 9, (f"价格 ${latest_close:.2f} 跌破布林带下轨 ${bb_lower:.2f}，"
                             f"且 RSI={rsi:.1f} < {self.rsi_oversold} (超卖)")
        
        # BUY (弱): 仅价格跌破下轨
        elif latest_close < bb_lower:
            return "BUY", 6, f"价格 ${latest_close:.2f} 跌破布林带下轨 ${bb_lower:.2f}"
        
        # SELL: 价格突破上轨 OR RSI 超买
        elif latest_close > bb_upper or rsi > self.rsi_overbought:
            if latest_close > bb_upper and rsi > self.rsi_overbought:
                return "SELL", 8, (f"价格 ${latest_close:.2f} 突破布林带上轨 ${bb_upper:.2f}，"
                                  f"且 RSI={rsi:.1f} > {self.rsi_overbought} (超买)")
            elif latest_close > bb_upper:
                return "SELL", 7, f"价格 ${latest_close:.2f} 突破布林带上轨 ${bb_upper:.2f}"
            else:
                return "SELL", 7, f"RSI={rsi:.1f} > {self.rsi_overbought} (超买)"
        
        # HOLD
        else:
            return "HOLD", 5, (f"价格 ${latest_close:.2f} 在布林带区间内 "
                              f"[${bb_lower:.2f}, ${bb_upper:.2f}]，RSI={rsi:.1f}")
    
    def get_signal(self, 
                   ticker: str,
                   new_data: pd.DataFrame,
                   verbose: bool = True) -> Tuple[Dict, float]:
        """
        获取交易信号。
        
        数据会与历史数据合并后再计算指标，确保有足够的数据点。
        
        Args:
            ticker: 股票代码
            new_data: 新的 OHLCV DataFrame，索引为时间戳，
                      必须包含 'open', 'high', 'low', 'close', 'volume' 列
            verbose: 是否打印详细信息
            
        Returns:
            Tuple[signal_dict, current_price]:
                - signal_dict: {'signal': str, 'confidence_score': int, 'reason': str}
                - current_price: 最新价格
        """
        # 1. 合并历史数据和新数据
        df = self._merge_data(ticker, new_data)
        
        if verbose:
            print(f"📊 {ticker} 数据: {len(df)} 条K线 (新增: {len(new_data)})")
        
        if df.empty:
            return {"signal": "HOLD", "confidence_score": 0, "reason": "No data"}, 0.0
        
        # 2. 计算技术指标
        df = self._calculate_bollinger_bands(df)
        df = self._calculate_rsi(df)
        
        # 3. 获取有效数据（去除 NaN）
        df_valid = df.dropna()
        
        min_required = max(self.bb_period, self.rsi_window)
        if df_valid.empty:
            if verbose:
                print(f"❌ 数据不足，需要至少 {min_required} 条有效数据")
            return {"signal": "HOLD", "confidence_score": 0, 
                    "reason": f"Insufficient data for indicators (need {min_required})"}, 0.0
        
        # 4. 获取最新数据点
        latest = df_valid.iloc[-1]
        current_price = latest['close']
        
        # 5. 生成信号
        signal, confidence, reason = self._generate_signal_from_indicators(
            current_price, latest['BB_UPPER'], latest['BB_LOWER'], 
            latest['SMA'], latest['RSI']
        )
        
        # 6. 打印信息
        if verbose:
            timestamp_str = df_valid.index[-1].strftime('%Y-%m-%d %H:%M') if hasattr(df_valid.index[-1], 'strftime') else str(df_valid.index[-1])
            print(f"   [{timestamp_str}] 价格: ${current_price:.2f} | "
                  f"BB: [${latest['BB_LOWER']:.2f}, ${latest['BB_UPPER']:.2f}] | RSI: {latest['RSI']:.1f}")
            print(f"   🎯 信号: {signal} (置信度: {confidence}/10) - {reason}")
        
        return {
            "signal": signal,
            "confidence_score": confidence,
            "reason": reason
        }, current_price
    
    def __str__(self):
        return f"MeanReversionStrategy(BB={self.bb_period}, RSI={self.rsi_window})"


# ==================== 测试用例 ====================
if __name__ == '__main__':
    from datetime import timedelta
    
    def create_test_data(num_bars: int, base_price: float, start_time: datetime) -> pd.DataFrame:
        """创建测试用 OHLCV 数据"""
        np.random.seed(42)
        prices = base_price + np.cumsum(np.random.randn(num_bars) * 0.5)
        index = pd.DatetimeIndex([start_time + timedelta(minutes=i*5) for i in range(num_bars)])
        return pd.DataFrame({
            'open': prices - 0.1,
            'high': prices + 0.3,
            'low': prices - 0.3,
            'close': prices,
            'volume': np.random.randint(1000, 5000, num_bars)
        }, index=index)
    
    print("="*60)
    print("测试 MeanReversionStrategy (无 data_fetcher 依赖)")
    print("="*60)
    
    # 初始化策略
    strategy = MeanReversionStrategy(
        bb_period=20,
        rsi_window=14,
        max_history_bars=100
    )
    
    # 模拟多次数据到达
    base_time = datetime(2025, 12, 5, 9, 0, 0, tzinfo=timezone.utc)
    
    print("\n--- 第1批数据 (15条，不足以计算指标) ---")
    data_1 = create_test_data(15, 100.0, base_time)
    signal, price = strategy.get_signal("TSLA", data_1)
    print(f"历史累积: {strategy.get_history_size('TSLA')} 条")
    
    print("\n--- 第2批数据 (10条，累积后足够) ---")
    data_2 = create_test_data(10, 102.0, base_time + timedelta(minutes=75))
    signal, price = strategy.get_signal("TSLA", data_2)
    print(f"历史累积: {strategy.get_history_size('TSLA')} 条")
    
    print("\n--- 第3批数据 (5条，继续累积) ---")
    data_3 = create_test_data(5, 101.0, base_time + timedelta(minutes=125))
    signal, price = strategy.get_signal("TSLA", data_3)
    print(f"历史累积: {strategy.get_history_size('TSLA')} 条")
    
    print("\n--- 测试独立 ticker ---")
    aapl_data = create_test_data(30, 150.0, base_time)
    signal, price = strategy.get_signal("AAPL", aapl_data)
    print(f"TSLA 历史: {strategy.get_history_size('TSLA')} 条")
    print(f"AAPL 历史: {strategy.get_history_size('AAPL')} 条")
    
    print("\n--- 清除 TSLA 历史 ---")
    strategy.clear_history("TSLA")
    print(f"TSLA 历史: {strategy.get_history_size('TSLA')} 条")
    print(f"AAPL 历史: {strategy.get_history_size('AAPL')} 条")