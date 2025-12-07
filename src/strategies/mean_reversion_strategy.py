# src/strategies/mean_reversion_strategy.py

import pandas as pd
import numpy as np
from typing import Literal, Tuple, Dict, Optional
from datetime import datetime, timezone
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# 导入数据获取器
from src.data_fetcher.alpaca_data_fetcher import AlpacaDataFetcher


class MeanReversionStrategy:
    """
    均值回归策略类 - 使用纯数学计算，基于布林带和 RSI 指标。
    
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
    
    def __init__(self, 
                 data_fetcher: AlpacaDataFetcher,
                 bb_period: int = DEFAULT_BB_PERIOD,
                 bb_std_dev: float = DEFAULT_BB_STD_DEV,
                 rsi_window: int = DEFAULT_RSI_WINDOW,
                 rsi_oversold: float = DEFAULT_RSI_OVERSOLD,
                 rsi_overbought: float = DEFAULT_RSI_OVERBOUGHT):
        """
        初始化均值回归策略。
        
        Args:
            data_fetcher: AlpacaDataFetcher 实例
            bb_period: 布林带计算周期
            bb_std_dev: 布林带标准差倍数
            rsi_window: RSI 计算窗口
            rsi_oversold: RSI 超卖阈值
            rsi_overbought: RSI 超买阈值
        """
        self.data_fetcher = data_fetcher
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
        print(f"📊 MeanReversionStrategy 初始化成功。")
        print(f"   参数: BB({bb_period}, {bb_std_dev}σ), RSI({rsi_window}), "
              f"超卖<{rsi_oversold}, 超买>{rsi_overbought}")
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算布林带指标。
        
        Args:
            df: 包含 'close' 列的 DataFrame
            
        Returns:
            pd.DataFrame: 添加了 SMA, BB_UPPER, BB_LOWER 列的 DataFrame
        """
        df = df.copy()
        df['SMA'] = df['close'].rolling(window=self.bb_period).mean()
        df['STD'] = df['close'].rolling(window=self.bb_period).std()
        df['BB_UPPER'] = df['SMA'] + (df['STD'] * self.bb_std_dev)
        df['BB_LOWER'] = df['SMA'] - (df['STD'] * self.bb_std_dev)
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 RSI (相对强弱指数) 指标。
        
        Args:
            df: 包含 'close' 列的 DataFrame
            
        Returns:
            pd.DataFrame: 添加了 RSI 列的 DataFrame
        """
        df = df.copy()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        
        # 避免除以零
        RS = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + RS))
        return df
    
    def _generate_signal_from_indicators(self, 
                                        latest_close: float,
                                        bb_upper: float,
                                        bb_lower: float,
                                        sma: float,
                                        rsi: float) -> Tuple[Literal["BUY", "SELL", "HOLD"], int, str]:
        """
        根据技术指标生成交易信号。
        
        Args:
            latest_close: 最新收盘价
            bb_upper: 布林带上轨
            bb_lower: 布林带下轨
            sma: 简单移动平均线
            rsi: RSI 指标值
            
        Returns:
            Tuple[signal, confidence, reason]:
                - signal: "BUY", "SELL", 或 "HOLD"
                - confidence: 信号置信度 (1-10)
                - reason: 信号原因说明
        """
        # 检查是否有无效数据
        if pd.isna([latest_close, bb_upper, bb_lower, sma, rsi]).any():
            return "HOLD", 0, "技术指标数据不足，无法计算信号"
        
        # BUY 信号：价格跌破下轨 AND RSI 超卖
        if latest_close < bb_lower and rsi < self.rsi_oversold:
            confidence = 9  # 双重确认，高置信度
            reason = (f"价格 ${latest_close:.2f} 跌破布林带下轨 ${bb_lower:.2f}，"
                     f"且 RSI={rsi:.1f} < {self.rsi_oversold} (超卖)")
            return "BUY", confidence, reason
        
        # BUY 信号 (弱)：仅价格跌破下轨
        elif latest_close < bb_lower:
            confidence = 6
            reason = f"价格 ${latest_close:.2f} 跌破布林带下轨 ${bb_lower:.2f}"
            return "BUY", confidence, reason
        
        # SELL 信号：价格突破上轨 OR RSI 超买
        elif latest_close > bb_upper or rsi > self.rsi_overbought:
            confidence = 8 if (latest_close > bb_upper and rsi > self.rsi_overbought) else 7
            
            if latest_close > bb_upper and rsi > self.rsi_overbought:
                reason = (f"价格 ${latest_close:.2f} 突破布林带上轨 ${bb_upper:.2f}，"
                         f"且 RSI={rsi:.1f} > {self.rsi_overbought} (超买)")
            elif latest_close > bb_upper:
                reason = f"价格 ${latest_close:.2f} 突破布林带上轨 ${bb_upper:.2f}"
            else:
                reason = f"RSI={rsi:.1f} > {self.rsi_overbought} (超买)"
            
            return "SELL", confidence, reason
        
        # HOLD 信号：价格在正常区间内
        else:
            confidence = 5
            reason = (f"价格 ${latest_close:.2f} 在布林带区间内 "
                     f"[${bb_lower:.2f}, ${bb_upper:.2f}]，RSI={rsi:.1f}")
            return "HOLD", confidence, reason
    
    def get_signal(self, 
                   ticker: str,
                   end_dt: Optional[datetime] = None,
                   lookback_minutes: int = 60,
                   timeframe: TimeFrame = TimeFrame(5, TimeFrameUnit.Minute)) -> Tuple[Dict, float]:
        """
        获取指定时间点的交易信号。
        
        Args:
            ticker: 股票代码
            end_dt: 结束时间（默认为当前时间）
            lookback_minutes: 回溯时间长度（分钟）
            timeframe: K线时间框架
            
        Returns:
            Tuple[signal_dict, current_price]:
                - signal_dict: 包含 signal, confidence_score, reason 的字典
                - current_price: 当前价格
        """
        # 1. 获取原始 K 线数据
        df = self.data_fetcher.get_latest_bars(
            ticker=ticker,
            lookback_minutes=lookback_minutes,
            timeframe=timeframe,
            end_dt=end_dt
        )
        
        if df.empty:
            print(f"❌ 无法获取 {ticker} 的数据。")
            return {"signal": "HOLD", "confidence_score": 0, "reason": "No data"}, 0.0
        
        # 2. 计算技术指标
        df = self._calculate_bollinger_bands(df)
        df = self._calculate_rsi(df)
        
        # 删除 NaN 行
        df = df.dropna()
        
        if df.empty:
            print(f"❌ 计算技术指标后数据不足。")
            return {"signal": "HOLD", "confidence_score": 0, "reason": "Insufficient data for indicators"}, 0.0
        
        # 3. 获取最新数据
        latest_row = df.iloc[-1]
        current_price = latest_row['close']
        bb_upper = latest_row['BB_UPPER']
        bb_lower = latest_row['BB_LOWER']
        sma = latest_row['SMA']
        rsi = latest_row['RSI']
        
        # 4. 生成信号
        signal, confidence, reason = self._generate_signal_from_indicators(
            current_price, bb_upper, bb_lower, sma, rsi
        )
        
        # 5. 打印信号信息
        timestamp_str = df.index[-1].strftime('%Y-%m-%d %H:%M UTC') if hasattr(df.index[-1], 'strftime') else str(df.index[-1])
        print(f"\n📊 [{timestamp_str}] {ticker} 技术分析:")
        print(f"   价格: ${current_price:.2f}")
        print(f"   布林带: [${bb_lower:.2f}, ${sma:.2f}, ${bb_upper:.2f}]")
        print(f"   RSI: {rsi:.1f}")
        print(f"   🎯 信号: {signal} (置信度: {confidence}/10)")
        print(f"   💡 原因: {reason}")
        
        signal_dict = {
            "signal": signal,
            "confidence_score": confidence,
            "reason": reason
        }
        
        return signal_dict, current_price


# 测试用例
if __name__ == '__main__':
    from datetime import datetime, timezone
    
    # 初始化数据获取器和策略
    fetcher = AlpacaDataFetcher()
    strategy = MeanReversionStrategy(
        data_fetcher=fetcher,
        bb_period=20,
        bb_std_dev=2,
        rsi_window=14,
        rsi_oversold=30,
        rsi_overbought=70
    )
    
    # 测试获取信号
    print("\n" + "="*60)
    print("测试 MeanReversionStrategy - 纯数学计算")
    print("="*60)
    
    signal_dict, price = strategy.get_signal(
        ticker="TSLA",
        lookback_minutes=120,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute)
    )
    
    print(f"\n最终输出:")
    print(f"  信号: {signal_dict['signal']}")
    print(f"  置信度: {signal_dict['confidence_score']}/10")
    print(f"  原因: {signal_dict['reason']}")
    print(f"  当前价格: ${price:.2f}")