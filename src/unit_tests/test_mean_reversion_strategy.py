# src/unit_tests/test_mean_reversion_strategy.py

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

from src.strategies.mean_reversion_strategy import MeanReversionStrategy
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


def create_mock_ohlcv_data(num_bars: int = 50, 
                           base_price: float = 100.0,
                           trend: str = 'neutral') -> pd.DataFrame:
    """
    创建模拟的 OHLCV 数据用于测试。
    
    Args:
        num_bars: K线数量
        base_price: 基准价格
        trend: 趋势类型 ('up', 'down', 'neutral', 'volatile')
    """
    base_time = datetime(2025, 12, 5, 9, 0, 0, tzinfo=timezone.utc)
    time_index = pd.DatetimeIndex([base_time + timedelta(minutes=i) for i in range(num_bars)])
    
    if trend == 'up':
        # 上升趋势
        prices = base_price + np.cumsum(np.random.randn(num_bars) * 0.5 + 0.3)
    elif trend == 'down':
        # 下降趋势
        prices = base_price + np.cumsum(np.random.randn(num_bars) * 0.5 - 0.3)
    elif trend == 'volatile':
        # 高波动
        prices = base_price + np.cumsum(np.random.randn(num_bars) * 2.0)
    else:
        # 中性趋势
        prices = base_price + np.cumsum(np.random.randn(num_bars) * 0.2)
    
    df = pd.DataFrame({
        'open': prices - 0.1,
        'high': prices + 0.3,
        'low': prices - 0.3,
        'close': prices,
        'volume': np.random.randint(1000, 5000, num_bars)
    }, index=time_index)
    
    return df


class TestMeanReversionStrategy(unittest.TestCase):
    
    def setUp(self):
        """为每个测试方法设置环境"""
        # 创建 mock data fetcher
        self.mock_fetcher = MagicMock()
        
        # 初始化策略（使用默认参数）
        self.strategy = MeanReversionStrategy(
            data_fetcher=self.mock_fetcher,
            bb_period=20,
            bb_std_dev=2,
            rsi_window=14,
            rsi_oversold=30,
            rsi_overbought=70
        )
        
        self.ticker = "TEST"

    def test_initialization(self):
        """测试策略初始化"""
        self.assertEqual(self.strategy.bb_period, 20)
        self.assertEqual(self.strategy.bb_std_dev, 2)
        self.assertEqual(self.strategy.rsi_window, 14)
        self.assertEqual(self.strategy.rsi_oversold, 30)
        self.assertEqual(self.strategy.rsi_overbought, 70)

    def test_calculate_bollinger_bands(self):
        """测试布林带计算"""
        df = create_mock_ohlcv_data(num_bars=50)
        
        df_with_bb = self.strategy._calculate_bollinger_bands(df)
        
        # 验证新列存在
        self.assertIn('SMA', df_with_bb.columns)
        self.assertIn('BB_UPPER', df_with_bb.columns)
        self.assertIn('BB_LOWER', df_with_bb.columns)
        
        # 验证前 19 行为 NaN（因为周期是 20）
        self.assertTrue(df_with_bb['SMA'].iloc[:19].isna().all())
        
        # 验证最后一行有有效值
        self.assertFalse(pd.isna(df_with_bb['SMA'].iloc[-1]))
        
        # 验证上下轨关系
        last_row = df_with_bb.iloc[-1]
        self.assertGreater(last_row['BB_UPPER'], last_row['SMA'])
        self.assertLess(last_row['BB_LOWER'], last_row['SMA'])

    def test_calculate_rsi(self):
        """测试 RSI 计算"""
        df = create_mock_ohlcv_data(num_bars=50)
        
        df_with_rsi = self.strategy._calculate_rsi(df)
        
        # 验证 RSI 列存在
        self.assertIn('RSI', df_with_rsi.columns)
        
        # 验证前 13 行为 NaN（因为窗口是 14）
        self.assertTrue(df_with_rsi['RSI'].iloc[:13].isna().all())
        
        # 验证 RSI 值在 0-100 之间
        valid_rsi = df_with_rsi['RSI'].dropna()
        self.assertTrue((valid_rsi >= 0).all() and (valid_rsi <= 100).all())

    def test_generate_buy_signal_strong(self):
        """测试强买入信号生成（价格低于下轨且 RSI 超卖）"""
        signal, confidence, reason = self.strategy._generate_signal_from_indicators(
            latest_close=90.0,
            bb_upper=105.0,
            bb_lower=95.0,
            sma=100.0,
            rsi=25.0  # 超卖
        )
        
        self.assertEqual(signal, "BUY")
        self.assertEqual(confidence, 9)  # 高置信度
        self.assertIn("跌破布林带下轨", reason)
        self.assertIn("超卖", reason)

    def test_generate_buy_signal_weak(self):
        """测试弱买入信号生成（仅价格低于下轨）"""
        signal, confidence, reason = self.strategy._generate_signal_from_indicators(
            latest_close=94.0,
            bb_upper=105.0,
            bb_lower=95.0,
            sma=100.0,
            rsi=40.0  # 未超卖
        )
        
        self.assertEqual(signal, "BUY")
        self.assertEqual(confidence, 6)  # 中等置信度
        self.assertIn("跌破布林带下轨", reason)

    def test_generate_sell_signal_strong(self):
        """测试强卖出信号生成（价格高于上轨且 RSI 超买）"""
        signal, confidence, reason = self.strategy._generate_signal_from_indicators(
            latest_close=106.0,
            bb_upper=105.0,
            bb_lower=95.0,
            sma=100.0,
            rsi=75.0  # 超买
        )
        
        self.assertEqual(signal, "SELL")
        self.assertEqual(confidence, 8)
        self.assertIn("突破布林带上轨", reason)
        self.assertIn("超买", reason)

    def test_generate_sell_signal_rsi_only(self):
        """测试仅 RSI 超买的卖出信号"""
        signal, confidence, reason = self.strategy._generate_signal_from_indicators(
            latest_close=102.0,
            bb_upper=105.0,
            bb_lower=95.0,
            sma=100.0,
            rsi=75.0  # 超买
        )
        
        self.assertEqual(signal, "SELL")
        self.assertEqual(confidence, 7)
        self.assertIn("超买", reason)

    def test_generate_hold_signal(self):
        """测试 HOLD 信号生成"""
        signal, confidence, reason = self.strategy._generate_signal_from_indicators(
            latest_close=100.0,
            bb_upper=105.0,
            bb_lower=95.0,
            sma=100.0,
            rsi=50.0
        )
        
        self.assertEqual(signal, "HOLD")
        self.assertEqual(confidence, 5)
        self.assertIn("布林带区间内", reason)

    def test_generate_signal_with_nan_data(self):
        """测试当指标数据为 NaN 时返回 HOLD"""
        signal, confidence, reason = self.strategy._generate_signal_from_indicators(
            latest_close=100.0,
            bb_upper=np.nan,
            bb_lower=95.0,
            sma=100.0,
            rsi=50.0
        )
        
        self.assertEqual(signal, "HOLD")
        self.assertEqual(confidence, 0)
        self.assertIn("技术指标数据不足", reason)

    def test_get_signal_success(self):
        """测试完整的信号生成流程"""
        # 创建模拟数据（确保有足够的数据计算指标）
        mock_data = create_mock_ohlcv_data(num_bars=50, base_price=100.0)
        
        # 配置 mock fetcher 返回数据
        self.mock_fetcher.get_latest_bars.return_value = mock_data
        
        # 调用 get_signal
        signal_dict, price = self.strategy.get_signal(
            ticker=self.ticker,
            lookback_minutes=60
        )
        
        # 验证 mock 被调用
        self.mock_fetcher.get_latest_bars.assert_called_once()
        
        # 验证返回结果结构
        self.assertIn('signal', signal_dict)
        self.assertIn('confidence_score', signal_dict)
        self.assertIn('reason', signal_dict)
        self.assertIn(signal_dict['signal'], ['BUY', 'SELL', 'HOLD'])
        self.assertGreater(price, 0)

    def test_get_signal_no_data(self):
        """测试当无法获取数据时的处理"""
        # 配置 mock fetcher 返回空 DataFrame
        self.mock_fetcher.get_latest_bars.return_value = pd.DataFrame()
        
        signal_dict, price = self.strategy.get_signal(
            ticker=self.ticker,
            lookback_minutes=60
        )
        
        # 验证返回 HOLD 信号和价格为 0
        self.assertEqual(signal_dict['signal'], "HOLD")
        self.assertEqual(signal_dict['confidence_score'], 0)
        self.assertEqual(price, 0.0)

    def test_get_signal_insufficient_data_for_indicators(self):
        """测试数据不足以计算指标的情况"""
        # 创建只有 10 个 bar 的数据（不足以计算 BB 和 RSI）
        insufficient_data = create_mock_ohlcv_data(num_bars=10)
        
        self.mock_fetcher.get_latest_bars.return_value = insufficient_data
        
        signal_dict, price = self.strategy.get_signal(
            ticker=self.ticker,
            lookback_minutes=60
        )
        
        # 验证返回 HOLD 信号
        self.assertEqual(signal_dict['signal'], "HOLD")
        self.assertEqual(signal_dict['confidence_score'], 0)
        self.assertIn("Insufficient data", signal_dict['reason'])

    def test_custom_parameters(self):
        """测试使用自定义参数初始化策略"""
        custom_strategy = MeanReversionStrategy(
            data_fetcher=self.mock_fetcher,
            bb_period=10,
            bb_std_dev=1.5,
            rsi_window=10,
            rsi_oversold=25,
            rsi_overbought=75
        )
        
        self.assertEqual(custom_strategy.bb_period, 10)
        self.assertEqual(custom_strategy.bb_std_dev, 1.5)
        self.assertEqual(custom_strategy.rsi_window, 10)
        self.assertEqual(custom_strategy.rsi_oversold, 25)
        self.assertEqual(custom_strategy.rsi_overbought, 75)

    def test_signal_generation_with_extreme_volatility(self):
        """测试在极端波动情况下的信号生成"""
        volatile_data = create_mock_ohlcv_data(num_bars=50, trend='volatile')
        
        self.mock_fetcher.get_latest_bars.return_value = volatile_data
        
        signal_dict, price = self.strategy.get_signal(
            ticker=self.ticker,
            lookback_minutes=60
        )
        
        # 验证能够处理极端波动数据
        self.assertIn(signal_dict['signal'], ['BUY', 'SELL', 'HOLD'])
        self.assertIsInstance(signal_dict['confidence_score'], int)


if __name__ == '__main__':
    unittest.main()