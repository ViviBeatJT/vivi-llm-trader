# src/unit_tests/test_mean_reversion_strategy.py

import unittest
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

from src.strategies.mean_reversion_strategy import MeanReversionStrategy


def create_mock_ohlcv_data(num_bars: int = 50, 
                           base_price: float = 100.0,
                           trend: str = 'neutral',
                           start_time: datetime = None) -> pd.DataFrame:
    """创建模拟 OHLCV 数据"""
    if start_time is None:
        start_time = datetime(2025, 12, 5, 9, 0, 0, tzinfo=timezone.utc)
    
    time_index = pd.DatetimeIndex([start_time + timedelta(minutes=i*5) for i in range(num_bars)])
    
    np.random.seed(42)  # 可重复性
    if trend == 'up':
        prices = base_price + np.cumsum(np.random.randn(num_bars) * 0.5 + 0.3)
    elif trend == 'down':
        prices = base_price + np.cumsum(np.random.randn(num_bars) * 0.5 - 0.3)
    elif trend == 'volatile':
        prices = base_price + np.cumsum(np.random.randn(num_bars) * 2.0)
    else:
        prices = base_price + np.cumsum(np.random.randn(num_bars) * 0.2)
    
    return pd.DataFrame({
        'open': prices - 0.1,
        'high': prices + 0.3,
        'low': prices - 0.3,
        'close': prices,
        'volume': np.random.randint(1000, 5000, num_bars)
    }, index=time_index)


class TestMeanReversionStrategy(unittest.TestCase):
    
    def setUp(self):
        """初始化策略"""
        self.strategy = MeanReversionStrategy(
            bb_period=20,
            bb_std_dev=2,
            rsi_window=14,
            rsi_oversold=30,
            rsi_overbought=70,
            max_history_bars=100,
            enable_short=True
        )
        self.ticker = "TEST"

    # ==================== 初始化测试 ====================
    
    def test_initialization(self):
        """测试策略初始化参数"""
        self.assertEqual(self.strategy.bb_period, 20)
        self.assertEqual(self.strategy.bb_std_dev, 2)
        self.assertEqual(self.strategy.rsi_window, 14)
        self.assertEqual(self.strategy.rsi_oversold, 30)
        self.assertEqual(self.strategy.rsi_overbought, 70)
        self.assertEqual(self.strategy.max_history_bars, 100)
        self.assertTrue(self.strategy.enable_short)
        self.assertEqual(self.strategy._history_data, {})
    
    def test_initialization_custom_params(self):
        """测试自定义参数"""
        strategy = MeanReversionStrategy(
            bb_period=10,
            bb_std_dev=1.5,
            rsi_window=10,
            rsi_oversold=25,
            rsi_overbought=75,
            max_history_bars=200,
            enable_short=False
        )
        self.assertEqual(strategy.bb_period, 10)
        self.assertEqual(strategy.rsi_oversold, 25)
        self.assertEqual(strategy.max_history_bars, 200)
        self.assertFalse(strategy.enable_short)

    # ==================== 技术指标计算测试 ====================
    
    def test_calculate_bollinger_bands(self):
        """测试布林带计算"""
        df = create_mock_ohlcv_data(num_bars=50)
        df_bb = self.strategy._calculate_bollinger_bands(df)
        
        self.assertIn('SMA', df_bb.columns)
        self.assertIn('BB_UPPER', df_bb.columns)
        self.assertIn('BB_LOWER', df_bb.columns)
        
        # 前 19 行应为 NaN
        self.assertTrue(df_bb['SMA'].iloc[:19].isna().all())
        self.assertFalse(pd.isna(df_bb['SMA'].iloc[-1]))
        
        # 上轨 > SMA > 下轨
        last = df_bb.iloc[-1]
        self.assertGreater(last['BB_UPPER'], last['SMA'])
        self.assertLess(last['BB_LOWER'], last['SMA'])

    def test_calculate_rsi(self):
        """测试 RSI 计算"""
        df = create_mock_ohlcv_data(num_bars=50)
        df_rsi = self.strategy._calculate_rsi(df)
        
        self.assertIn('RSI', df_rsi.columns)
        
        valid_rsi = df_rsi['RSI'].dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())

    # ==================== 信号生成测试 ====================
    
    def test_generate_buy_signal_strong(self):
        """测试强买入信号（超卖）"""
        signal, conf, reason = self.strategy._generate_signal_from_indicators(
            latest_close=90.0, bb_upper=105.0, bb_lower=95.0, sma=100.0, rsi=25.0
        )
        self.assertEqual(signal, "BUY")
        self.assertEqual(conf, 9)
        self.assertIn("超卖", reason)

    def test_generate_buy_signal_weak(self):
        """测试弱买入信号（仅价格低于下轨）"""
        signal, conf, reason = self.strategy._generate_signal_from_indicators(
            latest_close=94.0, bb_upper=105.0, bb_lower=95.0, sma=100.0, rsi=40.0
        )
        self.assertEqual(signal, "BUY")
        self.assertEqual(conf, 6)

    def test_generate_buy_signal_rsi_only(self):
        """测试仅 RSI 超卖"""
        signal, conf, reason = self.strategy._generate_signal_from_indicators(
            latest_close=96.0, bb_upper=105.0, bb_lower=95.0, sma=100.0, rsi=25.0
        )
        self.assertEqual(signal, "BUY")
        self.assertEqual(conf, 5)

    def test_generate_short_signal_strong(self):
        """测试强做空信号（超买，启用做空）"""
        signal, conf, reason = self.strategy._generate_signal_from_indicators(
            latest_close=106.0, bb_upper=105.0, bb_lower=95.0, sma=100.0, rsi=75.0
        )
        self.assertEqual(signal, "SHORT")
        self.assertEqual(conf, 9)

    def test_generate_short_signal_weak_price_only(self):
        """测试弱做空信号（仅价格高于上轨）"""
        signal, conf, reason = self.strategy._generate_signal_from_indicators(
            latest_close=106.0, bb_upper=105.0, bb_lower=95.0, sma=100.0, rsi=60.0
        )
        self.assertEqual(signal, "SHORT")
        self.assertEqual(conf, 6)

    def test_generate_short_signal_weak_rsi_only(self):
        """测试弱做空信号（仅 RSI 超买）"""
        signal, conf, reason = self.strategy._generate_signal_from_indicators(
            latest_close=103.0, bb_upper=105.0, bb_lower=95.0, sma=100.0, rsi=75.0
        )
        self.assertEqual(signal, "SHORT")
        self.assertEqual(conf, 5)

    def test_generate_sell_signal_when_short_disabled(self):
        """测试禁用做空时产生 SELL 信号"""
        strategy_no_short = MeanReversionStrategy(enable_short=False)
        
        signal, conf, reason = strategy_no_short._generate_signal_from_indicators(
            latest_close=106.0, bb_upper=105.0, bb_lower=95.0, sma=100.0, rsi=75.0
        )
        self.assertEqual(signal, "SELL")
        self.assertEqual(conf, 9)

    def test_generate_hold_signal(self):
        """测试持有信号"""
        signal, conf, reason = self.strategy._generate_signal_from_indicators(
            latest_close=100.0, bb_upper=105.0, bb_lower=95.0, sma=100.0, rsi=50.0
        )
        self.assertEqual(signal, "HOLD")
        self.assertEqual(conf, 5)

    def test_generate_signal_with_nan(self):
        """测试 NaN 数据处理"""
        signal, conf, reason = self.strategy._generate_signal_from_indicators(
            latest_close=100.0, bb_upper=np.nan, bb_lower=95.0, sma=100.0, rsi=50.0
        )
        self.assertEqual(signal, "HOLD")
        self.assertEqual(conf, 0)

    # ==================== 信号验证测试 ====================
    
    def test_validate_signal_valid(self):
        """测试有效信号验证"""
        self.assertEqual(self.strategy._validate_signal('BUY'), 'BUY')
        self.assertEqual(self.strategy._validate_signal('SELL'), 'SELL')
        self.assertEqual(self.strategy._validate_signal('SHORT'), 'SHORT')
        self.assertEqual(self.strategy._validate_signal('COVER'), 'COVER')
        self.assertEqual(self.strategy._validate_signal('HOLD'), 'HOLD')

    def test_validate_signal_lowercase(self):
        """测试小写信号验证"""
        self.assertEqual(self.strategy._validate_signal('buy'), 'BUY')
        self.assertEqual(self.strategy._validate_signal('short'), 'SHORT')

    def test_validate_signal_invalid(self):
        """测试无效信号验证"""
        self.assertEqual(self.strategy._validate_signal('INVALID'), 'HOLD')
        self.assertEqual(self.strategy._validate_signal(''), 'HOLD')

    # ==================== 历史数据管理测试 ====================
    
    def test_merge_data_empty_history(self):
        """测试空历史时的合并"""
        new_df = create_mock_ohlcv_data(num_bars=10)
        merged = self.strategy._merge_data(self.ticker, new_df)
        
        self.assertEqual(len(merged), 10)
        self.assertEqual(self.strategy.get_history_size(self.ticker), 10)

    def test_merge_data_accumulation(self):
        """测试数据累积"""
        t1 = datetime(2025, 12, 5, 9, 0, 0, tzinfo=timezone.utc)
        df1 = create_mock_ohlcv_data(num_bars=20, start_time=t1)
        self.strategy._merge_data(self.ticker, df1)
        
        t2 = datetime(2025, 12, 5, 10, 40, 0, tzinfo=timezone.utc)
        df2 = create_mock_ohlcv_data(num_bars=15, start_time=t2)
        merged = self.strategy._merge_data(self.ticker, df2)
        
        self.assertEqual(len(merged), 35)

    def test_merge_data_deduplication(self):
        """测试去重（保留最新）"""
        t1 = datetime(2025, 12, 5, 9, 0, 0, tzinfo=timezone.utc)
        df1 = create_mock_ohlcv_data(num_bars=20, start_time=t1)
        self.strategy._merge_data(self.ticker, df1)
        
        # 重叠数据
        t2 = datetime(2025, 12, 5, 9, 50, 0, tzinfo=timezone.utc)
        df2 = create_mock_ohlcv_data(num_bars=15, base_price=200.0, start_time=t2)
        merged = self.strategy._merge_data(self.ticker, df2)
        
        # 10 (旧的前10条) + 15 (新的) = 25
        self.assertEqual(len(merged), 25)
        
        # 重叠位置应使用新数据
        self.assertGreater(merged.loc[t2]['close'], 150)

    def test_merge_data_max_limit(self):
        """测试历史数据大小限制"""
        self.strategy.max_history_bars = 50
        large_df = create_mock_ohlcv_data(num_bars=100)
        merged = self.strategy._merge_data(self.ticker, large_df)
        
        self.assertEqual(len(merged), 50)

    def test_get_history_data_copy(self):
        """测试获取的是副本"""
        df = create_mock_ohlcv_data(num_bars=20)
        self.strategy._history_data[self.ticker] = df
        
        retrieved = self.strategy.get_history_data(self.ticker)
        retrieved.iloc[0] = 0  # 修改副本
        
        # 原数据不受影响
        self.assertNotEqual(self.strategy._history_data[self.ticker].iloc[0]['close'], 0)

    def test_clear_history_single(self):
        """测试清除单个 ticker"""
        self.strategy._history_data["TSLA"] = create_mock_ohlcv_data(20)
        self.strategy._history_data["AAPL"] = create_mock_ohlcv_data(15)
        
        self.strategy.clear_history("TSLA")
        
        self.assertEqual(self.strategy.get_history_size("TSLA"), 0)
        self.assertEqual(self.strategy.get_history_size("AAPL"), 15)

    def test_clear_history_all(self):
        """测试清除所有"""
        self.strategy._history_data["TSLA"] = create_mock_ohlcv_data(20)
        self.strategy._history_data["AAPL"] = create_mock_ohlcv_data(15)
        
        self.strategy.clear_history()
        
        self.assertEqual(len(self.strategy._history_data), 0)

    # ==================== get_signal 完整流程测试 ====================
    
    def test_get_signal_success(self):
        """测试完整信号生成"""
        data = create_mock_ohlcv_data(num_bars=50)
        signal_dict, price = self.strategy.get_signal(self.ticker, data, verbose=False)
        
        self.assertIn('signal', signal_dict)
        self.assertIn('confidence_score', signal_dict)
        self.assertIn('reason', signal_dict)
        self.assertIn(signal_dict['signal'], ['BUY', 'SELL', 'SHORT', 'COVER', 'HOLD'])
        self.assertGreater(price, 0)
        self.assertEqual(self.strategy.get_history_size(self.ticker), 50)

    def test_get_signal_accumulates_history(self):
        """测试多次调用累积历史"""
        t1 = datetime(2025, 12, 5, 9, 0, 0, tzinfo=timezone.utc)
        data_1 = create_mock_ohlcv_data(15, start_time=t1)
        self.strategy.get_signal(self.ticker, data_1, verbose=False)
        self.assertEqual(self.strategy.get_history_size(self.ticker), 15)
        
        t2 = datetime(2025, 12, 5, 10, 15, 0, tzinfo=timezone.utc)
        data_2 = create_mock_ohlcv_data(15, start_time=t2)
        self.strategy.get_signal(self.ticker, data_2, verbose=False)
        self.assertEqual(self.strategy.get_history_size(self.ticker), 30)

    def test_get_signal_insufficient_data(self):
        """测试数据不足"""
        data = create_mock_ohlcv_data(num_bars=10)
        signal_dict, price = self.strategy.get_signal(self.ticker, data, verbose=False)
        
        self.assertEqual(signal_dict['signal'], "HOLD")
        self.assertEqual(signal_dict['confidence_score'], 0)
        self.assertIn("Insufficient", signal_dict['reason'])

    def test_get_signal_empty_data(self):
        """测试空数据"""
        signal_dict, price = self.strategy.get_signal(self.ticker, pd.DataFrame(), verbose=False)
        
        self.assertEqual(signal_dict['signal'], "HOLD")
        self.assertEqual(price, 0.0)

    def test_get_signal_uses_accumulated_history(self):
        """测试累积后数据足够"""
        # 第一次：不足
        t1 = datetime(2025, 12, 5, 9, 0, 0, tzinfo=timezone.utc)
        data_1 = create_mock_ohlcv_data(10, start_time=t1)
        sig1, _ = self.strategy.get_signal(self.ticker, data_1, verbose=False)
        self.assertEqual(sig1['signal'], "HOLD")
        
        # 第二次：累积后足够
        t2 = datetime(2025, 12, 5, 9, 50, 0, tzinfo=timezone.utc)
        data_2 = create_mock_ohlcv_data(15, start_time=t2)
        sig2, price2 = self.strategy.get_signal(self.ticker, data_2, verbose=False)
        
        self.assertEqual(self.strategy.get_history_size(self.ticker), 25)
        self.assertIn(sig2['signal'], ['BUY', 'SELL', 'SHORT', 'COVER', 'HOLD'])
        self.assertGreater(price2, 0)

    def test_multiple_tickers_independent(self):
        """测试多 ticker 独立"""
        tsla_data = create_mock_ohlcv_data(30, base_price=200.0)
        aapl_data = create_mock_ohlcv_data(25, base_price=150.0)
        
        self.strategy.get_signal("TSLA", tsla_data, verbose=False)
        self.strategy.get_signal("AAPL", aapl_data, verbose=False)
        
        self.assertEqual(self.strategy.get_history_size("TSLA"), 30)
        self.assertEqual(self.strategy.get_history_size("AAPL"), 25)
        
        self.strategy.clear_history("TSLA")
        self.assertEqual(self.strategy.get_history_size("TSLA"), 0)
        self.assertEqual(self.strategy.get_history_size("AAPL"), 25)

    def test_str_representation(self):
        """测试字符串表示"""
        s = str(self.strategy)
        self.assertIn("MeanReversion", s)
        self.assertIn("BB=20", s)
        self.assertIn("SHORT", s)  # 因为 enable_short=True

    def test_str_representation_no_short(self):
        """测试禁用做空时的字符串表示"""
        strategy_no_short = MeanReversionStrategy(enable_short=False)
        s = str(strategy_no_short)
        self.assertNotIn("SHORT", s)


if __name__ == '__main__':
    unittest.main()