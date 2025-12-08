# src/unit_tests/test_live_engine.py

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta, time as dt_time
import pandas as pd
import numpy as np

from src.engine.live_engine import LiveEngine, MARKET_OPEN_TIME, MARKET_CLOSE_TIME, US_EASTERN
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


def create_mock_ohlcv_data(num_bars: int = 30,
                           base_price: float = 100.0,
                           start_time: datetime = None) -> pd.DataFrame:
    """创建模拟 OHLCV 数据"""
    if start_time is None:
        start_time = datetime(2025, 12, 5, 9, 30, 0, tzinfo=timezone.utc)
    
    time_index = pd.DatetimeIndex([start_time + timedelta(minutes=i*5) for i in range(num_bars)])
    
    np.random.seed(42)
    prices = base_price + np.cumsum(np.random.randn(num_bars) * 0.5)
    
    return pd.DataFrame({
        'open': prices - 0.1,
        'high': prices + 0.3,
        'low': prices - 0.3,
        'close': prices,
        'volume': np.random.randint(1000, 5000, num_bars)
    }, index=time_index)


class TestLiveEngine(unittest.TestCase):
    
    def setUp(self):
        """设置测试环境"""
        self.mock_data_fetcher = MagicMock()
        self.mock_strategy = MagicMock()
        self.mock_strategy.__str__ = MagicMock(return_value="MockStrategy")
        self.mock_strategy.get_signal.return_value = (
            {"signal": "HOLD", "confidence_score": 5, "reason": "Test"},
            100.0
        )
        
        self.mock_position_manager = MagicMock()
        self.mock_position_manager.get_account_status.return_value = {
            'cash': 100000.0,
            'position': 0.0,
            'avg_cost': 0.0,
            'equity': 100000.0,
            'market_value': 0.0
        }
        self.mock_position_manager.get_trade_log.return_value = pd.DataFrame()
        self.mock_position_manager.execute_and_update.return_value = True
        
        self.mock_cache = MagicMock()
        self.mock_cache.data = {}
        
        self.ticker = "TEST"
        self.timeframe = TimeFrame(5, TimeFrameUnit.Minute)
        
        # 创建引擎（不遵守交易时间，方便测试）
        self.engine = LiveEngine(
            ticker=self.ticker,
            strategy=self.mock_strategy,
            position_manager=self.mock_position_manager,
            data_fetcher=self.mock_data_fetcher,
            cache=self.mock_cache,
            interval_seconds=60,
            lookback_minutes=120,
            timeframe=self.timeframe,
            respect_market_hours=False,
            max_runtime_minutes=None
        )
    
    # ==================== 初始化测试 ====================
    
    def test_initialization(self):
        """测试引擎初始化"""
        self.assertEqual(self.engine.ticker, self.ticker)
        self.assertEqual(self.engine.interval_seconds, 60)
        self.assertEqual(self.engine.lookback_minutes, 120)
        self.assertEqual(self.engine.timeframe, self.timeframe)
        self.assertFalse(self.engine.respect_market_hours)
        self.assertIsNone(self.engine.max_runtime_minutes)
        self.assertFalse(self.engine.is_running)
    
    def test_initialization_default_timeframe(self):
        """测试默认 timeframe"""
        engine = LiveEngine(
            ticker=self.ticker,
            strategy=self.mock_strategy,
            position_manager=self.mock_position_manager,
            data_fetcher=self.mock_data_fetcher
        )
        
        self.assertEqual(engine.timeframe.amount, 5)
        self.assertEqual(engine.timeframe.unit, TimeFrameUnit.Minute)
    
    # ==================== 市场时间测试 ====================
    
    @patch('src.live.live_engine.datetime')
    def test_is_market_open_during_trading_hours(self, mock_datetime):
        """测试交易时间内返回 True"""
        # 模拟周一 10:00 AM ET
        mock_now = US_EASTERN.localize(datetime(2025, 12, 8, 10, 0, 0))  # Monday
        mock_datetime.now.return_value = mock_now
        
        engine = LiveEngine(
            ticker=self.ticker,
            strategy=self.mock_strategy,
            position_manager=self.mock_position_manager,
            data_fetcher=self.mock_data_fetcher,
            respect_market_hours=True
        )
        
        # 直接测试内部方法
        with patch.object(engine, '_get_current_time_et', return_value=mock_now):
            self.assertTrue(engine._is_market_open())
    
    @patch('src.live.live_engine.datetime')
    def test_is_market_open_before_market_open(self, mock_datetime):
        """测试开盘前返回 False"""
        # 模拟周一 8:00 AM ET (开盘前)
        mock_now = US_EASTERN.localize(datetime(2025, 12, 8, 8, 0, 0))
        
        engine = LiveEngine(
            ticker=self.ticker,
            strategy=self.mock_strategy,
            position_manager=self.mock_position_manager,
            data_fetcher=self.mock_data_fetcher,
            respect_market_hours=True
        )
        
        with patch.object(engine, '_get_current_time_et', return_value=mock_now):
            self.assertFalse(engine._is_market_open())
    
    @patch('src.live.live_engine.datetime')
    def test_is_market_open_after_market_close(self, mock_datetime):
        """测试收盘后返回 False"""
        # 模拟周一 5:00 PM ET (收盘后)
        mock_now = US_EASTERN.localize(datetime(2025, 12, 8, 17, 0, 0))
        
        engine = LiveEngine(
            ticker=self.ticker,
            strategy=self.mock_strategy,
            position_manager=self.mock_position_manager,
            data_fetcher=self.mock_data_fetcher,
            respect_market_hours=True
        )
        
        with patch.object(engine, '_get_current_time_et', return_value=mock_now):
            self.assertFalse(engine._is_market_open())
    
    @patch('src.live.live_engine.datetime')
    def test_is_market_open_weekend(self, mock_datetime):
        """测试周末返回 False"""
        # 模拟周六 10:00 AM ET
        mock_now = US_EASTERN.localize(datetime(2025, 12, 6, 10, 0, 0))  # Saturday
        
        engine = LiveEngine(
            ticker=self.ticker,
            strategy=self.mock_strategy,
            position_manager=self.mock_position_manager,
            data_fetcher=self.mock_data_fetcher,
            respect_market_hours=True
        )
        
        with patch.object(engine, '_get_current_time_et', return_value=mock_now):
            self.assertFalse(engine._is_market_open())
    
    # ==================== _fetch_data 测试 ====================
    
    def test_fetch_data_success(self):
        """测试成功获取数据"""
        mock_data = create_mock_ohlcv_data(num_bars=20)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        
        df, price = self.engine._fetch_data()
        
        self.assertEqual(len(df), 20)
        self.assertAlmostEqual(price, mock_data.iloc[-1]['close'])
        self.mock_data_fetcher.get_latest_bars.assert_called_once()
    
    def test_fetch_data_empty(self):
        """测试数据为空时返回 0 价格"""
        self.mock_data_fetcher.get_latest_bars.return_value = pd.DataFrame()
        
        df, price = self.engine._fetch_data()
        
        self.assertTrue(df.empty)
        self.assertEqual(price, 0.0)
    
    # ==================== _run_single_iteration 测试 ====================
    
    def test_run_single_iteration_hold_signal(self):
        """测试单次迭代 - HOLD 信号"""
        mock_data = create_mock_ohlcv_data(num_bars=30)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        
        self.mock_strategy.get_signal.return_value = (
            {"signal": "HOLD", "confidence_score": 5, "reason": "Neutral"},
            100.0
        )
        
        result = self.engine._run_single_iteration()
        
        self.assertTrue(result)
        self.assertEqual(self.engine._iteration_count, 1)
        self.assertEqual(self.engine._signal_count, 0)
        self.mock_position_manager.execute_and_update.assert_not_called()
    
    def test_run_single_iteration_buy_signal(self):
        """测试单次迭代 - BUY 信号"""
        mock_data = create_mock_ohlcv_data(num_bars=30)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        
        self.mock_strategy.get_signal.return_value = (
            {"signal": "BUY", "confidence_score": 8, "reason": "Oversold"},
            100.0
        )
        
        result = self.engine._run_single_iteration()
        
        self.assertTrue(result)
        self.assertEqual(self.engine._signal_count, 1)
        self.mock_position_manager.execute_and_update.assert_called_once()
        
        call_args = self.mock_position_manager.execute_and_update.call_args
        self.assertEqual(call_args.kwargs['signal'], "BUY")
    
    def test_run_single_iteration_sell_signal(self):
        """测试单次迭代 - SELL 信号"""
        mock_data = create_mock_ohlcv_data(num_bars=30)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        
        self.mock_strategy.get_signal.return_value = (
            {"signal": "SELL", "confidence_score": 7, "reason": "Overbought"},
            100.0
        )
        
        result = self.engine._run_single_iteration()
        
        self.assertTrue(result)
        self.assertEqual(self.engine._signal_count, 1)
        self.mock_position_manager.execute_and_update.assert_called_once()
        
        call_args = self.mock_position_manager.execute_and_update.call_args
        self.assertEqual(call_args.kwargs['signal'], "SELL")
    
    def test_run_single_iteration_no_data(self):
        """测试单次迭代 - 无数据"""
        self.mock_data_fetcher.get_latest_bars.return_value = pd.DataFrame()
        
        result = self.engine._run_single_iteration()
        
        self.assertFalse(result)
        self.mock_strategy.get_signal.assert_not_called()
    
    def test_run_single_iteration_strategy_error(self):
        """测试单次迭代 - 策略错误"""
        mock_data = create_mock_ohlcv_data(num_bars=30)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        
        self.mock_strategy.get_signal.side_effect = Exception("Strategy error")
        
        result = self.engine._run_single_iteration()
        
        self.assertFalse(result)
        self.mock_position_manager.execute_and_update.assert_not_called()
    
    # ==================== 回调测试 ====================
    
    def test_signal_callback_called(self):
        """测试信号回调被调用"""
        mock_callback = MagicMock()
        
        engine = LiveEngine(
            ticker=self.ticker,
            strategy=self.mock_strategy,
            position_manager=self.mock_position_manager,
            data_fetcher=self.mock_data_fetcher,
            on_signal_callback=mock_callback,
            respect_market_hours=False
        )
        
        mock_data = create_mock_ohlcv_data(num_bars=30)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        
        self.mock_strategy.get_signal.return_value = (
            {"signal": "BUY", "confidence_score": 8, "reason": "Test"},
            100.0
        )
        
        engine._run_single_iteration()
        
        mock_callback.assert_called_once()
        call_args = mock_callback.call_args[0]
        self.assertEqual(call_args[0]['signal'], "BUY")
    
    def test_signal_callback_error_handled(self):
        """测试回调错误被处理"""
        mock_callback = MagicMock(side_effect=Exception("Callback error"))
        
        engine = LiveEngine(
            ticker=self.ticker,
            strategy=self.mock_strategy,
            position_manager=self.mock_position_manager,
            data_fetcher=self.mock_data_fetcher,
            on_signal_callback=mock_callback,
            respect_market_hours=False
        )
        
        mock_data = create_mock_ohlcv_data(num_bars=30)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        
        self.mock_strategy.get_signal.return_value = (
            {"signal": "BUY", "confidence_score": 8, "reason": "Test"},
            100.0
        )
        
        # 不应该崩溃
        result = engine._run_single_iteration()
        self.assertTrue(result)
    
    # ==================== stop() 测试 ====================
    
    def test_stop(self):
        """测试手动停止"""
        self.engine._running = True
        self.assertTrue(self.engine.is_running)
        
        self.engine.stop()
        
        self.assertFalse(self.engine.is_running)
    
    # ==================== _format_duration 测试 ====================
    
    def test_format_duration_seconds(self):
        """测试时长格式化 - 秒"""
        result = self.engine._format_duration(45)
        self.assertEqual(result, "45s")
    
    def test_format_duration_minutes(self):
        """测试时长格式化 - 分钟"""
        result = self.engine._format_duration(125)
        self.assertEqual(result, "2m 5s")
    
    def test_format_duration_hours(self):
        """测试时长格式化 - 小时"""
        result = self.engine._format_duration(3725)
        self.assertEqual(result, "1h 2m 5s")
    
    # ==================== _generate_report 测试 ====================
    
    def test_generate_report(self):
        """测试生成报告"""
        self.engine._start_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        self.engine._iteration_count = 10
        self.engine._signal_count = 3
        
        mock_data = create_mock_ohlcv_data(num_bars=5)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        
        self.mock_position_manager.get_account_status.return_value = {
            'cash': 95000.0,
            'position': 50.0,
            'avg_cost': 100.0,
            'equity': 100000.0,
            'market_value': 5000.0
        }
        
        report = self.engine._generate_report()
        
        self.assertEqual(report['ticker'], self.ticker)
        self.assertEqual(report['iterations'], 10)
        self.assertEqual(report['signals'], 3)
        self.assertGreater(report['runtime_seconds'], 0)
        self.assertEqual(report['final_equity'], 100000.0)


class TestLiveEngineIntegration(unittest.TestCase):
    """集成测试 - 测试 run() 方法"""
    
    def setUp(self):
        self.mock_data_fetcher = MagicMock()
        self.mock_strategy = MagicMock()
        self.mock_strategy.__str__ = MagicMock(return_value="MockStrategy")
        self.mock_position_manager = MagicMock()
        self.mock_position_manager.get_account_status.return_value = {
            'cash': 100000.0, 'position': 0.0, 'avg_cost': 0.0,
            'equity': 100000.0, 'market_value': 0.0
        }
        self.mock_position_manager.get_trade_log.return_value = pd.DataFrame()
        self.mock_cache = MagicMock()
        self.mock_cache.data = {}
    
    @patch('time.sleep')
    def test_run_with_max_runtime(self, mock_sleep):
        """测试带最大运行时间的 run()"""
        engine = LiveEngine(
            ticker="TEST",
            strategy=self.mock_strategy,
            position_manager=self.mock_position_manager,
            data_fetcher=self.mock_data_fetcher,
            cache=self.mock_cache,
            interval_seconds=1,
            respect_market_hours=False,
            max_runtime_minutes=0.01  # 非常短的运行时间
        )
        
        mock_data = create_mock_ohlcv_data(num_bars=30)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        
        self.mock_strategy.get_signal.return_value = (
            {"signal": "HOLD", "confidence_score": 5, "reason": "Test"},
            100.0
        )
        
        report = engine.run()
        
        self.assertIsNotNone(report)
        self.assertIn('final_equity', report)
        self.assertFalse(engine.is_running)


if __name__ == '__main__':
    unittest.main()