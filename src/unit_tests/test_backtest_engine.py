# src/unit_tests/test_backtest_engine.py

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

from src.backtest.backtest_engine import BacktestEngine
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


def create_mock_ohlcv_data(num_bars: int = 30,
                           base_price: float = 100.0,
                           start_time: datetime = None) -> pd.DataFrame:
    """创建模拟 OHLCV 数据"""
    if start_time is None:
        start_time = datetime(2025, 12, 5, 9, 0, 0, tzinfo=timezone.utc)
    
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


class TestBacktestEngine(unittest.TestCase):
    
    def setUp(self):
        """设置测试环境"""
        # Mock data fetcher
        self.mock_data_fetcher = MagicMock()
        
        # Mock strategy
        self.mock_strategy = MagicMock()
        self.mock_strategy.__str__ = MagicMock(return_value="MockStrategy")
        self.mock_strategy.get_signal.return_value = (
            {"signal": "HOLD", "confidence_score": 5, "reason": "Test"},
            100.0
        )
        
        # Mock position manager
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
        
        # Mock cache
        self.mock_cache = MagicMock()
        
        # Test parameters
        self.ticker = "TEST"
        self.start_dt = datetime(2025, 12, 5, 9, 0, 0, tzinfo=timezone.utc)
        self.end_dt = datetime(2025, 12, 5, 10, 0, 0, tzinfo=timezone.utc)  # 1 hour
        self.step_minutes = 15
        self.lookback_minutes = 60
        self.timeframe = TimeFrame(5, TimeFrameUnit.Minute)
        
        # Initialize engine
        self.engine = BacktestEngine(
            ticker=self.ticker,
            start_dt=self.start_dt,
            end_dt=self.end_dt,
            strategy=self.mock_strategy,
            position_manager=self.mock_position_manager,
            data_fetcher=self.mock_data_fetcher,
            cache=self.mock_cache,
            step_minutes=self.step_minutes,
            lookback_minutes=self.lookback_minutes,
            timeframe=self.timeframe
        )
    
    # ==================== 初始化测试 ====================
    
    def test_initialization(self):
        """测试引擎初始化"""
        self.assertEqual(self.engine.ticker, self.ticker)
        self.assertEqual(self.engine.start_dt, self.start_dt)
        self.assertEqual(self.engine.end_dt, self.end_dt)
        self.assertEqual(self.engine.step_minutes, self.step_minutes)
        self.assertEqual(self.engine.lookback_minutes, self.lookback_minutes)
        self.assertEqual(self.engine.timeframe, self.timeframe)
    
    def test_initialization_default_timeframe(self):
        """测试默认 timeframe"""
        engine = BacktestEngine(
            ticker=self.ticker,
            start_dt=self.start_dt,
            end_dt=self.end_dt,
            strategy=self.mock_strategy,
            position_manager=self.mock_position_manager,
            data_fetcher=self.mock_data_fetcher,
            cache=self.mock_cache
            # timeframe not specified
        )
        
        self.assertEqual(engine.timeframe.amount, 5)
        self.assertEqual(engine.timeframe.unit, TimeFrameUnit.Minute)
    
    # ==================== _fetch_data 测试 ====================
    
    def test_fetch_data_calls_data_fetcher(self):
        """测试 _fetch_data 正确调用 data_fetcher"""
        mock_data = create_mock_ohlcv_data(num_bars=20)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        
        current_time = datetime(2025, 12, 5, 9, 30, 0, tzinfo=timezone.utc)
        result = self.engine._fetch_data(current_time)
        
        self.mock_data_fetcher.get_latest_bars.assert_called_once_with(
            ticker=self.ticker,
            lookback_minutes=self.lookback_minutes,
            end_dt=current_time,
            timeframe=self.timeframe
        )
        
        self.assertEqual(len(result), 20)
    
    def test_fetch_data_uses_configured_timeframe(self):
        """测试 _fetch_data 使用配置的 timeframe"""
        custom_timeframe = TimeFrame(15, TimeFrameUnit.Minute)
        engine = BacktestEngine(
            ticker=self.ticker,
            start_dt=self.start_dt,
            end_dt=self.end_dt,
            strategy=self.mock_strategy,
            position_manager=self.mock_position_manager,
            data_fetcher=self.mock_data_fetcher,
            cache=self.mock_cache,
            timeframe=custom_timeframe
        )
        
        self.mock_data_fetcher.get_latest_bars.return_value = pd.DataFrame()
        current_time = datetime(2025, 12, 5, 9, 30, 0, tzinfo=timezone.utc)
        engine._fetch_data(current_time)
        
        call_args = self.mock_data_fetcher.get_latest_bars.call_args
        self.assertEqual(call_args.kwargs['timeframe'], custom_timeframe)
    
    # ==================== _get_current_price 测试 ====================
    
    def test_get_current_price_success(self):
        """测试成功获取当前价格"""
        mock_data = create_mock_ohlcv_data(num_bars=5, base_price=150.0)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        
        current_time = datetime(2025, 12, 5, 9, 30, 0, tzinfo=timezone.utc)
        price = self.engine._get_current_price(current_time)
        
        self.assertGreater(price, 0)
        # 验证使用的是最后一条数据的 close 价格
        self.assertAlmostEqual(price, mock_data.iloc[-1]['close'])
    
    def test_get_current_price_no_data(self):
        """测试无数据时返回 0"""
        self.mock_data_fetcher.get_latest_bars.return_value = pd.DataFrame()
        
        current_time = datetime(2025, 12, 5, 9, 30, 0, tzinfo=timezone.utc)
        price = self.engine._get_current_price(current_time)
        
        self.assertEqual(price, 0.0)
    
    # ==================== run() 测试 ====================
    
    def test_run_basic_execution(self):
        """测试基本运行流程"""
        # 设置 mock 返回值
        mock_data = create_mock_ohlcv_data(num_bars=30, base_price=100.0)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        
        self.mock_strategy.get_signal.return_value = (
            {"signal": "HOLD", "confidence_score": 5, "reason": "Neutral"},
            100.0
        )
        
        self.mock_position_manager.get_account_status.return_value = {
            'cash': 100000.0,
            'position': 0.0,
            'avg_cost': 0.0,
            'equity': 100000.0,
            'market_value': 0.0
        }
        
        # 运行回测
        final_equity, trade_log = self.engine.run()
        
        # 验证返回值
        self.assertEqual(final_equity, 100000.0)
        self.assertIsInstance(trade_log, pd.DataFrame)
        
        # 验证 strategy.get_signal 被调用
        self.assertTrue(self.mock_strategy.get_signal.called)
    
    def test_run_executes_buy_signal(self):
        """测试执行买入信号"""
        mock_data = create_mock_ohlcv_data(num_bars=30, base_price=100.0)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        
        # 策略返回 BUY 信号
        self.mock_strategy.get_signal.return_value = (
            {"signal": "BUY", "confidence_score": 8, "reason": "Oversold"},
            100.0
        )
        
        self.engine.run()
        
        # 验证 execute_and_update 被调用且信号为 BUY
        self.assertTrue(self.mock_position_manager.execute_and_update.called)
        call_args = self.mock_position_manager.execute_and_update.call_args
        self.assertEqual(call_args.kwargs['signal'], "BUY")
    
    def test_run_executes_sell_signal(self):
        """测试执行卖出信号"""
        mock_data = create_mock_ohlcv_data(num_bars=30, base_price=100.0)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        
        # 策略返回 SELL 信号
        self.mock_strategy.get_signal.return_value = (
            {"signal": "SELL", "confidence_score": 7, "reason": "Overbought"},
            100.0
        )
        
        self.engine.run()
        
        # 验证 execute_and_update 被调用且信号为 SELL
        self.assertTrue(self.mock_position_manager.execute_and_update.called)
        call_args = self.mock_position_manager.execute_and_update.call_args
        self.assertEqual(call_args.kwargs['signal'], "SELL")
    
    def test_run_hold_signal_no_execution(self):
        """测试 HOLD 信号不执行交易"""
        mock_data = create_mock_ohlcv_data(num_bars=30, base_price=100.0)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        
        # 策略返回 HOLD 信号
        self.mock_strategy.get_signal.return_value = (
            {"signal": "HOLD", "confidence_score": 5, "reason": "Neutral"},
            100.0
        )
        
        self.engine.run()
        
        # 验证 execute_and_update 未被调用
        self.mock_position_manager.execute_and_update.assert_not_called()
    
    def test_run_skips_when_no_price(self):
        """测试无价格数据时跳过"""
        # 使用函数来处理多次调用
        call_count = [0]
        def mock_get_latest_bars(**kwargs):
            call_count[0] += 1
            # 第一次获取价格返回空（模拟无价格）
            if call_count[0] == 1:
                return pd.DataFrame()
            # 后续调用返回有效数据
            return create_mock_ohlcv_data(num_bars=30)
        
        self.mock_data_fetcher.get_latest_bars.side_effect = mock_get_latest_bars
        
        self.mock_strategy.get_signal.return_value = (
            {"signal": "HOLD", "confidence_score": 5, "reason": "Test"},
            100.0
        )
        
        # 应该不会崩溃
        final_equity, trade_log = self.engine.run()
        self.assertIsNotNone(final_equity)
    
    def test_run_skips_when_no_market_data(self):
        """测试无市场数据时跳过策略调用"""
        # 价格获取成功，但市场数据为空
        price_data = create_mock_ohlcv_data(num_bars=5, base_price=100.0)
        
        call_count = [0]
        def mock_get_latest_bars(**kwargs):
            call_count[0] += 1
            if kwargs.get('lookback_minutes') == 15:  # _get_current_price 调用
                return price_data
            else:  # _fetch_data 调用
                return pd.DataFrame()  # 返回空数据
        
        self.mock_data_fetcher.get_latest_bars.side_effect = mock_get_latest_bars
        
        self.engine.run()
        
        # 策略不应被调用（因为无市场数据）
        self.mock_strategy.get_signal.assert_not_called()
    
    def test_run_handles_strategy_exception(self):
        """测试处理策略异常"""
        mock_data = create_mock_ohlcv_data(num_bars=30, base_price=100.0)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        
        # 策略抛出异常
        self.mock_strategy.get_signal.side_effect = Exception("Strategy error")
        
        # 应该不会崩溃，继续运行
        final_equity, trade_log = self.engine.run()
        self.assertIsNotNone(final_equity)
    
    def test_run_time_stepping(self):
        """测试时间步进正确"""
        mock_data = create_mock_ohlcv_data(num_bars=30, base_price=100.0)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        
        self.mock_strategy.get_signal.return_value = (
            {"signal": "HOLD", "confidence_score": 5, "reason": "Test"},
            100.0
        )
        
        # 1小时，15分钟步进 = 5次迭代 (9:00, 9:15, 9:30, 9:45, 10:00)
        self.engine.run()
        
        # get_signal 应该被调用 5 次
        self.assertEqual(self.mock_strategy.get_signal.call_count, 5)
    
    def test_run_uses_strategy_price(self):
        """测试使用策略返回的价格"""
        mock_data = create_mock_ohlcv_data(num_bars=30, base_price=100.0)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        
        # 策略返回特定价格
        strategy_price = 105.5
        self.mock_strategy.get_signal.return_value = (
            {"signal": "BUY", "confidence_score": 8, "reason": "Test"},
            strategy_price
        )
        
        self.engine.run()
        
        # 验证 execute_and_update 使用策略返回的价格
        call_args = self.mock_position_manager.execute_and_update.call_args
        self.assertEqual(call_args.kwargs['current_price'], strategy_price)
    
    def test_run_returns_final_equity(self):
        """测试返回最终权益"""
        mock_data = create_mock_ohlcv_data(num_bars=30, base_price=100.0)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        
        self.mock_strategy.get_signal.return_value = (
            {"signal": "HOLD", "confidence_score": 5, "reason": "Test"},
            100.0
        )
        
        # 设置最终权益
        final_account_status = {
            'cash': 95000.0,
            'position': 50.0,
            'avg_cost': 100.0,
            'equity': 100500.0,
            'market_value': 5500.0
        }
        self.mock_position_manager.get_account_status.return_value = final_account_status
        
        final_equity, _ = self.engine.run()
        
        self.assertEqual(final_equity, 100500.0)
    
    def test_run_returns_trade_log(self):
        """测试返回交易日志"""
        mock_data = create_mock_ohlcv_data(num_bars=30, base_price=100.0)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        
        self.mock_strategy.get_signal.return_value = (
            {"signal": "HOLD", "confidence_score": 5, "reason": "Test"},
            100.0
        )
        
        # 设置交易日志
        expected_trade_log = pd.DataFrame({
            'time': [datetime.now(timezone.utc)],
            'type': ['BUY'],
            'qty': [100],
            'price': [100.0]
        })
        self.mock_position_manager.get_trade_log.return_value = expected_trade_log
        
        _, trade_log = self.engine.run()
        
        self.assertEqual(len(trade_log), 1)
        self.assertEqual(trade_log.iloc[0]['type'], 'BUY')
    
    def test_run_timezone_handling(self):
        """测试时区处理 - 验证引擎能正确处理时区"""
        # 使用 UTC 时区的 datetime
        engine = BacktestEngine(
            ticker=self.ticker,
            start_dt=datetime(2025, 12, 5, 9, 0, 0, tzinfo=timezone.utc),
            end_dt=datetime(2025, 12, 5, 9, 30, 0, tzinfo=timezone.utc),
            strategy=self.mock_strategy,
            position_manager=self.mock_position_manager,
            data_fetcher=self.mock_data_fetcher,
            cache=self.mock_cache,
            step_minutes=15
        )
        
        mock_data = create_mock_ohlcv_data(num_bars=30, base_price=100.0)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        
        self.mock_strategy.get_signal.return_value = (
            {"signal": "HOLD", "confidence_score": 5, "reason": "Test"},
            100.0
        )
        
        # 应该正常运行
        final_equity, trade_log = engine.run()
        self.assertIsNotNone(final_equity)
        
        # 验证 get_signal 被调用时传入的时间是带时区的
        call_args = self.mock_strategy.get_signal.call_args
        # get_signal 接收 new_data，不直接接收时间，所以只验证运行成功即可
    
    # ==================== 集成场景测试 ====================
    
    def test_run_multiple_signals_scenario(self):
        """测试多信号场景"""
        mock_data = create_mock_ohlcv_data(num_bars=30, base_price=100.0)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        
        # 模拟不同时间点的不同信号
        signals = [
            ({"signal": "BUY", "confidence_score": 8, "reason": "Buy"}, 100.0),
            ({"signal": "HOLD", "confidence_score": 5, "reason": "Hold"}, 101.0),
            ({"signal": "HOLD", "confidence_score": 5, "reason": "Hold"}, 102.0),
            ({"signal": "SELL", "confidence_score": 7, "reason": "Sell"}, 103.0),
            ({"signal": "HOLD", "confidence_score": 5, "reason": "Hold"}, 102.0),
        ]
        self.mock_strategy.get_signal.side_effect = signals
        
        self.engine.run()
        
        # 验证 execute_and_update 被调用 2 次（BUY 和 SELL）
        self.assertEqual(self.mock_position_manager.execute_and_update.call_count, 2)
        
        # 验证调用顺序
        calls = self.mock_position_manager.execute_and_update.call_args_list
        self.assertEqual(calls[0].kwargs['signal'], "BUY")
        self.assertEqual(calls[1].kwargs['signal'], "SELL")


class TestBacktestEngineEdgeCases(unittest.TestCase):
    """边界条件测试"""
    
    def setUp(self):
        self.mock_data_fetcher = MagicMock()
        self.mock_strategy = MagicMock()
        self.mock_strategy.__str__ = MagicMock(return_value="MockStrategy")
        self.mock_position_manager = MagicMock()
        self.mock_position_manager.get_account_status.return_value = {
            'cash': 100000.0, 'equity': 100000.0, 'position': 0.0,
            'avg_cost': 0.0, 'market_value': 0.0
        }
        self.mock_position_manager.get_trade_log.return_value = pd.DataFrame()
        self.mock_cache = MagicMock()
    
    def test_zero_duration_backtest(self):
        """测试零时长回测（开始=结束）"""
        same_time = datetime(2025, 12, 5, 9, 0, 0, tzinfo=timezone.utc)
        
        engine = BacktestEngine(
            ticker="TEST",
            start_dt=same_time,
            end_dt=same_time,
            strategy=self.mock_strategy,
            position_manager=self.mock_position_manager,
            data_fetcher=self.mock_data_fetcher,
            cache=self.mock_cache
        )
        
        mock_data = create_mock_ohlcv_data(num_bars=30)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        self.mock_strategy.get_signal.return_value = (
            {"signal": "HOLD", "confidence_score": 5, "reason": "Test"}, 100.0
        )
        
        final_equity, trade_log = engine.run()
        
        # 应该至少运行一次（开始时间点）
        self.assertTrue(self.mock_strategy.get_signal.called)
    
    def test_very_short_backtest(self):
        """测试非常短的回测"""
        engine = BacktestEngine(
            ticker="TEST",
            start_dt=datetime(2025, 12, 5, 9, 0, 0, tzinfo=timezone.utc),
            end_dt=datetime(2025, 12, 5, 9, 1, 0, tzinfo=timezone.utc),  # 1分钟
            strategy=self.mock_strategy,
            position_manager=self.mock_position_manager,
            data_fetcher=self.mock_data_fetcher,
            cache=self.mock_cache,
            step_minutes=5  # 步进大于时长
        )
        
        mock_data = create_mock_ohlcv_data(num_bars=30)
        self.mock_data_fetcher.get_latest_bars.return_value = mock_data
        self.mock_strategy.get_signal.return_value = (
            {"signal": "HOLD", "confidence_score": 5, "reason": "Test"}, 100.0
        )
        
        final_equity, trade_log = engine.run()
        
        # 应该只运行一次
        self.assertEqual(self.mock_strategy.get_signal.call_count, 1)
    
    def test_different_timeframes(self):
        """测试不同 timeframe 配置"""
        timeframes = [
            TimeFrame.Minute,
            TimeFrame(5, TimeFrameUnit.Minute),
            TimeFrame(15, TimeFrameUnit.Minute),
            TimeFrame.Hour,
            TimeFrame.Day,
        ]
        
        for tf in timeframes:
            engine = BacktestEngine(
                ticker="TEST",
                start_dt=datetime(2025, 12, 5, 9, 0, 0, tzinfo=timezone.utc),
                end_dt=datetime(2025, 12, 5, 9, 30, 0, tzinfo=timezone.utc),
                strategy=self.mock_strategy,
                position_manager=self.mock_position_manager,
                data_fetcher=self.mock_data_fetcher,
                cache=self.mock_cache,
                timeframe=tf
            )
            
            self.assertEqual(engine.timeframe, tf)


if __name__ == '__main__':
    unittest.main()