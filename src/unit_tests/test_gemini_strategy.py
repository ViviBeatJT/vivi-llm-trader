# src/unit_tests/test_gemini_strategy.py

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import json

from src.strategies.gemini_strategy import GeminiStrategy, TradingSignal


def create_mock_ohlcv_data(num_bars: int = 50, 
                           base_price: float = 100.0,
                           start_time: datetime = None) -> pd.DataFrame:
    """创建模拟的 OHLCV 数据"""
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


class TestGeminiStrategy(unittest.TestCase):
    
    def setUp(self):
        """设置测试环境"""
        # Mock cache
        self.mock_cache = MagicMock()
        self.mock_cache.get.return_value = None
        
        # Mock Gemini client
        self.mock_client = MagicMock()
        
        # Patch genai.Client
        self.client_patcher = patch('src.strategies.gemini_strategy.genai.Client')
        self.MockClient = self.client_patcher.start()
        self.MockClient.return_value = self.mock_client
        
        # 初始化策略
        self.strategy = GeminiStrategy(
            cache=self.mock_cache,
            use_cache=True,
            temperature=0.2,
            delay_seconds=0,
            bb_period=20,
            rsi_window=14,
            max_history_bars=100
        )
        self.strategy.client = self.mock_client
        
        self.ticker = "TEST"
    
    def tearDown(self):
        self.client_patcher.stop()
    
    # ==================== 初始化测试 ====================
    
    def test_initialization(self):
        """测试策略初始化"""
        self.assertIsNotNone(self.strategy.client)
        self.assertEqual(self.strategy.temperature, 0.2)
        self.assertTrue(self.strategy.use_cache)
        self.assertEqual(self.strategy.delay_seconds, 0)
        self.assertEqual(self.strategy.bb_period, 20)
        self.assertEqual(self.strategy.rsi_window, 14)
        self.assertEqual(self.strategy.max_history_bars, 100)
        self.assertEqual(self.strategy._history_data, {})
    
    def test_initialization_without_cache(self):
        """测试不使用缓存的初始化"""
        strategy = GeminiStrategy(cache=None, use_cache=False)
        self.assertFalse(strategy.use_cache)
    
    def test_initialization_custom_prompt(self):
        """测试自定义系统提示词"""
        custom_prompt = "You are a conservative trader."
        strategy = GeminiStrategy(
            cache=self.mock_cache,
            system_prompt=custom_prompt
        )
        self.assertEqual(strategy.system_prompt, custom_prompt)
    
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
        retrieved.iloc[0] = 0
        
        self.assertNotEqual(self.strategy._history_data[self.ticker].iloc[0]['close'], 0)
    
    def test_get_history_data_nonexistent(self):
        """测试获取不存在的 ticker"""
        retrieved = self.strategy.get_history_data("NONEXISTENT")
        self.assertTrue(retrieved.empty)
    
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
    
    # ==================== 技术指标计算测试 ====================
    
    def test_calculate_technical_indicators(self):
        """测试技术指标计算"""
        df = create_mock_ohlcv_data(num_bars=50)
        df_with_indicators = self.strategy._calculate_technical_indicators(df)
        
        self.assertIn('SMA', df_with_indicators.columns)
        self.assertIn('BB_UPPER', df_with_indicators.columns)
        self.assertIn('BB_LOWER', df_with_indicators.columns)
        self.assertIn('RSI', df_with_indicators.columns)
        self.assertNotIn('STD', df_with_indicators.columns)
        
        valid_data = df_with_indicators.dropna()
        self.assertGreater(len(valid_data), 0)
    
    # ==================== LLM 格式化测试 ====================
    
    def test_format_data_for_llm(self):
        """测试 LLM 数据格式化"""
        df = create_mock_ohlcv_data(num_bars=50)
        df = self.strategy._calculate_technical_indicators(df)
        df = df.dropna()
        
        formatted_text = self.strategy._format_data_for_llm(df, "TSLA")
        
        self.assertIn("TSLA", formatted_text)
        self.assertIn("Close", formatted_text)
        self.assertIn("RSI", formatted_text)
        self.assertIsInstance(formatted_text, str)
    
    def test_format_data_for_llm_empty(self):
        """测试空 DataFrame 的格式化"""
        formatted_text = self.strategy._format_data_for_llm(pd.DataFrame(), "TSLA")
        self.assertEqual(formatted_text, "没有可用的市场数据。")
    
    def test_generate_cache_key(self):
        """测试缓存键生成"""
        ticker = "TSLA"
        timestamp = datetime(2025, 12, 5, 10, 0, 0, tzinfo=timezone.utc)
        data = "some market data"
        
        key1 = self.strategy._generate_cache_key(ticker, timestamp, data)
        key2 = self.strategy._generate_cache_key(ticker, timestamp, data)
        
        self.assertEqual(key1, key2)
        self.assertEqual(len(key1), 64)
        
        key3 = self.strategy._generate_cache_key(ticker, timestamp, "different data")
        self.assertNotEqual(key1, key3)
    
    # ==================== API 调用测试 ====================
    
    @patch('time.sleep')
    def test_call_gemini_api_success(self, mock_sleep):
        """测试成功调用 Gemini API"""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "signal": "BUY",
            "confidence_score": 8,
            "reason": "Strong bullish indicators"
        })
        self.mock_client.models.generate_content.return_value = mock_response
        
        result = self.strategy._call_gemini_api("Test prompt")
        
        self.assertEqual(result['signal'], "BUY")
        self.assertEqual(result['confidence_score'], 8)
        self.mock_client.models.generate_content.assert_called_once()
    
    def test_call_gemini_api_failure(self):
        """测试 API 调用失败"""
        self.mock_client.models.generate_content.side_effect = Exception("API Error")
        
        result = self.strategy._call_gemini_api("Test prompt")
        
        self.assertEqual(result['signal'], "HOLD")
        self.assertEqual(result['confidence_score'], 0)
        self.assertIn("API Error", result['reason'])
    
    def test_call_gemini_api_empty_response(self):
        """测试 API 返回空响应"""
        mock_response = MagicMock()
        mock_response.text = ""
        self.mock_client.models.generate_content.return_value = mock_response
        
        result = self.strategy._call_gemini_api("Test prompt")
        
        self.assertEqual(result['signal'], "HOLD")
        self.assertEqual(result['confidence_score'], 0)
    
    def test_call_gemini_api_no_client(self):
        """测试客户端未初始化"""
        self.strategy.client = None
        
        result = self.strategy._call_gemini_api("Test prompt")
        
        self.assertEqual(result['signal'], "HOLD")
        self.assertIn("not initialized", result['reason'])
    
    # ==================== get_signal 完整流程测试 ====================
    
    @patch('time.sleep')
    def test_get_signal_success_cache_miss(self, mock_sleep):
        """测试成功获取信号（缓存未命中）"""
        mock_data = create_mock_ohlcv_data(num_bars=50, base_price=100.0)
        self.mock_cache.get.return_value = None
        
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "signal": "BUY",
            "confidence_score": 7,
            "reason": "Price below BB lower band"
        })
        self.mock_client.models.generate_content.return_value = mock_response
        
        signal_dict, price = self.strategy.get_signal(
            ticker=self.ticker,
            new_data=mock_data,
            verbose=False
        )
        
        self.assertEqual(signal_dict['signal'], "BUY")
        self.assertEqual(signal_dict['confidence_score'], 7)
        self.assertGreater(price, 0)
        
        self.mock_cache.get.assert_called_once()
        self.mock_cache.add.assert_called_once()
    
    def test_get_signal_cache_hit(self):
        """测试成功获取信号（缓存命中）"""
        mock_data = create_mock_ohlcv_data(num_bars=50)
        
        cached_signal = {
            "signal": "SELL",
            "confidence_score": 9,
            "reason": "Cached result"
        }
        self.mock_cache.get.return_value = cached_signal
        
        signal_dict, price = self.strategy.get_signal(
            ticker=self.ticker,
            new_data=mock_data,
            verbose=False
        )
        
        self.assertEqual(signal_dict, cached_signal)
        self.mock_client.models.generate_content.assert_not_called()
        self.mock_cache.add.assert_not_called()
    
    def test_get_signal_no_data(self):
        """测试无数据时的处理"""
        signal_dict, price = self.strategy.get_signal(
            ticker=self.ticker,
            new_data=pd.DataFrame(),
            verbose=False
        )
        
        self.assertEqual(signal_dict['signal'], "HOLD")
        self.assertEqual(signal_dict['confidence_score'], 0)
        self.assertEqual(price, 0.0)
        self.mock_client.models.generate_content.assert_not_called()
    
    def test_get_signal_insufficient_data(self):
        """测试数据不足"""
        insufficient_data = create_mock_ohlcv_data(num_bars=10)
        
        signal_dict, price = self.strategy.get_signal(
            ticker=self.ticker,
            new_data=insufficient_data,
            verbose=False
        )
        
        self.assertEqual(signal_dict['signal'], "HOLD")
        self.assertEqual(signal_dict['confidence_score'], 0)
        self.assertIn("Insufficient", signal_dict['reason'])
    
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
    
    @patch('time.sleep')
    def test_get_signal_uses_accumulated_history(self, mock_sleep):
        """测试累积后数据足够"""
        # 第一次：不足
        t1 = datetime(2025, 12, 5, 9, 0, 0, tzinfo=timezone.utc)
        data_1 = create_mock_ohlcv_data(10, start_time=t1)
        sig1, _ = self.strategy.get_signal(self.ticker, data_1, verbose=False)
        self.assertEqual(sig1['signal'], "HOLD")
        
        # 设置 API mock
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "signal": "BUY",
            "confidence_score": 7,
            "reason": "Test"
        })
        self.mock_client.models.generate_content.return_value = mock_response
        
        # 第二次：累积后足够
        t2 = datetime(2025, 12, 5, 9, 50, 0, tzinfo=timezone.utc)
        data_2 = create_mock_ohlcv_data(15, start_time=t2)
        sig2, price2 = self.strategy.get_signal(self.ticker, data_2, verbose=False)
        
        self.assertEqual(self.strategy.get_history_size(self.ticker), 25)
        self.assertEqual(sig2['signal'], "BUY")
        self.assertGreater(price2, 0)
    
    @patch('time.sleep')
    def test_get_signal_without_cache(self, mock_sleep):
        """测试不使用缓存"""
        strategy = GeminiStrategy(cache=None, use_cache=False, delay_seconds=0)
        strategy.client = self.mock_client
        
        mock_data = create_mock_ohlcv_data(num_bars=50)
        
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "signal": "HOLD",
            "confidence_score": 5,
            "reason": "Neutral market"
        })
        self.mock_client.models.generate_content.return_value = mock_response
        
        signal_dict, price = strategy.get_signal(
            ticker=self.ticker,
            new_data=mock_data,
            verbose=False
        )
        
        self.assertEqual(signal_dict['signal'], "HOLD")
        self.mock_client.models.generate_content.assert_called_once()
    
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
        self.assertIn("GeminiStrategy", s)
        self.assertIn("cache", s)


if __name__ == '__main__':
    unittest.main()