# src/unit_tests/test_gemini_strategy.py

import unittest
from unittest.mock import MagicMock, patch, Mock
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import json

from src.strategies.gemini_strategy import GeminiStrategy, TradingSignal
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


def create_mock_ohlcv_data(num_bars: int = 50, base_price: float = 100.0) -> pd.DataFrame:
    """创建模拟的 OHLCV 数据"""
    base_time = datetime(2025, 12, 5, 9, 0, 0, tzinfo=timezone.utc)
    time_index = pd.DatetimeIndex([base_time + timedelta(minutes=i*5) for i in range(num_bars)])
    
    prices = base_price + np.cumsum(np.random.randn(num_bars) * 0.5)
    
    df = pd.DataFrame({
        'open': prices - 0.1,
        'high': prices + 0.3,
        'low': prices - 0.3,
        'close': prices,
        'volume': np.random.randint(1000, 5000, num_bars)
    }, index=time_index)
    
    return df


class TestGeminiStrategy(unittest.TestCase):
    
    def setUp(self):
        """为每个测试方法设置环境"""
        # Mock data fetcher
        self.mock_fetcher = MagicMock()
        
        # Mock cache
        self.mock_cache = MagicMock()
        self.mock_cache.get.return_value = None  # 默认缓存未命中
        
        # Mock Gemini client
        self.mock_client = MagicMock()
        
        # Patch genai.Client at the module level
        self.client_patcher = patch('src.strategies.gemini_strategy.genai.Client')
        self.MockClient = self.client_patcher.start()
        self.MockClient.return_value = self.mock_client
        
        # 初始化策略
        self.strategy = GeminiStrategy(
            data_fetcher=self.mock_fetcher,
            cache=self.mock_cache,
            use_cache=True,
            temperature=0.2,
            delay_seconds=0  # 测试时不延迟
        )
        
        # 替换 client 为 mock
        self.strategy.client = self.mock_client
        
        self.ticker = "TEST"
    
    def tearDown(self):
        """清理 patches"""
        self.client_patcher.stop()
    
    def test_initialization_success(self):
        """测试成功初始化策略"""
        self.assertIsNotNone(self.strategy.client)
        self.assertEqual(self.strategy.temperature, 0.2)
        self.assertTrue(self.strategy.use_cache)
        self.assertEqual(self.strategy.delay_seconds, 0)
    
    def test_initialization_without_cache(self):
        """测试不使用缓存的初始化"""
        strategy_no_cache = GeminiStrategy(
            data_fetcher=self.mock_fetcher,
            cache=None,
            use_cache=False
        )
        
        self.assertFalse(strategy_no_cache.use_cache)
    
    def test_calculate_technical_indicators(self):
        """测试技术指标计算"""
        df = create_mock_ohlcv_data(num_bars=50)
        
        df_with_indicators = self.strategy._calculate_technical_indicators(df)
        
        # 验证新列存在
        self.assertIn('SMA', df_with_indicators.columns)
        self.assertIn('BB_UPPER', df_with_indicators.columns)
        self.assertIn('BB_LOWER', df_with_indicators.columns)
        self.assertIn('RSI', df_with_indicators.columns)
        
        # 验证 STD 列被删除
        self.assertNotIn('STD', df_with_indicators.columns)
        
        # 验证计算结果有效
        valid_data = df_with_indicators.dropna()
        self.assertGreater(len(valid_data), 0)
    
    def test_format_data_for_llm(self):
        """测试 LLM 数据格式化"""
        df = create_mock_ohlcv_data(num_bars=50)
        df = self.strategy._calculate_technical_indicators(df)
        df = df.dropna()
        
        formatted_text = self.strategy._format_data_for_llm(df, "TSLA")
        
        # 验证输出包含关键信息
        self.assertIn("TSLA", formatted_text)
        self.assertIn("Close", formatted_text)
        self.assertIn("RSI", formatted_text)
        self.assertIsInstance(formatted_text, str)
    
    def test_format_data_for_llm_empty_dataframe(self):
        """测试空 DataFrame 的格式化"""
        empty_df = pd.DataFrame()
        
        formatted_text = self.strategy._format_data_for_llm(empty_df, "TSLA")
        
        self.assertEqual(formatted_text, "没有可用的市场数据。")
    
    def test_generate_cache_key(self):
        """测试缓存键生成"""
        ticker = "TSLA"
        timestamp = datetime(2025, 12, 5, 10, 0, 0, tzinfo=timezone.utc)
        data = "some market data"
        
        key1 = self.strategy._generate_cache_key(ticker, timestamp, data)
        key2 = self.strategy._generate_cache_key(ticker, timestamp, data)
        
        # 相同输入应产生相同的键
        self.assertEqual(key1, key2)
        self.assertEqual(len(key1), 64)  # SHA256 哈希长度
        
        # 不同输入应产生不同的键
        key3 = self.strategy._generate_cache_key(ticker, timestamp, "different data")
        self.assertNotEqual(key1, key3)
    
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_call_gemini_api_success(self, mock_sleep):
        """测试成功调用 Gemini API"""
        # Mock API response
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "signal": "BUY",
            "confidence_score": 8,
            "reason": "Strong bullish indicators"
        })
        
        self.mock_client.models.generate_content.return_value = mock_response
        
        result = self.strategy._call_gemini_api("Test prompt")
        
        # 验证返回结果
        self.assertEqual(result['signal'], "BUY")
        self.assertEqual(result['confidence_score'], 8)
        self.assertIn("bullish", result['reason'])
        
        # 验证 API 被调用
        self.mock_client.models.generate_content.assert_called_once()
    
    def test_call_gemini_api_failure(self):
        """测试 API 调用失败的处理"""
        # Mock API 抛出异常
        self.mock_client.models.generate_content.side_effect = Exception("API Error")
        
        result = self.strategy._call_gemini_api("Test prompt")
        
        # 验证返回默认 HOLD 信号
        self.assertEqual(result['signal'], "HOLD")
        self.assertEqual(result['confidence_score'], 0)
        self.assertIn("API Error", result['reason'])
    
    def test_call_gemini_api_empty_response(self):
        """测试 API 返回空响应的处理"""
        mock_response = MagicMock()
        mock_response.text = ""
        
        self.mock_client.models.generate_content.return_value = mock_response
        
        result = self.strategy._call_gemini_api("Test prompt")
        
        self.assertEqual(result['signal'], "HOLD")
        self.assertEqual(result['confidence_score'], 0)
    
    @patch('time.sleep')
    def test_get_signal_success_with_cache_miss(self, mock_sleep):
        """测试成功获取信号（缓存未命中）"""
        # 1. Mock data fetcher 返回数据
        mock_data = create_mock_ohlcv_data(num_bars=50, base_price=100.0)
        self.mock_fetcher.get_latest_bars.return_value = mock_data
        
        # 2. Mock cache 未命中
        self.mock_cache.get.return_value = None
        
        # 3. Mock Gemini API 响应
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "signal": "BUY",
            "confidence_score": 7,
            "reason": "Price below BB lower band"
        })
        self.mock_client.models.generate_content.return_value = mock_response
        
        # 4. 调用 get_signal
        signal_dict, price = self.strategy.get_signal(
            ticker=self.ticker,
            lookback_minutes=120
        )
        
        # 5. 验证结果
        self.assertEqual(signal_dict['signal'], "BUY")
        self.assertEqual(signal_dict['confidence_score'], 7)
        self.assertGreater(price, 0)
        
        # 6. 验证缓存被调用
        self.mock_cache.get.assert_called_once()
        self.mock_cache.add.assert_called_once()
    
    def test_get_signal_success_with_cache_hit(self):
        """测试成功获取信号（缓存命中）"""
        # 1. Mock data fetcher 返回数据
        mock_data = create_mock_ohlcv_data(num_bars=50)
        self.mock_fetcher.get_latest_bars.return_value = mock_data
        
        # 2. Mock cache 命中
        cached_signal = {
            "signal": "SELL",
            "confidence_score": 9,
            "reason": "Cached result"
        }
        self.mock_cache.get.return_value = cached_signal
        
        # 3. 调用 get_signal
        signal_dict, price = self.strategy.get_signal(
            ticker=self.ticker,
            lookback_minutes=120
        )
        
        # 4. 验证返回缓存结果
        self.assertEqual(signal_dict, cached_signal)
        
        # 5. 验证 Gemini API 未被调用
        self.mock_client.models.generate_content.assert_not_called()
        
        # 6. 验证 cache.add 未被调用
        self.mock_cache.add.assert_not_called()
    
    def test_get_signal_no_data(self):
        """测试无数据时的处理"""
        # Mock fetcher 返回空 DataFrame
        self.mock_fetcher.get_latest_bars.return_value = pd.DataFrame()
        
        signal_dict, price = self.strategy.get_signal(
            ticker=self.ticker,
            lookback_minutes=120
        )
        
        # 验证返回 HOLD 信号
        self.assertEqual(signal_dict['signal'], "HOLD")
        self.assertEqual(signal_dict['confidence_score'], 0)
        self.assertEqual(price, 0.0)
        
        # 验证 API 未被调用
        self.mock_client.models.generate_content.assert_not_called()
    
    def test_get_signal_insufficient_data_after_indicators(self):
        """测试计算指标后数据不足的处理"""
        # Mock fetcher 返回很少的数据（不足以计算指标）
        insufficient_data = create_mock_ohlcv_data(num_bars=10)
        self.mock_fetcher.get_latest_bars.return_value = insufficient_data
        
        signal_dict, price = self.strategy.get_signal(
            ticker=self.ticker,
            lookback_minutes=120
        )
        
        # 验证返回 HOLD 信号
        self.assertEqual(signal_dict['signal'], "HOLD")
        self.assertEqual(signal_dict['confidence_score'], 0)
        self.assertIn("Insufficient", signal_dict['reason'])
    
    @patch('time.sleep')
    def test_get_signal_success_without_cache(self, mock_sleep):
        """测试不使用缓存时成功获取信号"""
        # 创建不使用缓存的策略
        strategy = GeminiStrategy(
            data_fetcher=self.mock_fetcher,
            cache=None,
            use_cache=False,
            delay_seconds=0
        )
        strategy.client = self.mock_client
        
        # Mock data
        mock_data = create_mock_ohlcv_data(num_bars=50)
        self.mock_fetcher.get_latest_bars.return_value = mock_data
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "signal": "HOLD",
            "confidence_score": 5,
            "reason": "Neutral market"
        })
        self.mock_client.models.generate_content.return_value = mock_response
        
        signal_dict, price = strategy.get_signal(
            ticker=self.ticker,
            lookback_minutes=120
        )
        
        # 验证结果
        self.assertEqual(signal_dict['signal'], "HOLD")
        self.assertEqual(signal_dict['confidence_score'], 5)
        
        # 验证 API 被调用
        self.mock_client.models.generate_content.assert_called_once()
    
    def test_custom_system_prompt(self):
        """测试自定义系统提示词"""
        custom_prompt = "You are a conservative trader."
        
        strategy_custom = GeminiStrategy(
            data_fetcher=self.mock_fetcher,
            cache=self.mock_cache,
            system_prompt=custom_prompt
        )
        
        self.assertEqual(strategy_custom.system_prompt, custom_prompt)
    
    def test_client_not_initialized(self):
        """测试客户端未初始化时的处理"""
        strategy_no_client = GeminiStrategy(
            data_fetcher=self.mock_fetcher,
            cache=self.mock_cache
        )
        strategy_no_client.client = None
        
        result = strategy_no_client._call_gemini_api("Test prompt")
        
        self.assertEqual(result['signal'], "HOLD")
        self.assertEqual(result['confidence_score'], 0)
        self.assertIn("not initialized", result['reason'])


if __name__ == '__main__':
    unittest.main()