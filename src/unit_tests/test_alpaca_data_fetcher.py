# src/unit_tests/test_alpaca_data_fetcher.py

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import os 

# 导入需要测试的类
from src.data_fetcher.alpaca_data_fetcher import AlpacaDataFetcher
from alpaca.data.timeframe import TimeFrame

# --- 辅助函数：创建模拟的 Alpaca BarSet DataFrame ---
def create_mock_multiindex_df(ticker: str, num_bars: int = 30) -> pd.DataFrame:
    """
    创建包含 OHLCV 数据的模拟 MultiIndex DataFrame.
    索引顺序为 (symbol, timestamp) 以匹配 Alpaca API 的行为。
    """
    base_time = datetime(2025, 12, 5, 9, 0, 0, tzinfo=timezone.utc)
    time_index = [base_time + timedelta(minutes=i) for i in range(num_bars)]
    
    base_price = 100
    prices = base_price + np.cumsum(np.random.randn(num_bars) * 0.1 + 0.05)
    
    data = {
        'open': prices - 0.1,
        'high': prices + 0.2,
        'low': prices - 0.2,
        'close': prices,
        'volume': np.random.randint(1000, 5000, num_bars),
        'symbol': [ticker] * num_bars, 
        'timestamp': time_index
    }
    
    df = pd.DataFrame(data)
    df.set_index(['symbol', 'timestamp'], inplace=True)
    
    return df

# --- 模拟 Alpaca BarSet 对象的容器类 ---
class MockBarSet:
    """模拟 Alpaca API 返回的 BarSet 对象"""
    def __init__(self, multiindex_df: pd.DataFrame):
        self.df = multiindex_df


class TestAlpacaDataFetcher(unittest.TestCase):
    
    def setUp(self):
        """为每个测试方法设置环境"""
        # 模拟环境变量
        with patch.dict('os.environ', {
            'ALPACA_API_KEY_ID': 'mock_key', 
            'ALPACA_SECRET_KEY': 'mock_secret'
        }):
            self.fetcher = AlpacaDataFetcher()

        # 直接替换实例中的 data_client 为 MagicMock
        self.mock_client = MagicMock()
        self.fetcher.data_client = self.mock_client

        # 准备通用的测试数据
        self.ticker = "MOCK"
        self.num_bars = 30
        
        # 创建 MultiIndex DF
        self.mock_multiindex_df = create_mock_multiindex_df(self.ticker, self.num_bars)
        
        # 从 MultiIndex 中提取单股票 DF
        self.mock_single_ticker_df = self.mock_multiindex_df.loc[self.ticker].copy()
        
        # MockBarSet 使用 MultiIndex DF
        self.mock_bar_set = MockBarSet(self.mock_multiindex_df)

    def test_get_latest_bars_success(self):
        """测试成功获取原始 K 线数据"""
        # 配置 Mock Client 行为
        self.mock_client.get_stock_bars.return_value = self.mock_bar_set

        # 调用方法
        df_output = self.fetcher.get_latest_bars(
            ticker=self.ticker, 
            lookback_minutes=30, 
            timeframe=TimeFrame.Minute
        )

        # 1. 验证方法调用
        self.mock_client.get_stock_bars.assert_called_once()

        # 2. 验证 DataFrame 内容 - 应返回原始数据，没有技术指标
        self.assertEqual(len(df_output), self.num_bars)
        
        # 验证只有原始 OHLCV 列，没有技术指标
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        self.assertEqual(df_output.columns.tolist(), expected_columns)
        
        # 验证没有 BB 或 RSI 列
        self.assertNotIn('BB_UPPER', df_output.columns)
        self.assertNotIn('RSI', df_output.columns)
        self.assertNotIn('SMA', df_output.columns)

    def test_get_latest_bars_empty(self):
        """测试 Alpaca API 返回空 BarSet 时的处理"""
        empty_df = pd.DataFrame(
            columns=['open', 'high', 'low', 'close', 'volume'],
            index=pd.MultiIndex.from_tuples([], names=['symbol', 'timestamp'])
        )
        empty_bar_set = MockBarSet(empty_df)

        self.mock_client.get_stock_bars.return_value = empty_bar_set

        df_output = self.fetcher.get_latest_bars(
            ticker=self.ticker, 
            lookback_minutes=30
        )

        # 验证返回空 DataFrame
        self.assertTrue(df_output.empty)

    def test_get_latest_bars_api_error(self):
        """测试 API 调用失败时的处理"""
        self.mock_client.get_stock_bars.side_effect = Exception("API Error")

        df_output = self.fetcher.get_latest_bars(
            ticker=self.ticker, 
            lookback_minutes=30
        )

        # 验证返回空 DataFrame
        self.assertTrue(df_output.empty)

    def test_get_latest_price_success(self):
        """测试成功获取最新的收盘价格"""
        # 创建包含 5 个 bar 的模拟数据
        price_df_multiindex = create_mock_multiindex_df(self.ticker, 5)
        
        # 设定一个确定的最新价格
        latest_close_price = 105.55
        latest_timestamp = price_df_multiindex.index.get_level_values('timestamp')[-1]
        price_df_multiindex.loc[(self.ticker, latest_timestamp), 'close'] = latest_close_price
        
        mock_bar_set_price = MockBarSet(price_df_multiindex)
        
        # 配置 Mock Client 行为
        self.mock_client.get_stock_bars.return_value = mock_bar_set_price

        # 调用方法
        price = self.fetcher.get_latest_price(self.ticker)

        # 验证返回的价格
        self.assertAlmostEqual(price, latest_close_price, places=2)
        
    @patch('builtins.print')
    def test_get_latest_price_failure(self, mock_print):
        """测试获取最新价格失败时返回 0.0"""
        # 模拟 API 抛出异常
        self.mock_client.get_stock_bars.side_effect = Exception("API connection error")

        price = self.fetcher.get_latest_price(self.ticker)
        self.assertEqual(price, 0.0)

        # 模拟 API 返回空数据
        self.mock_client.get_stock_bars.side_effect = None
        empty_df = pd.DataFrame(
            columns=['open', 'high', 'low', 'close', 'volume'],
            index=pd.MultiIndex.from_tuples([], names=['symbol', 'timestamp'])
        )
        empty_bar_set = MockBarSet(empty_df)
        self.mock_client.get_stock_bars.return_value = empty_bar_set
        
        price = self.fetcher.get_latest_price(self.ticker)
        self.assertEqual(price, 0.0)

    @patch('builtins.print')  # Suppress print output during test
    def test_data_client_not_initialized(self, mock_print):
        """测试当 API 密钥缺失时，data_client 为 None"""
        # Clear environment variables to simulate missing API keys
        with patch.dict('os.environ', {}, clear=True):
            fetcher_no_key = AlpacaDataFetcher()
            
            # Verify data_client is None when keys are missing
            self.assertIsNone(fetcher_no_key.data_client)
            
            # Verify methods return empty results
            df = fetcher_no_key.get_latest_bars("TSLA", 30)
            self.assertTrue(df.empty)
            
            price = fetcher_no_key.get_latest_price("TSLA")
            self.assertEqual(price, 0.0)


if __name__ == '__main__':
    unittest.main()