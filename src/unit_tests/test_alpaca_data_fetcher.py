# src/test/test_alpaca_data_fetcher.py

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import os 

# 导入需要测试的类和常量
from src.data.alpaca_data_fetcher import AlpacaDataFetcher
from alpaca.data.timeframe import TimeFrame

# --- 辅助函数：创建模拟的 Alpaca BarSet DataFrame ---
def create_mock_multiindex_df(ticker: str, num_bars: int = 30) -> pd.DataFrame:
    """
    创建包含 OHLCV 数据的模拟 MultiIndex DataFrame.
    关键修复：索引顺序必须是 (symbol, timestamp) 以匹配 Alpaca API 的行为。
    """
    
    # 设定一个基准时间
    base_time = datetime(2025, 12, 5, 9, 0, 0, tzinfo=timezone.utc)
    
    # 创建时间索引 (以分钟为间隔)
    time_index = [base_time + timedelta(minutes=i) for i in range(num_bars)]
    
    # 创建模拟价格 (价格围绕 100 波动，模拟上涨趋势)
    base_price = 100
    prices = base_price + np.cumsum(np.random.randn(num_bars) * 0.1 + 0.05)
    
    # 模拟 OHLCV 数据
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
    
    # 【关键修复】设置索引顺序为 ['symbol', 'timestamp']
    # 这样 df.loc[ticker] 才能正确工作，不会把 ticker 当作时间解析
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
        # 1. 模拟环境变量并实例化 Fetcher
        with patch.dict('os.environ', {'ALPACA_API_KEY_ID': 'mock_key', 'ALPACA_SECRET_KEY': 'mock_secret'}):
            self.fetcher = AlpacaDataFetcher()

        # 2. 【关键修复】直接替换实例中的 data_client 为 MagicMock
        # 这样确保测试方法中对 self.mock_client 的配置能生效
        self.mock_client = MagicMock()
        self.fetcher.data_client = self.mock_client

        # 3. 准备通用的测试数据
        self.ticker = "MOCK"
        self.num_bars = 30
        
        # 创建 MultiIndex DF (用于 MockBarSet)
        self.mock_multiindex_df = create_mock_multiindex_df(self.ticker, self.num_bars)
        
        # 从 MultiIndex 中提取单股票 DF (用于 indicator_calculation 测试)
        # 因为索引现在是 (symbol, timestamp)，直接用 .loc[ticker] 即可提取
        self.mock_single_ticker_df = self.mock_multiindex_df.loc[self.ticker].copy()
        
        # MockBarSet 直接使用 MultiIndex DF
        self.mock_bar_set = MockBarSet(self.mock_multiindex_df)

    # --- 测试：布林带和 RSI 计算 ---
    def test_indicator_calculation(self):
        """测试布林带和RSI计算方法的准确性"""
        
        # 复制数据以进行测试，传入的是单股票 DF
        df = self.mock_single_ticker_df.copy()

        # 1. 计算布林带 (周期 20)
        df = self.fetcher._calculate_bollinger_band(df)
        
        # 【修复】验证 BB NaN 行数 (前 19 行应为 NaN)
        # BB_PERIOD = 20, 所以前 19 个数据无法计算 MA/STD
        expected_bb_nan = self.fetcher.BB_PERIOD - 1
        self.assertEqual(df['SMA'].iloc[:expected_bb_nan].isnull().sum(), expected_bb_nan,
                         msg="SMA 起始 NaN 行数不正确")
        
        # 验证数值
        expected_sma_last = df['close'].iloc[-self.fetcher.BB_PERIOD:].mean()
        self.assertAlmostEqual(df['SMA'].iloc[-1], expected_sma_last, places=5)

        # 2. 计算 RSI (窗口 14)
        df = self.fetcher._calculate_rsi(df)
        
        # 【修复】验证 RSI NaN 行数 (前 13 行应为 NaN)
        # RSI_WINDOW = 14
        expected_rsi_nan = self.fetcher.RSI_WINDOW - 1
        self.assertEqual(df['RSI'].iloc[:expected_rsi_nan].isnull().sum(), expected_rsi_nan,
                         msg="RSI 起始 NaN 行数不正确")
        
        self.assertTrue('RSI' in df.columns)


    # --- 测试：成功获取 K 线数据并计算指标 ---
    def test_get_latest_bars_success(self):
        """测试成功获取 K 线数据并返回正确的格式和DataFrame"""
        
        # 配置 Mock Client 行为 (使用 setUp 中创建的 self.mock_client)
        self.mock_client.get_stock_bars.return_value = self.mock_bar_set

        # 调用方法
        text_output, df_output = self.fetcher.get_latest_bars(
            ticker=self.ticker, 
            lookback_minutes=30, 
            timeframe=TimeFrame.Minute
        )

        # 1. 验证方法调用
        self.mock_client.get_stock_bars.assert_called_once()

        # 2. 验证 DataFrame 内容
        # 最终有效行数应为 num_bars - BB_PERIOD + 1 (dropna 删除后的行数)
        # 30 - 20 + 1 = 11 行
        expected_valid_rows = self.num_bars - self.fetcher.BB_PERIOD + 1
        
        self.assertEqual(len(df_output), expected_valid_rows, 
                         msg=f"DataFrame 的长度不正确。预期 {expected_valid_rows}, 实际 {len(df_output)}")
        self.assertTrue('BB_UPPER' in df_output.columns and 'RSI' in df_output.columns,
                        msg="输出 DataFrame 缺少技术指标列。")

        # 3. 验证 LLM 文本格式
        self.assertIsInstance(text_output, str)
        self.assertTrue(f"股票代码: {self.ticker}" in text_output)


    # --- 测试：获取 K 线数据但返回空结果 ---
    def test_get_latest_bars_empty(self):
        """测试 Alpaca API 返回空 BarSet 时的处理"""
        
        # 创建一个空的 MultiIndex DataFrame
        # 同样需要正确的 MultiIndex 结构 (symbol, timestamp)
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'],
                                index=pd.MultiIndex.from_tuples([], names=['symbol', 'timestamp']))
        empty_bar_set = MockBarSet(empty_df)

        # 配置 Mock Client 行为
        self.mock_client.get_stock_bars.return_value = empty_bar_set

        # 调用方法
        text_output, df_output = self.fetcher.get_latest_bars(
            ticker=self.ticker, 
            lookback_minutes=30
        )

        # 验证返回
        self.assertTrue("没有找到可用的 K 线数据。" in text_output)
        self.assertTrue(df_output.empty)


    # --- 测试：成功获取最新价格 ---
    def test_get_latest_price_success(self):
        """测试成功获取最新的收盘价格"""
        
        # 创建包含 5 个 bar 的模拟 MultiIndex数据 (symbol, timestamp)
        price_df_multiindex = create_mock_multiindex_df(self.ticker, 5)
        
        # 设定一个确定的最新价格
        latest_close_price = 105.55
        
        # 通过 MultiIndex 定位并修改最新的收盘价
        # 注意：现在 symbol 是第一层索引
        latest_timestamp = price_df_multiindex.index.get_level_values('timestamp')[-1]
        price_df_multiindex.loc[(self.ticker, latest_timestamp), 'close'] = latest_close_price
        
        # 使用这个 price_df 创建 MockBarSet
        mock_bar_set_price = MockBarSet(price_df_multiindex)
        
        # 配置 Mock Client 行为
        self.mock_client.get_stock_bars.return_value = mock_bar_set_price

        # 调用方法
        price = self.fetcher.get_latest_price(self.ticker)

        # 验证返回的价格
        self.assertAlmostEqual(price, latest_close_price, places=2,
                               msg="get_latest_price 返回的价格不正确。")
        
    
    # --- 测试：获取最新价格失败（价格为 0.0） ---
    @patch('builtins.print') # 屏蔽 print 输出
    def test_get_latest_price_failure(self, mock_print):
        """测试获取最新价格时 API 抛出异常或返回空 DataFrame 时的处理"""
        
        # 1. 模拟 API 抛出异常
        self.mock_client.get_stock_bars.side_effect = Exception("API connection error")

        price = self.fetcher.get_latest_price(self.ticker)
        self.assertEqual(price, 0.0, msg="API 异常时应返回 0.0")

        # 2. 模拟 API 返回空数据
        self.mock_client.get_stock_bars.side_effect = None # 重置
        
        # 空的 MultiIndex DataFrame (symbol, timestamp)
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'],
                                index=pd.MultiIndex.from_tuples([], names=['symbol', 'timestamp']))
        empty_bar_set = MockBarSet(empty_df)
        
        self.mock_client.get_stock_bars.return_value = empty_bar_set
        
        # get_latest_price 在 df.empty 时会抛出 ValueError，我们检查它是否被捕获
        price = self.fetcher.get_latest_price(self.ticker)
        self.assertEqual(price, 0.0, msg="API 返回空数据时应返回 0.0")


if __name__ == '__main__':
    unittest.main()