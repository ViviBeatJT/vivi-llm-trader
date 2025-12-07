# src/data_fetcher/alpaca_data_fetcher.py

import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

# 导入 Alpaca 数据 API 客户端
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

# 加载环境变量（但不在模块级别读取，避免测试时的问题）
load_dotenv()

class AlpacaDataFetcher:
    """
    用于从 Alpaca 获取原始历史 K 线数据的类。
    职责：仅负责获取和返回原始 OHLCV 数据，不进行任何技术指标计算。
    """

    def __init__(self):
        """初始化 Alpaca 客户端。"""
        # 在 __init__ 中读取环境变量，而不是模块级别
        # 这样测试时可以正确模拟环境变量的变化
        api_key = os.getenv('ALPACA_API_KEY_ID')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key:
            print("⚠️ 警告: Alpaca API 密钥未设置。")
            self.data_client = None
        else:
            self.data_client = StockHistoricalDataClient(api_key, secret_key)

    def get_latest_bars(self, 
                       ticker: str, 
                       lookback_minutes: int = 60, 
                       timeframe: TimeFrame = TimeFrame.Minute, 
                       end_dt: Optional[datetime] = None) -> pd.DataFrame:
        """
        从 Alpaca 获取指定时间段的原始 K 线数据 (OHLCV)。
        
        Args:
            ticker: 股票代码
            lookback_minutes: 回溯时间长度（分钟）
            timeframe: K线时间框架
            end_dt: 结束时间（默认为当前UTC时间）
            
        Returns:
            pd.DataFrame: 包含 OHLCV 数据的 DataFrame，索引为时间戳。
                         如果获取失败，返回空 DataFrame。
        """
        if not self.data_client:
            print("❌ Alpaca 客户端未初始化，无法获取数据。")
            return pd.DataFrame()

        # 确定结束时间 (默认使用 UTC 当前时间)
        if end_dt is None:
            end_time = datetime.now(timezone.utc)
        else:
            end_time = end_dt.astimezone(timezone.utc)

        start_time = end_time - timedelta(minutes=lookback_minutes)

        # 构造请求对象
        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=timeframe,
            start=start_time.isoformat(),
            end=end_time.isoformat(),
            feed=DataFeed.IEX  # 使用 IEX 数据源
        )

        try:
            # 获取数据 (返回一个 BarSet)
            bar_set = self.data_client.get_stock_bars(request_params)
            df = bar_set.df
        except Exception as e:
            print(f"❌ 获取 Alpaca 数据失败: {e}")
            return pd.DataFrame()

        if df.empty:
            print(f"⚠️ 未获取到 {ticker} 的数据。")
            return pd.DataFrame()

        # 提取单个股票的 DataFrame
        try:
            # Alpaca 返回的是 MultiIndex (symbol, timestamp)
            ticker_df = df.loc[ticker].copy()
        except KeyError:
            print(f"⚠️ 在返回数据中找不到 {ticker}。")
            return pd.DataFrame()

        print(f"✅ 成功获取 {ticker} 过去 {lookback_minutes} 分钟的 {timeframe.value} K 线数据 (共 {len(ticker_df)} 条)。")
        
        return ticker_df

    def get_latest_price(self, ticker: str) -> float:
        """
        从 Alpaca 获取标的物的最新收盘价。
        用于实时/纸盘模式下的交易执行。
        
        Args:
            ticker: 股票代码
            
        Returns:
            float: 最新收盘价，如果获取失败返回 0.0
        """
        if not self.data_client:
            print("❌ Alpaca 客户端未初始化，无法获取实时价格。")
            return 0.0

        # 只获取最近几分钟的数据
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=5)

        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame.Minute,
            start=start_time.isoformat(),
            end=end_time.isoformat(),
            feed=DataFeed.IEX
        )

        try:
            bar_set = self.data_client.get_stock_bars(request_params)
            df = bar_set.df

            if df.empty:
                raise ValueError(f"无法获取 {ticker} 的最新 K 线数据。")

            # 返回最近一个 bar 的收盘价
            latest_price = df.loc[ticker].iloc[-1]['close']
            print(f"💰 实时价格获取成功: {ticker} 最新收盘价 ${latest_price:.2f}")
            return latest_price
            
        except Exception as e:
            print(f"❌ 实时价格获取失败 ({ticker}): {e}")
            return 0.0


if __name__ == '__main__':
    # 测试用例：获取最近一小时的 TSLA 数据
    fetcher = AlpacaDataFetcher()
    
    df = fetcher.get_latest_bars(ticker="TSLA", lookback_minutes=60)
    print("\n原始数据示例:")
    print(df.head(10))
    print(f"\n数据列: {df.columns.tolist()}")
    
    # 测试用例：获取实时价格
    price = fetcher.get_latest_price(ticker="TSLA")
    print(f"\nTSLA 最新价格: ${price:.2f}")