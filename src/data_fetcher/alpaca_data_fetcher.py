# src/data_fetcher/alpaca_data_fetcher.py

import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional

# 导入 Alpaca 数据 API 客户端
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

# 加载环境变量
load_dotenv()


class AlpacaDataFetcher:
    """
    用于从 Alpaca 获取原始历史 K 线数据的类。
    职责：仅负责获取和返回原始 OHLCV 数据，不进行任何技术指标计算。
    """

    def __init__(self):
        """初始化 Alpaca 客户端。"""
        api_key = os.getenv('ALPACA_API_KEY_ID')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key:
            print("⚠️ 警告: Alpaca API 密钥未设置。")
            self.data_client = None
        else:
            self.data_client = StockHistoricalDataClient(api_key, secret_key)

    def _format_timestamp(self, dt: Optional[datetime]) -> str:
        """格式化时间戳用于日志输出。"""
        if dt is None:
            return "now"
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime('%Y-%m-%d %H:%M UTC')

    def _format_timeframe(self, timeframe: TimeFrame) -> str:
        """格式化 timeframe 用于日志输出。"""
        return f"{timeframe.amount}{timeframe.unit.name[0]}"  # e.g., "5M", "1H", "1D"

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
        
        # 格式化日志信息
        timestamp_str = self._format_timestamp(end_time)
        timeframe_str = self._format_timeframe(timeframe)

        # 构造请求对象
        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=timeframe,
            start=start_time.isoformat(),
            end=end_time.isoformat(),
            feed=DataFeed.IEX
        )

        try:
            bar_set = self.data_client.get_stock_bars(request_params)
            df = bar_set.df
        except Exception as e:
            print(f"❌ [{timestamp_str}] 获取 {ticker} 数据失败: {e}")
            return pd.DataFrame()

        if df.empty:
            print(f"⚠️ [{timestamp_str}] 未获取到 {ticker} 的 {timeframe_str} K线数据 (回溯 {lookback_minutes} 分钟)")
            return pd.DataFrame()

        # 提取单个股票的 DataFrame
        try:
            ticker_df = df.loc[ticker].copy()
        except KeyError:
            print(f"⚠️ [{timestamp_str}] 在返回数据中找不到 {ticker}")
            return pd.DataFrame()

        print(f"✅ [{timestamp_str}] 获取 {ticker} {timeframe_str} K线: {len(ticker_df)} 条 (回溯 {lookback_minutes} 分钟)")
        
        return ticker_df

    def get_latest_price(self, ticker: str) -> float:
        """
        从 Alpaca 获取标的物的最新收盘价。
        
        Args:
            ticker: 股票代码
            
        Returns:
            float: 最新收盘价，如果获取失败返回 0.0
        """
        if not self.data_client:
            print("❌ Alpaca 客户端未初始化，无法获取实时价格。")
            return 0.0

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=5)
        timestamp_str = self._format_timestamp(end_time)

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
                raise ValueError(f"无法获取 {ticker} 的最新 K 线数据")

            latest_price = df.loc[ticker].iloc[-1]['close']
            print(f"💰 [{timestamp_str}] {ticker} 最新价格: ${latest_price:.2f}")
            return latest_price
            
        except Exception as e:
            print(f"❌ [{timestamp_str}] 获取 {ticker} 实时价格失败: {e}")
            return 0.0


if __name__ == '__main__':
    # 测试用例
    fetcher = AlpacaDataFetcher()
    
    print("\n--- 测试 get_latest_bars ---")
    df = fetcher.get_latest_bars(ticker="TSLA", lookback_minutes=60)
    if not df.empty:
        print(f"数据列: {df.columns.tolist()}")
        print(df.tail(3))
    
    print("\n--- 测试 get_latest_bars (指定时间) ---")
    historical_time = datetime(2025, 12, 5, 15, 30, 0, tzinfo=timezone.utc)
    df = fetcher.get_latest_bars(
        ticker="TSLA", 
        lookback_minutes=30,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        end_dt=historical_time
    )
    
    print("\n--- 测试 get_latest_price ---")
    price = fetcher.get_latest_price(ticker="TSLA")