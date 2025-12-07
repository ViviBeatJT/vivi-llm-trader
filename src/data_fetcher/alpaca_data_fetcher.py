# src/data/alpaca_data_fetcher.py

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

# 导入配置
load_dotenv()
API_KEY_ID = os.getenv('ALPACA_API_KEY_ID')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

class AlpacaDataFetcher:
    """
    一个用于从 Alpaca 获取历史 K 线数据并计算技术指标的类。
    """
    
    # 默认配置参数
    BB_PERIOD = 20    # 布林带周期
    BB_STD_DEV = 2    # 布林带标准差
    RSI_WINDOW = 14   # RSI 窗口期

    def __init__(self):
        """初始化 Alpaca 客户端。"""
        # 在测试环境下，如果环境变量缺失，允许初始化但不建立连接（或者抛出错误）
        # 这里保留原有逻辑
        if not API_KEY_ID or not SECRET_KEY:
             # 为了避免导入时的硬性崩溃，这里可以仅仅打印警告，但在实际调用时会失败
             # 或者保持抛出 ValueError
             pass 
            
        # 初始化 Alpaca 客户端
        # 注意：如果在没有 key 的情况下运行，这一步可能会报错，取决于 Alpaca SDK 的实现
        if API_KEY_ID and SECRET_KEY:
            self.data_client = StockHistoricalDataClient(API_KEY_ID, SECRET_KEY)
        else:
            self.data_client = None

    # --- 辅助方法：技术指标计算 ---
    def _calculate_bollinger_band(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算布林带 (Bollinger Bands)"""
        # 使用类属性进行计算
        df['SMA'] = df['close'].rolling(window=self.BB_PERIOD).mean()
        df['STD'] = df['close'].rolling(window=self.BB_PERIOD).std()
        df['BB_UPPER'] = df['SMA'] + (df['STD'] * self.BB_STD_DEV)
        df['BB_LOWER'] = df['SMA'] - (df['STD'] * self.BB_STD_DEV)
        return df

    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算相对强弱指数 (RSI)"""
        # 使用类属性进行计算
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.RSI_WINDOW).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.RSI_WINDOW).mean()
        RS = gain / loss
        df['RSI'] = 100 - (100 / (1 + RS))
        return df

    # --- 辅助方法：格式化 K 线数据 ---
    def _format_bars_for_llm(self, df: pd.DataFrame, ticker: str) -> str:
        """
        将包含 OHLCV 和 TA 指标的 DataFrame 转换为 LLM 友好的 Markdown 表格。
        """
        if df.empty:
            return "没有找到可用的 K 线数据。"

        # 【修复】使用副本操作，避免修改原始 df 的索引
        df_display = df.copy()

        # 将索引 (时间戳) 格式化 (转换为纽约时间以匹配 Alpaca 常用惯例)
        df_display.index = df_display.index.tz_convert('America/New_York').strftime('%H:%M')

        # 选择需要显示的列（最后 10 个 bar）
        df_display = df_display.tail(10)

        # 选择需要的列并重命名
        # 确保列名存在于 DataFrame 中
        cols_to_keep = ['open', 'high', 'low', 'close', 'volume', 'SMA', 'BB_UPPER', 'BB_LOWER', 'RSI']
        # 过滤掉不存在的列（防止出错）
        cols_to_keep = [c for c in cols_to_keep if c in df_display.columns]
        
        df_display = df_display[cols_to_keep]
        
        # 定义列名映射
        col_mapping = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume',
            'SMA': 'SMA', 'BB_UPPER': 'BB_Upper', 'BB_LOWER': 'BB_Lower', 'RSI': 'RSI'
        }
        df_display.rename(columns=col_mapping, inplace=True)
        
        # 格式化浮点数为两位小数
        float_cols = ['Open', 'High', 'Low', 'Close', 'SMA', 'BB_Upper', 'BB_Lower', 'RSI']
        for col in float_cols:
            if col in df_display.columns:
                df_display[col] = df_display[col].round(2)

        # 转换为 Markdown 字符串
        markdown_table = df_display.to_markdown(numalign="left", stralign="left")

        # 【修复】使用传入的 ticker 参数，而不是 df.index.name
        return f"股票代码: {ticker}\n技术指标 K 线数据:\n{markdown_table}"

    # --- 核心方法：获取 LLM K 线数据 ---
    def get_latest_bars(self, ticker: str, lookback_minutes: int = 60, timeframe: TimeFrame = TimeFrame.Minute, end_dt: Optional[datetime] = None) -> Tuple[str, pd.DataFrame]:
        """
        从 Alpaca 获取指定时间段的 K 线数据，计算技术指标，并格式化为 LLM 友好的文本。
        
        Returns:
            Tuple[str, pd.DataFrame]: 格式化的文本数据和包含指标的 DataFrame。
        """
        if not self.data_client:
             # 如果客户端未初始化（例如缺少 API Key），返回空
             print("❌ Alpaca 客户端未初始化，无法获取数据。")
             return self._format_bars_for_llm(pd.DataFrame(), ticker), pd.DataFrame()

        # 确定结束时间 (默认使用 UTC 当前时间)
        if end_dt is None:
            end_time = datetime.now(timezone.utc).astimezone(timezone.utc)
        else:
            end_time = end_dt.astimezone(timezone.utc)

        start_time = end_time - timedelta(minutes=lookback_minutes)

        # 构造请求对象
        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=timeframe,
            start=start_time.isoformat(),
            end=end_time.isoformat(),
            feed=DataFeed.IEX # 使用 IEX 数据源
        )

        try:
            # 获取数据 (返回一个 BarSet)
            bar_set = self.data_client.get_stock_bars(request_params)
            df = bar_set.df
        except Exception as e:
            print(f"❌ 获取 Alpaca 数据失败: {e}")
            return self._format_bars_for_llm(pd.DataFrame(), ticker), pd.DataFrame()

        if df.empty:
            return self._format_bars_for_llm(df, ticker), pd.DataFrame()

        # 提取单个股票的 DataFrame 并计算指标
        # 注意：Alpaca 返回的是 MultiIndex (symbol, timestamp) 或 (timestamp, symbol)，这里假设使用 loc[ticker] 获取单股票数据
        try:
            # 尝试通过 loc 获取 ticker 数据，这会自动处理 MultiIndex
            ticker_df = df.loc[ticker].copy()
        except KeyError:
             # 处理可能索引不匹配的情况
             print(f"⚠️ 在返回数据中找不到 {ticker}。")
             return self._format_bars_for_llm(pd.DataFrame(), ticker), pd.DataFrame()
        
        # --- 1. 计算布林带 (Bollinger Bands) ---
        ticker_df = self._calculate_bollinger_band(ticker_df)

        # --- 2. 计算 RSI (Relative Strength Index) ---
        ticker_df = self._calculate_rsi(ticker_df)

        # 删除 NaN 行 (因为需要 20 个周期的数据来计算指标)
        ticker_df = ticker_df.dropna()
        
        # 如果计算完指标后数据为空，也视为失败
        if ticker_df.empty:
             return self._format_bars_for_llm(pd.DataFrame(), ticker), pd.DataFrame()

        print(f"✅ 成功获取 {ticker} 过去 {lookback_minutes} 分钟的 {timeframe.value} K 线数据。")
        
        # 格式化为 LLM 文本
        # 【更新】传入 ticker 参数
        formatted_bars = self._format_bars_for_llm(ticker_df, ticker)
        
        return formatted_bars, ticker_df

    # --- 专门用于实时模式获取最新收盘价的方法 ---
    def get_latest_price(self, ticker: str) -> float:
        """
        从 Alpaca 获取标的物的最新收盘价。
        用于实时/纸盘模式下的交易执行。
        """
        if not self.data_client:
            print("❌ Alpaca 客户端未初始化，无法获取实时价格。")
            return 0.0

        # 只获取最近一分钟的数据
        end_time = datetime.now(timezone.utc).astimezone(timezone.utc)
        start_time = end_time - timedelta(minutes=5) # 稍微多获取一点以确保拿到数据

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
            # 处理 MultiIndex
            latest_price = df.loc[ticker].iloc[-1]['close']
            print(f"💰 实时价格获取成功: {ticker} 最新收盘价 ${latest_price:.2f}")
            return latest_price
            
        except Exception as e:
            print(f"❌ 实时价格获取失败 ({ticker}): {e}")
            # 在实时模式下，如果获取失败，抛出错误或返回 0.0 防止错误交易
            return 0.0

if __name__ == '__main__':
    # 实例化数据获取器
    fetcher = AlpacaDataFetcher()
    
    # 测试用例：获取最近一小时的 TSLA 数据，并格式化给 LLM
    text, df = fetcher.get_latest_bars(ticker="TSLA", lookback_minutes=60)
    print(text)

    # 测试用例：获取实时价格
    price = fetcher.get_latest_price(ticker="TSLA")
    print(f"TSLA latest price: {price}")