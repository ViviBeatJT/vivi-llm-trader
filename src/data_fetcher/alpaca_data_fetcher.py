# src/data_fetcher/alpaca_data_fetcher.py

import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List

# 导入 Alpaca 数据 API 客户端
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest, StockQuotesRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

# 导入 Alpaca 交易 API 客户端
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass

# 加载环境变量
load_dotenv()


class AlpacaDataFetcher:
    """
    用于从 Alpaca 获取原始历史 K 线数据和账户信息的类。
    职责：
    1. 获取和返回原始 OHLCV 数据
    2. 获取账户状态和持仓信息

    注意：免费账户只能使用 IEX 数据源，不能使用 SIP 数据源。
    """

    def __init__(self, paper: bool = True, data_feed: DataFeed = DataFeed.IEX):
        """
        初始化 Alpaca 客户端。

        Args:
            paper: 是否使用模拟盘 API（默认 True）
            data_feed: 数据源（默认 IEX，免费账户只能用 IEX）
        """
        api_key = os.getenv('ALPACA_API_KEY_ID')
        secret_key = os.getenv('ALPACA_SECRET_KEY')

        if not api_key or not secret_key:
            print("⚠️ 警告: Alpaca API 密钥未设置。")
            self.data_client = None
            self.trading_client = None
        else:
            # 数据客户端（用于获取市场数据）
            self.data_client = StockHistoricalDataClient(api_key, secret_key)
            # 交易客户端（用于获取账户和持仓信息）
            self.trading_client = TradingClient(
                api_key, secret_key, paper=paper)

        self.paper = paper
        self.data_feed = data_feed  # 保存数据源设置

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

    # ==================== 市场数据 API ====================

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

        # 构造请求对象 - 使用实例的 data_feed 设置
        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=timeframe,
            start=start_time.isoformat(),
            end=end_time.isoformat(),
            feed=self.data_feed  # 使用实例设置的数据源
        )

        try:
            bar_set = self.data_client.get_stock_bars(request_params)
            df = bar_set.df
        except Exception as e:
            print(f"❌ [{timestamp_str}] 获取 {ticker} 数据失败: {e}")
            return pd.DataFrame()

        if df.empty:
            print(
                f"⚠️ [{timestamp_str}] 未获取到 {ticker} 的 {timeframe_str} K线数据 (回溯 {lookback_minutes} 分钟)")
            return pd.DataFrame()

        # 提取单个股票的 DataFrame
        try:
            ticker_df = df.loc[ticker].copy()
        except KeyError:
            print(f"⚠️ [{timestamp_str}] 在返回数据中找不到 {ticker}")
            return pd.DataFrame()

        print(
            f"✅ [{timestamp_str}] 获取 {ticker} {timeframe_str} K线: {len(ticker_df)} 条 (回溯 {lookback_minutes} 分钟)")

        return ticker_df

    def get_latest_price(self, ticker: str, current_time: Optional[datetime] = None) -> float:
        """
        从 Alpaca 获取标的物的最新收盘价。

        🔧 修复：使用 IEX 数据源，避免 SIP 订阅错误

        Args:
            ticker: 股票代码
            current_time: 指定时间（用于回测），None 表示获取最新价格

        Returns:
            float: 最新收盘价，如果获取失败返回 0.0
        """
        if not self.data_client:
            print("❌ Alpaca 客户端未初始化，无法获取实时价格。")
            return 0.0

        timestamp_str = self._format_timestamp(datetime.now(timezone.utc))

        try:
            if current_time is None:
                # 🔧 修复：获取最新价格时也使用 IEX 数据源
                # 方法1：使用 StockLatestQuoteRequest 但指定 IEX feed
                try:
                    request_params = StockLatestQuoteRequest(
                        symbol_or_symbols=[ticker],
                        feed=self.data_feed  # 使用 IEX 而不是默认的 SIP
                    )
                    latest_quote = self.data_client.get_stock_latest_quote(
                        request_params)

                    quote = latest_quote[ticker]
                    # 优先使用 bid_price，如果没有则使用 ask_price
                    if quote.bid_price and quote.bid_price > 0:
                        return float(quote.bid_price)
                    elif quote.ask_price and quote.ask_price > 0:
                        return float(quote.ask_price)
                    else:
                        # 如果 quote 没有价格，回退到获取最近的 bar
                        raise ValueError("Quote has no valid price")

                except Exception as quote_error:
                    # 🔧 方法2：如果 quote 失败，回退到获取最近的 bar 数据
                    print(f"⚠️ Quote 获取失败，尝试使用 Bar 数据: {quote_error}")

                    end_time = datetime.now(timezone.utc)
                    start_time = end_time - timedelta(minutes=5)

                    request_params = StockBarsRequest(
                        symbol_or_symbols=[ticker],
                        timeframe=TimeFrame.Minute,
                        start=start_time.isoformat(),
                        end=end_time.isoformat(),
                        feed=self.data_feed  # 使用 IEX
                    )

                    bars_response = self.data_client.get_stock_bars(
                        request_params)
                    bars_df = bars_response.df

                    if not bars_df.empty:
                        # 获取最后一根 bar 的收盘价
                        if ticker in bars_df.index.get_level_values(0):
                            ticker_bars = bars_df.loc[ticker]
                            return float(ticker_bars.iloc[-1]['close'])
                        else:
                            return float(bars_df.iloc[-1]['close'])
                    else:
                        print(f"⚠️ [{timestamp_str}] 无法获取 {ticker} 的最新价格")
                        return 0.0
            else:
                # 回测模式：获取指定时间的价格
                start_time = current_time - timedelta(minutes=1)
                end_time = current_time

                request_params = StockBarsRequest(
                    symbol_or_symbols=[ticker],
                    timeframe=TimeFrame.Minute,
                    start=start_time.isoformat(),
                    end=end_time.isoformat(),
                    feed=self.data_feed  # 🔧 修复：使用 IEX
                )

                bars_response = self.data_client.get_stock_bars(request_params)
                bars_df = bars_response.df

                if not bars_df.empty:
                    # 处理多级索引
                    if ticker in bars_df.index.get_level_values(0):
                        ticker_bars = bars_df.loc[ticker]
                        close_price = ticker_bars.iloc[-1]['close']
                    else:
                        close_price = bars_df.iloc[-1]['close']
                    return float(close_price)
                else:
                    print(f"⚠️ 在 {current_time} 这一分钟未找到 {ticker} 的 Bar 数据。")
                    return 0.0

        except Exception as e:
            print(f"❌ [{timestamp_str}] 获取 {ticker} 实时价格失败: {e}")
            return 0.0

    # ==================== 账户与持仓 API ====================

    def get_account(self) -> Dict[str, Any]:
        """
        获取 Alpaca 账户信息。

        Returns:
            dict: 账户信息，包含以下字段：
                - cash: 可用现金
                - portfolio_value: 总资产价值
                - buying_power: 购买力
                - equity: 权益
                - currency: 货币类型
                - account_blocked: 账户是否被冻结
                - trading_blocked: 交易是否被冻结
                - pattern_day_trader: 是否为日内交易者
                - daytrading_buying_power: 日内交易购买力
        """
        if not self.trading_client:
            print("❌ Alpaca 交易客户端未初始化，无法获取账户信息。")
            return {}

        timestamp_str = self._format_timestamp(datetime.now(timezone.utc))

        try:
            account = self.trading_client.get_account()

            account_info = {
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'currency': account.currency,
                'account_blocked': account.account_blocked,
                'trading_blocked': account.trading_blocked,
                'pattern_day_trader': account.pattern_day_trader,
                'daytrading_buying_power': float(account.daytrading_buying_power) if account.daytrading_buying_power else 0.0,
                'last_equity': float(account.last_equity) if account.last_equity else 0.0,
            }

            mode_str = "模拟盘" if self.paper else "实盘"
            print(f"💼 [{timestamp_str}] 获取 {mode_str} 账户信息成功")
            print(f"   现金: ${account_info['cash']:,.2f}")
            print(f"   总权益: ${account_info['equity']:,.2f}")
            print(f"   购买力: ${account_info['buying_power']:,.2f}")

            return account_info

        except Exception as e:
            print(f"❌ [{timestamp_str}] 获取账户信息失败: {e}")
            return {}

    def get_position(self, ticker: str) -> Dict[str, Any]:
        """
        获取指定股票的持仓信息。

        Args:
            ticker: 股票代码

        Returns:
            dict: 持仓信息，包含以下字段：
                - symbol: 股票代码
                - qty: 持仓数量
                - avg_entry_price: 平均成本价
                - market_value: 市值
                - current_price: 当前价格
                - unrealized_pl: 未实现盈亏
                - unrealized_plpc: 未实现盈亏百分比
                - side: 持仓方向 (long/short)
            如果无持仓，返回空字典
        """
        if not self.trading_client:
            print("❌ Alpaca 交易客户端未初始化，无法获取持仓信息。")
            return {}

        timestamp_str = self._format_timestamp(datetime.now(timezone.utc))

        try:
            position = self.trading_client.get_open_position(ticker)

            position_info = {
                'symbol': position.symbol,
                'qty': float(position.qty),
                'avg_entry_price': float(position.avg_entry_price),
                'market_value': float(position.market_value),
                'current_price': float(position.current_price),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc),
                'side': position.side.value,
                'cost_basis': float(position.cost_basis),
            }

            print(f"📊 [{timestamp_str}] {ticker} 持仓信息:")
            print(
                f"   数量: {position_info['qty']:.0f} 股 ({position_info['side']})")
            print(f"   均价: ${position_info['avg_entry_price']:.2f}")
            print(f"   现价: ${position_info['current_price']:.2f}")
            print(f"   市值: ${position_info['market_value']:,.2f}")
            print(
                f"   盈亏: ${position_info['unrealized_pl']:,.2f} ({position_info['unrealized_plpc']*100:.2f}%)")

            return position_info

        except Exception as e:
            # 如果没有持仓，API 会抛出异常
            if "position does not exist" in str(e).lower():
                print(f"📊 [{timestamp_str}] {ticker} 无持仓")
                return {}
            print(f"❌ [{timestamp_str}] 获取 {ticker} 持仓信息失败: {e}")
            return {}

    def get_all_positions(self) -> List[Dict[str, Any]]:
        """
        获取所有持仓信息。

        Returns:
            list: 持仓列表，每个元素为一个持仓字典
        """
        if not self.trading_client:
            print("❌ Alpaca 交易客户端未初始化，无法获取持仓信息。")
            return []

        timestamp_str = self._format_timestamp(datetime.now(timezone.utc))

        try:
            positions = self.trading_client.get_all_positions()

            if not positions:
                print(f"📊 [{timestamp_str}] 当前无任何持仓")
                return []

            position_list = []
            for position in positions:
                position_info = {
                    'symbol': position.symbol,
                    'qty': float(position.qty),
                    'avg_entry_price': float(position.avg_entry_price),
                    'market_value': float(position.market_value),
                    'current_price': float(position.current_price),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'side': position.side.value,
                    'cost_basis': float(position.cost_basis),
                }
                position_list.append(position_info)

            print(f"📊 [{timestamp_str}] 获取到 {len(position_list)} 个持仓:")
            for pos in position_list:
                pnl_str = f"+${pos['unrealized_pl']:.2f}" if pos[
                    'unrealized_pl'] >= 0 else f"-${abs(pos['unrealized_pl']):.2f}"
                print(
                    f"   {pos['symbol']}: {pos['qty']:.0f} 股 @ ${pos['avg_entry_price']:.2f} | {pnl_str}")

            return position_list

        except Exception as e:
            print(f"❌ [{timestamp_str}] 获取所有持仓失败: {e}")
            return []

    def sync_position_status(self, ticker: str) -> Dict[str, Any]:
        """
        同步指定股票的仓位状态（用于 PositionManager 同步）。

        Args:
            ticker: 股票代码

        Returns:
            dict: 同步后的状态，包含：
                - cash: 可用现金
                - position: 持仓数量
                - avg_cost: 平均成本
                - equity: 总权益
                - market_value: 持仓市值
                - current_price: 当前价格
        """
        timestamp_str = self._format_timestamp(datetime.now(timezone.utc))

        # 获取账户信息
        account = self.get_account()
        if not account:
            return {}

        # 获取持仓信息
        position = self.get_position(ticker)

        status = {
            'cash': account.get('cash', 0.0),
            'position': position.get('qty', 0.0),
            'avg_cost': position.get('avg_entry_price', 0.0),
            'equity': account.get('equity', 0.0),
            'market_value': position.get('market_value', 0.0),
            'current_price': position.get('current_price', 0.0),
        }

        print(f"🔄 [{timestamp_str}] {ticker} 仓位状态同步完成")

        return status


if __name__ == '__main__':
    # 测试用例
    print("=" * 60)
    print("测试 AlpacaDataFetcher (使用 IEX 数据源)")
    print("=" * 60)

    fetcher = AlpacaDataFetcher(paper=True, data_feed=DataFeed.IEX)

    print("\n--- 测试 get_latest_price (TSLA) ---")
    price = fetcher.get_latest_price("TSLA")
    print(f"TSLA 最新价格: ${price:.2f}")

    print("\n--- 测试 get_latest_bars (TSLA) ---")
    bars = fetcher.get_latest_bars("TSLA", lookback_minutes=60)
    if not bars.empty:
        print(f"获取到 {len(bars)} 根 K线")
        print(bars.tail())

    print("\n--- 测试 get_account ---")
    account = fetcher.get_account()

    print("\n--- 测试 get_all_positions ---")
    positions = fetcher.get_all_positions()

    print("\n--- 测试 get_position (TSLA) ---")
    position = fetcher.get_position("TSLA")

    print("\n--- 测试 sync_position_status (TSLA) ---")
    status = fetcher.sync_position_status("TSLA")
