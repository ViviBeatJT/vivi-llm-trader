# src/data_fetcher/alpaca_data_fetcher.py

import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List

# 导入 Alpaca 数据 API 客户端
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest, StockLatestBarRequest
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
            self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        
        self.paper = paper
        self.data_feed = data_feed
        
        # 缓存最后已知价格（用于非交易时段）
        self._last_known_price: Dict[str, float] = {}

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

    def _is_market_hours(self, dt: datetime = None) -> bool:
        """
        检查是否在美股交易时段
        美股交易时间: 9:30 AM - 4:00 PM ET (东部时间)
        """
        import pytz
        
        if dt is None:
            dt = datetime.now(timezone.utc)
        
        # 转换为美东时间
        et = pytz.timezone('America/New_York')
        et_time = dt.astimezone(et)
        
        # 检查是否是工作日
        if et_time.weekday() >= 5:  # 周六=5, 周日=6
            return False
        
        # 检查时间
        market_open = et_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = et_time.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= et_time <= market_close

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

        # 构造请求对象
        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=timeframe,
            start=start_time.isoformat(),
            end=end_time.isoformat(),
            feed=self.data_feed
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

        # 更新缓存的最后价格
        if not ticker_df.empty:
            self._last_known_price[ticker] = float(ticker_df.iloc[-1]['close'])

        print(f"✅ [{timestamp_str}] 获取 {ticker} {timeframe_str} K线: {len(ticker_df)} 条 (回溯 {lookback_minutes} 分钟)")
        
        return ticker_df

    def get_latest_price(self, ticker: str, current_time: Optional[datetime] = None) -> float:
        """
        从 Alpaca 获取标的物的最新价格。
        
        采用多级回退策略：
        1. 尝试获取最新 Quote
        2. 尝试获取最新 Bar
        3. 尝试获取最近几分钟的 Bar
        4. 使用缓存的最后已知价格
        
        Args:
            ticker: 股票代码
            current_time: 指定时间（用于回测），None 表示获取最新价格
            
        Returns:
            float: 最新价格，如果获取失败返回 0.0
        """
        if not self.data_client:
            print("❌ Alpaca 客户端未初始化，无法获取实时价格。")
            return 0.0

        timestamp_str = self._format_timestamp(datetime.now(timezone.utc))

        # ========== 回测模式：获取指定时间的价格 ==========
        if current_time is not None:
            return self._get_historical_price(ticker, current_time)

        # ========== 实时模式：多级回退策略 ==========
        
        # 策略1: 尝试获取最新 Bar（最可靠）
        try:
            request_params = StockLatestBarRequest(
                symbol_or_symbols=[ticker],
                feed=self.data_feed
            )
            latest_bar = self.data_client.get_stock_latest_bar(request_params)
            
            if ticker in latest_bar and latest_bar[ticker]:
                price = float(latest_bar[ticker].close)
                self._last_known_price[ticker] = price
                return price
        except Exception as e:
            print(f"⚠️ 获取最新 Bar 失败: {e}")

        # 策略2: 尝试获取最新 Quote
        try:
            request_params = StockLatestQuoteRequest(
                symbol_or_symbols=[ticker],
                feed=self.data_feed
            )
            latest_quote = self.data_client.get_stock_latest_quote(request_params)
            
            if ticker in latest_quote:
                quote = latest_quote[ticker]
                # 优先使用 bid_price，然后 ask_price
                if quote.bid_price and quote.bid_price > 0:
                    price = float(quote.bid_price)
                    self._last_known_price[ticker] = price
                    return price
                elif quote.ask_price and quote.ask_price > 0:
                    price = float(quote.ask_price)
                    self._last_known_price[ticker] = price
                    return price
        except Exception as e:
            print(f"⚠️ 获取最新 Quote 失败: {e}")

        # 策略3: 获取最近一段时间的 Bar 数据
        try:
            end_time = datetime.now(timezone.utc)
            # 非交易时段可能需要回溯更长时间
            lookback = 60 if self._is_market_hours() else 1440  # 非交易时段回溯24小时
            start_time = end_time - timedelta(minutes=lookback)
            
            request_params = StockBarsRequest(
                symbol_or_symbols=[ticker],
                timeframe=TimeFrame.Minute,
                start=start_time.isoformat(),
                end=end_time.isoformat(),
                feed=self.data_feed
            )
            
            bars_response = self.data_client.get_stock_bars(request_params)
            bars_df = bars_response.df
            
            if not bars_df.empty:
                if ticker in bars_df.index.get_level_values(0):
                    ticker_bars = bars_df.loc[ticker]
                    price = float(ticker_bars.iloc[-1]['close'])
                else:
                    price = float(bars_df.iloc[-1]['close'])
                
                self._last_known_price[ticker] = price
                print(f"✅ [{timestamp_str}] 获取 {ticker} 历史价格: ${price:.2f}")
                return price
        except Exception as e:
            print(f"⚠️ 获取历史 Bar 失败: {e}")

        # 策略4: 使用缓存的最后已知价格
        if ticker in self._last_known_price:
            cached_price = self._last_known_price[ticker]
            print(f"⚠️ [{timestamp_str}] 使用 {ticker} 缓存价格: ${cached_price:.2f}")
            return cached_price

        # 所有策略都失败
        is_market_open = self._is_market_hours()
        if not is_market_open:
            print(f"⚠️ [{timestamp_str}] 当前为非交易时段，无法获取 {ticker} 实时价格")
        else:
            print(f"❌ [{timestamp_str}] 无法获取 {ticker} 的价格")
        
        return 0.0

    def _get_historical_price(self, ticker: str, target_time: datetime) -> float:
        """
        获取历史某个时间点的价格（用于回测）
        
        Args:
            ticker: 股票代码
            target_time: 目标时间
            
        Returns:
            float: 价格
        """
        timestamp_str = self._format_timestamp(target_time)
        
        # 先尝试精确时间
        start_time = target_time - timedelta(minutes=1)
        end_time = target_time

        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=[ticker],
                timeframe=TimeFrame.Minute,
                start=start_time.isoformat(),
                end=end_time.isoformat(),
                feed=self.data_feed
            )

            bars_response = self.data_client.get_stock_bars(request_params)
            bars_df = bars_response.df
            
            if not bars_df.empty:
                if ticker in bars_df.index.get_level_values(0):
                    ticker_bars = bars_df.loc[ticker]
                    return float(ticker_bars.iloc[-1]['close'])
                else:
                    return float(bars_df.iloc[-1]['close'])
        except Exception as e:
            print(f"⚠️ 精确时间获取失败: {e}")

        # 扩大搜索范围（前后5分钟）
        try:
            start_time = target_time - timedelta(minutes=5)
            end_time = target_time + timedelta(minutes=1)
            
            request_params = StockBarsRequest(
                symbol_or_symbols=[ticker],
                timeframe=TimeFrame.Minute,
                start=start_time.isoformat(),
                end=end_time.isoformat(),
                feed=self.data_feed
            )

            bars_response = self.data_client.get_stock_bars(request_params)
            bars_df = bars_response.df
            
            if not bars_df.empty:
                if ticker in bars_df.index.get_level_values(0):
                    ticker_bars = bars_df.loc[ticker]
                    # 找到最接近目标时间的 bar
                    return float(ticker_bars.iloc[-1]['close'])
                else:
                    return float(bars_df.iloc[-1]['close'])
        except Exception as e:
            print(f"⚠️ 扩展搜索也失败: {e}")

        print(f"⚠️ [{timestamp_str}] 未找到 {ticker} 在该时间点的数据")
        return 0.0

    # ==================== 账户与持仓 API ====================

    def get_account(self) -> Dict[str, Any]:
        """
        获取 Alpaca 账户信息。
        
        Returns:
            dict: 账户信息
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
            dict: 持仓信息，如果无持仓返回空字典
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
            print(f"   数量: {position_info['qty']:.0f} 股 ({position_info['side']})")
            print(f"   均价: ${position_info['avg_entry_price']:.2f}")
            print(f"   现价: ${position_info['current_price']:.2f}")
            print(f"   市值: ${position_info['market_value']:,.2f}")
            print(f"   盈亏: ${position_info['unrealized_pl']:,.2f} ({position_info['unrealized_plpc']*100:.2f}%)")
            
            return position_info
            
        except Exception as e:
            if "position does not exist" in str(e).lower():
                print(f"📊 [{timestamp_str}] {ticker} 无持仓")
                return {}
            print(f"❌ [{timestamp_str}] 获取 {ticker} 持仓信息失败: {e}")
            return {}

    def get_all_positions(self) -> List[Dict[str, Any]]:
        """
        获取所有持仓信息。
        
        Returns:
            list: 持仓列表
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
                pnl_str = f"+${pos['unrealized_pl']:.2f}" if pos['unrealized_pl'] >= 0 else f"-${abs(pos['unrealized_pl']):.2f}"
                print(f"   {pos['symbol']}: {pos['qty']:.0f} 股 @ ${pos['avg_entry_price']:.2f} | {pnl_str}")
            
            return position_list
            
        except Exception as e:
            print(f"❌ [{timestamp_str}] 获取所有持仓失败: {e}")
            return []

    def sync_position_status(self, ticker: str) -> Dict[str, Any]:
        """
        同步指定股票的仓位状态。
        
        Args:
            ticker: 股票代码
            
        Returns:
            dict: 同步后的状态
        """
        timestamp_str = self._format_timestamp(datetime.now(timezone.utc))
        
        account = self.get_account()
        if not account:
            return {}
        
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
    import pytz
    
    print("=" * 60)
    print("测试 AlpacaDataFetcher (使用 IEX 数据源)")
    print("=" * 60)
    
    fetcher = AlpacaDataFetcher(paper=True, data_feed=DataFeed.IEX)
    
    # 检查当前是否是交易时段
    et = pytz.timezone('America/New_York')
    now_et = datetime.now(et)
    is_market_open = fetcher._is_market_hours()
    
    print(f"\n当前美东时间: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"交易时段: {'是' if is_market_open else '否'}")
    
    print("\n--- 测试 get_latest_price (TSLA) ---")
    price = fetcher.get_latest_price("TSLA")
    if price > 0:
        print(f"✅ TSLA 最新价格: ${price:.2f}")
    else:
        print("❌ 无法获取价格")
    
    print("\n--- 测试 get_latest_bars (TSLA) ---")
    bars = fetcher.get_latest_bars("TSLA", lookback_minutes=120)
    if not bars.empty:
        print(f"获取到 {len(bars)} 根 K线")
        print(bars.tail(3))
    
    print("\n--- 测试 get_account ---")
    account = fetcher.get_account()
    
    print("\n--- 测试 get_all_positions ---")
    positions = fetcher.get_all_positions()