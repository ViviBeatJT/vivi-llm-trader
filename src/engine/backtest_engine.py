# src/backtest/backtest_engine.py

from datetime import datetime, timezone, timedelta
from typing import Tuple, Optional
import pandas as pd
from src.cache.trading_cache import TradingCache
from src.manager.position_manager import PositionManager
from src.data_fetcher.alpaca_data_fetcher import AlpacaDataFetcher
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from src.strategies.base_strategy import BaseStrategy


class BacktestEngine:
    """
    回测引擎 - 协调数据获取、策略执行和仓位管理。
    
    职责分离：
    - DataFetcher: 获取市场数据
    - Strategy: 分析数据，生成信号（不负责获取数据）
    - PositionManager: 执行交易，管理仓位
    - BacktestEngine: 协调以上组件
    """

    def __init__(self, 
                 ticker: str, 
                 start_dt: datetime, 
                 end_dt: datetime, 
                 strategy: BaseStrategy, 
                 position_manager: PositionManager, 
                 data_fetcher: AlpacaDataFetcher, 
                 cache: TradingCache,
                 step_minutes: int = 5,
                 lookback_minutes: int = 120,
                 timeframe: Optional[TimeFrame] = None):
        """
        初始化回测引擎。

        Args:
            ticker: 股票代码
            start_dt: 回测开始时间
            end_dt: 回测结束时间
            strategy: 策略实例（只负责生成信号）
            position_manager: 仓位管理器
            data_fetcher: 数据获取器
            cache: 缓存对象
            step_minutes: 模拟步进间隔（分钟）
            lookback_minutes: 每次获取数据的回溯时间（分钟）
            timeframe: K线时间框架（默认为5分钟）
        """
        self.ticker = ticker
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.strategy = strategy
        self.position_manager = position_manager
        self.data_fetcher = data_fetcher
        self.cache = cache
        self.step_minutes = step_minutes
        self.lookback_minutes = lookback_minutes
        self.timeframe = timeframe or TimeFrame(5, TimeFrameUnit.Minute)

    def _fetch_data(self, current_time: datetime) -> Tuple[pd.DataFrame, float]:
        """
        获取指定时间点的市场数据和当前价格。
        
        Args:
            current_time: 当前模拟时间
            
        Returns:
            Tuple[pd.DataFrame, float]: (OHLCV 数据, 最新价格)
        """
        df = self.data_fetcher.get_latest_bars(
            ticker=self.ticker,
            lookback_minutes=self.lookback_minutes,
            end_dt=current_time,
            timeframe=self.timeframe
        )
        
        # 从获取的数据中提取最新价格
        if not df.empty:
            current_price = df.iloc[-1]['close']
        else:
            current_price = 0.0
        
        return df, current_price

    def run(self) -> Tuple[float, pd.DataFrame]:
        """
        执行回测循环。
        
        Returns:
            Tuple[final_equity, trade_log_df]
        """
        current_time = self.start_dt
        results = []
        current_price = 0.0  # 用于最后计算权益
        
        initial_status = self.position_manager.get_account_status(current_price=0.0)
        print(f"📈 回测开始: {self.start_dt} → {self.end_dt}")
        print(f"   初始资金: ${initial_status['cash']:,.2f}")
        print(f"   策略: {self.strategy}")
        print(f"   K线周期: {self.timeframe.amount} {self.timeframe.unit.name}")
        print("-" * 50)
        
        while current_time <= self.end_dt:
            # 确保时区
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=timezone.utc)
            
            # 1. 获取数据（一次性获取，同时得到 DataFrame 和当前价格）
            market_data, current_price = self._fetch_data(current_time)
            
            if market_data.empty or current_price <= 0:
                print(f"⚠️ {current_time.strftime('%m-%d %H:%M')}: 无市场数据，跳过")
                current_time += timedelta(minutes=self.step_minutes)
                continue

            # 2. 调用策略获取信号
            try:
                signal_data, strategy_price = self.strategy.get_signal(
                    ticker=self.ticker,
                    new_data=market_data,
                    verbose=False
                )
                
                signal = signal_data.get('signal', 'HOLD')
                confidence = signal_data.get('confidence_score', 0)
                reason = signal_data.get('reason', '')
                
                # 优先使用策略返回的价格（如果有效）
                if strategy_price > 0:
                    current_price = strategy_price

            except Exception as e:
                print(f"❌ 策略错误 @ {current_time}: {e}")
                signal = "HOLD"
                confidence = 0
                reason = f"Error: {e}"

            # 3. 执行交易
            if signal in ["BUY", "SELL", "SHORT", "COVER"]:
                signal_emoji = {"BUY": "🟢", "SELL": "🔴", "SHORT": "🔻", "COVER": "🔺"}.get(signal, "⚪")
                print(f"{signal_emoji} {current_time.strftime('%m-%d %H:%M')} | {signal} | "
                      f"${current_price:.2f} | 置信度: {confidence}")
                
                trade_result = self.position_manager.execute_and_update(
                    timestamp=current_time,
                    signal=signal,
                    current_price=current_price
                )
                
                results.append({
                    'timestamp': current_time,
                    'signal': signal,
                    'confidence': confidence,
                    'price': current_price,
                    'executed': trade_result,
                    'reason': reason
                })
            
            current_time += timedelta(minutes=self.step_minutes)

        # 汇总结果
        final_status = self.position_manager.get_account_status(current_price=current_price)
        final_equity = final_status['equity']
        trade_log_df = self.position_manager.get_trade_log()
        
        print("-" * 50)
        print(f"✅ 回测完成")
        print(f"   总信号数: {len(results)}")
        print(f"   最终权益: ${final_equity:,.2f}")
        
        return final_equity, trade_log_df