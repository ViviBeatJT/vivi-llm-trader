# src/engine/improved_backtest_engine.py

"""
改进的回测引擎 - 支持向策略传递持仓信息

主要改进：
1. 自动检测策略是否需要持仓信息（current_position, avg_cost）
2. 兼容旧策略（MeanReversionStrategy, GeminiStrategy）
3. 支持新激进策略（AggressiveM eanReversionStrategy）的止损机制
"""

from datetime import datetime, timezone, timedelta
from typing import Tuple, Optional
import pandas as pd
import inspect

from src.cache.trading_cache import TradingCache
from src.manager.position_manager import PositionManager
from src.data_fetcher.alpaca_data_fetcher import AlpacaDataFetcher
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from src.strategies.base_strategy import BaseStrategy


class ImprovedBacktestEngine:
    """
    改进的回测引擎 - 协调数据获取、策略执行和仓位管理。
    
    新特性：
    - 自动检测策略签名，智能传递参数
    - 支持需要持仓信息的策略（如止损策略）
    - 向后兼容旧策略
    
    职责分离：
    - DataFetcher: 获取市场数据
    - Strategy: 分析数据，生成信号（可选接收持仓信息）
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
            strategy: 策略实例
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
        
        # 检测策略是否支持持仓信息
        self._strategy_supports_position_info = self._check_strategy_signature()

    def _check_strategy_signature(self) -> bool:
        """
        检查策略的 get_signal 方法是否支持持仓信息参数。
        
        Returns:
            bool: True 如果支持 current_position 和 avg_cost 参数
        """
        try:
            sig = inspect.signature(self.strategy.get_signal)
            params = sig.parameters
            
            has_position = 'current_position' in params
            has_avg_cost = 'avg_cost' in params
            
            if has_position and has_avg_cost:
                print(f"✅ 策略 {self.strategy} 支持持仓信息（止损功能）")
                return True
            else:
                print(f"ℹ️ 策略 {self.strategy} 不需要持仓信息（标准模式）")
                return False
        except Exception as e:
            print(f"⚠️ 无法检测策略签名: {e}")
            return False

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
        current_price = 0.0
        
        initial_status = self.position_manager.get_account_status(current_price=0.0)
        print(f"📈 回测开始: {self.start_dt} → {self.end_dt}")
        print(f"   初始资金: ${initial_status['cash']:,.2f}")
        print(f"   策略: {self.strategy}")
        print(f"   K线周期: {self.timeframe.amount} {self.timeframe.unit.name}")
        print(f"   步进间隔: {self.step_minutes} 分钟")
        print("-" * 50)
        
        while current_time <= self.end_dt:
            # 确保时区
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=timezone.utc)
            
            # 1. 获取数据
            market_data, current_price = self._fetch_data(current_time)
            
            if market_data.empty or current_price <= 0:
                print(f"⚠️ {current_time.strftime('%m-%d %H:%M')}: 无市场数据，跳过")
                current_time += timedelta(minutes=self.step_minutes)
                continue

            # 2. 获取当前持仓状态（用于支持需要持仓信息的策略）
            account_status = self.position_manager.get_account_status(current_price)
            current_position = account_status.get('position', 0.0)
            avg_cost = account_status.get('avg_cost', 0.0)
            
            # 3. 调用策略获取信号
            try:
                if self._strategy_supports_position_info:
                    # 新策略：传递持仓信息
                    signal_data, strategy_price = self.strategy.get_signal(
                        ticker=self.ticker,
                        new_data=market_data,
                        current_position=current_position,
                        avg_cost=avg_cost,
                        verbose=False
                    )
                else:
                    # 旧策略：不传递持仓信息
                    signal_data, strategy_price = self.strategy.get_signal(
                        ticker=self.ticker,
                        new_data=market_data,
                        verbose=False
                    )
                
                signal = signal_data.get('signal', 'HOLD')
                confidence = signal_data.get('confidence_score', 0)
                reason = signal_data.get('reason', '')
                
                # 优先使用策略返回的价格
                if strategy_price > 0:
                    current_price = strategy_price

            except Exception as e:
                print(f"❌ 策略错误 @ {current_time}: {e}")
                signal = "HOLD"
                confidence = 0
                reason = f"Error: {e}"

            # 4. 执行交易
            if signal in ["BUY", "SELL", "SHORT", "COVER"]:
                signal_emoji = {"BUY": "🟢", "SELL": "🔴", "SHORT": "🔻", "COVER": "🔺"}.get(signal, "⚪")
                
                # 显示持仓状态
                if current_position > 0:
                    pos_str = f"多仓{current_position:.0f}股@${avg_cost:.2f}"
                elif current_position < 0:
                    pos_str = f"空仓{abs(current_position):.0f}股@${avg_cost:.2f}"
                else:
                    pos_str = "无仓位"
                
                print(f"{signal_emoji} {current_time.strftime('%m-%d %H:%M')} | {signal} | "
                      f"${current_price:.2f} | {pos_str} | 置信度: {confidence}")
                print(f"   原因: {reason}")
                
                trade_result = self.position_manager.execute_and_update(
                    timestamp=current_time,
                    signal=signal,
                    current_price=current_price,
                    ticker=self.ticker
                )
                
                results.append({
                    'timestamp': current_time,
                    'signal': signal,
                    'confidence': confidence,
                    'price': current_price,
                    'executed': trade_result,
                    'reason': reason,
                    'position_before': current_position,
                    'avg_cost_before': avg_cost
                })
            
            current_time += timedelta(minutes=self.step_minutes)

        # 汇总结果
        final_status = self.position_manager.get_account_status(current_price=current_price)
        final_equity = final_status['equity']
        trade_log_df = self.position_manager.get_trade_log()
        
        print("-" * 50)
        print(f"✅ 回测完成")
        print(f"   总信号数: {len([r for r in results if r['signal'] != 'HOLD'])}")
        print(f"   最终权益: ${final_equity:,.2f}")
        
        return final_equity, trade_log_df