from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Any
import pandas as pd
from src.cache.trading_cache import TradingCache
from src.manager.position_manager import PositionManager
from src.data_fetcher.alpaca_data_fetcher import AlpacaDataFetcher
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from src.strategies.base_strategy import BaseStrategy

class BacktestEngine:
    """
    Class-based Backtest Engine.
    Orchestrates the interaction between Data, Strategy, and Execution.
    The engine is now strictly for backtesting historical data.
    """

    def __init__(self, 
                 ticker: str, 
                 start_dt: datetime, 
                 end_dt: datetime, 
                 strategy: BaseStrategy, 
                 position_manager: PositionManager, 
                 data_fetcher: AlpacaDataFetcher, 
                 cache: TradingCache,
                 step_minutes: int = 5):
        """
        Initialize the Backtest engine.

        Args:
            ticker: Stock symbol to trade.
            start_dt: Start datetime for the backtest.
            end_dt: End datetime for the backtest.
            strategy: Strategy object (must have a get_signal method).
            position_manager: Initialized PositionManager for execution and state tracking.
            data_fetcher: Data fetcher for retrieving market data.
            cache: Cache object for storing/retrieving AI analysis or data.
            step_minutes: Time step for the simulation loop.
        """
        self.ticker = ticker
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.strategy = strategy
        self.position_manager = position_manager
        self.data_fetcher = data_fetcher
        self.cache = cache
        self.step_minutes = step_minutes

    def _get_current_price(self, current_time: datetime) -> float:
        """
        Helper to get the price at a specific time for backtesting.
        Fetches historical bar data.
        """
        # For backtest, we need the price at 'current_time'
        # We fetch a small window ending at current_time
        df = self.data_fetcher.get_latest_bars(
            ticker=self.ticker,
            lookback_minutes=15, # Look back a bit to ensure we find a bar
            end_dt=current_time,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute) # Granular data for price
        )
        
        if not df.empty:
            # Return the close of the most recent bar relative to current_time
            return df.iloc[-1]['close']
        return 0.0

    def run(self) -> Tuple[float, pd.DataFrame]:
        """
        Execute the backtest loop.
        
        Returns:
            Tuple[float, pd.DataFrame]: Final equity and the trade log.
        """
        current_time = self.start_dt
        results = []
        
        initial_status = self.position_manager.get_account_status(current_price=0.0)
        print(f"📈 Engine Started: {self.start_dt} to {self.end_dt} | Initial Cash: ${initial_status['cash']:,.2f}")
        
        while current_time <= self.end_dt:
            # Ensure timezone awareness
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=timezone.utc)
            
            # 1. Get Price at this moment
            current_price = self._get_current_price(current_time)
            
            if current_price <= 0:
                print(f"⚠️ No price data for {current_time}, skipping step.")
                current_time += timedelta(minutes=self.step_minutes)
                continue

            # 2. Update Position Manager State (Mark-to-Market)
            current_status = self.position_manager.get_account_status(current_price=current_price)
            
            # 3. Get Signal from Strategy
            try:
                signal_data, analysis_price = self.strategy.get_signal(
                    ticker=self.ticker,
                    end_dt=current_time, # Context time
                    lookback_minutes=120 # Standard lookback
                )
                
                signal = signal_data.get('signal', 'HOLD')
                confidence = signal_data.get('confidence_score', 0)
                reason = signal_data.get('reason', '')
                
                # Use the price from strategy if it's more relevant, otherwise use our fetched price
                if analysis_price > 0:
                    current_price = analysis_price

            except Exception as e:
                print(f"❌ Strategy Error at {current_time}: {e}")
                signal = "HOLD"
                confidence = 0
                reason = f"Error: {e}"

            # 4. Execute Trade (if any) via PositionManager
            if signal in ["BUY", "SELL"]:
                print(f"🔥 Signal: {signal} | Price: ${current_price:.2f} | Conf: {confidence} | {reason[:50]}...")
                
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
            else:
                # Log HOLDs if needed, or just print
                pass
            
            current_time += timedelta(minutes=self.step_minutes)

        # --- Summary ---
        final_status = self.position_manager.get_account_status(current_price=current_price)
        final_equity = final_status['equity']
        trade_log_df = self.position_manager.get_trade_log()
        
        print("\n--- ✅ Backtest Complete ---")
        print(f"Total Signals: {len(results)}")
        print(f"Final Equity: ${final_equity:,.2f}")
        
        return final_equity, trade_log_df