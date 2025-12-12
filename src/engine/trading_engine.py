# src/engine/trading_engine.py

"""
TradingEngine - Core trading logic shared between backtest and live trading.

This engine encapsulates:
1. Data fetching and management
2. Strategy signal generation
3. Position management and trade execution
4. Time-based controls (market hours, force close)
5. Progress tracking and reporting

Usage:
    # For backtest
    engine = TradingEngine(config, mode='backtest')
    result = engine.run(start_time, end_time)
    
    # For live trading
    engine = TradingEngine(config, mode='live')
    result = engine.run()
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta, time as dt_time
from typing import Dict, Any, Optional, Callable, List, Literal
import pandas as pd
import pytz

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


# ==========================================
# Configuration Classes
# ==========================================

@dataclass
class TimeConfig:
    """Trading time configuration."""
    last_entry_time: dt_time = dt_time(15, 50)   # No new positions after this
    force_close_time: dt_time = dt_time(15, 55)  # Force close all positions
    market_close_time: dt_time = dt_time(16, 0)  # Market close
    market_open_time: dt_time = dt_time(9, 30)   # Market open
    timezone: str = 'America/New_York'
    
    def __post_init__(self):
        self.tz = pytz.timezone(self.timezone)


@dataclass
class FinanceConfig:
    """Financial parameters configuration."""
    initial_capital: float = 1000.0
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0001
    min_lot_size: int = 1
    max_allocation: float = 0.95
    stamp_duty_rate: float = 0.001
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for PositionManager."""
        return {
            'INITIAL_CAPITAL': self.initial_capital,
            'COMMISSION_RATE': self.commission_rate,
            'SLIPPAGE_RATE': self.slippage_rate,
            'MIN_LOT_SIZE': self.min_lot_size,
            'MAX_ALLOCATION': self.max_allocation,
            'STAMP_DUTY_RATE': self.stamp_duty_rate,
        }


@dataclass
class DataConfig:
    """Data fetching configuration."""
    lookback_minutes: int = 300
    timeframe: TimeFrame = field(default_factory=lambda: TimeFrame(5, TimeFrameUnit.Minute))
    step_seconds: int = 30  # For backtest stepping


@dataclass
class EngineConfig:
    """Complete engine configuration."""
    ticker: str
    strategy: Any  # Strategy instance
    position_manager: Any  # PositionManager instance
    data_fetcher: Any  # DataFetcher instance
    executor: Any  # Executor instance
    
    time_config: TimeConfig = field(default_factory=TimeConfig)
    finance_config: FinanceConfig = field(default_factory=FinanceConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    
    # Optional components
    cache: Any = None
    visualizer: Any = None
    
    # Callbacks
    on_signal_callback: Optional[Callable] = None
    on_trade_callback: Optional[Callable] = None
    on_iteration_callback: Optional[Callable] = None
    
    # Behavior flags
    respect_market_hours: bool = True
    verbose: bool = True


# ==========================================
# TradingEngine
# ==========================================

class TradingEngine:
    """
    Unified trading engine for both backtest and live trading.
    
    The engine handles:
    - Data fetching and preprocessing
    - Strategy signal generation
    - Trade execution via PositionManager
    - Time-based controls (market hours, force close)
    - Progress tracking and reporting
    
    Differences between modes:
    - Backtest: Steps through historical time, uses end_dt for data fetching
    - Live: Uses real-time data, respects actual market hours
    """
    
    def __init__(self, config: EngineConfig, mode: Literal['backtest', 'live'] = 'backtest'):
        """
        Initialize the trading engine.
        
        Args:
            config: Engine configuration
            mode: 'backtest' or 'live'
        """
        self.config = config
        self.mode = mode
        
        # Core components
        self.ticker = config.ticker
        self.strategy = config.strategy
        self.position_manager = config.position_manager
        self.data_fetcher = config.data_fetcher
        self.executor = config.executor
        
        # Configuration
        self.time_config = config.time_config
        self.finance_config = config.finance_config
        self.data_config = config.data_config
        
        # Optional components
        self.cache = config.cache
        self.visualizer = config.visualizer
        
        # Callbacks
        self.on_signal_callback = config.on_signal_callback
        self.on_trade_callback = config.on_trade_callback
        self.on_iteration_callback = config.on_iteration_callback
        
        # State tracking
        self._running = False
        self._iteration = 0
        self._signals_count = 0
        self._trades_count = 0
        self._start_time: Optional[datetime] = None
        self._last_entry_reached = False
        self._force_close_reached = False
        
        # Results
        self._equity_history: List[Dict] = []
        
        print(f"🔧 TradingEngine initialized:")
        print(f"   Mode: {mode.upper()}")
        print(f"   Ticker: {self.ticker}")
        print(f"   Strategy: {type(self.strategy).__name__}")
        print(f"   Initial Capital: ${self.finance_config.initial_capital:,.2f}")
    
    # ==========================================
    # Time Utilities
    # ==========================================
    
    def _to_eastern(self, dt: datetime) -> datetime:
        """Convert datetime to Eastern time."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(self.time_config.tz)
    
    def _is_market_hours(self, dt: datetime) -> bool:
        """Check if time is within market hours."""
        et = self._to_eastern(dt)
        current_time = et.time()
        
        # Check if it's a weekday
        if et.weekday() >= 5:
            return False
        
        return self.time_config.market_open_time <= current_time < self.time_config.market_close_time
    
    def _is_in_no_entry_window(self, dt: datetime) -> bool:
        """Check if in no-new-entry window (e.g., 15:50+)."""
        et = self._to_eastern(dt)
        return et.time() >= self.time_config.last_entry_time
    
    def _is_force_close_time(self, dt: datetime) -> bool:
        """Check if it's force close time (e.g., 15:55+)."""
        et = self._to_eastern(dt)
        return et.time() >= self.time_config.force_close_time
    
    def _format_time_et(self, dt: datetime) -> str:
        """Format datetime as ET string."""
        et = self._to_eastern(dt)
        return et.strftime('%H:%M:%S ET')
    
    # ==========================================
    # Data Management
    # ==========================================
    
    def _fetch_data(self, end_dt: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch market data.
        
        Args:
            end_dt: End time for data fetch (for backtest)
            
        Returns:
            DataFrame with OHLCV data
        """
        if self.data_fetcher is None:
            return pd.DataFrame()
        
        return self.data_fetcher.get_latest_bars(
            ticker=self.ticker,
            lookback_minutes=self.data_config.lookback_minutes,
            end_dt=end_dt,
            timeframe=self.data_config.timeframe
        )
    
    # ==========================================
    # Signal & Trade Execution
    # ==========================================
    
    def _process_iteration(self, current_time: datetime) -> Dict[str, Any]:
        """
        Process a single iteration (one time step).
        
        Args:
            current_time: Current timestamp (UTC)
            
        Returns:
            Iteration result dictionary
        """
        self._iteration += 1
        
        # Fetch data
        df = self._fetch_data(end_dt=current_time if self.mode == 'backtest' else None)
        
        if df.empty:
            return {'status': 'no_data', 'iteration': self._iteration}
        
        current_price = df.iloc[-1]['close']
        current_et = self._to_eastern(current_time)
        
        # Get account status
        account_status = self.position_manager.get_account_status(current_price)
        current_position = account_status.get('position', 0.0)
        avg_cost = account_status.get('avg_cost', 0.0)
        current_equity = account_status.get('equity', self.finance_config.initial_capital)
        
        # Check time windows
        is_force_close = self._is_force_close_time(current_time)
        is_no_entry = self._is_in_no_entry_window(current_time)
        
        # Log time window transitions
        if not self._last_entry_reached and is_no_entry:
            print(f"\n⏰ Reached last entry time: {self._format_time_et(current_time)}")
            self._last_entry_reached = True
        
        if not self._force_close_reached and is_force_close:
            print(f"\n🔔 Reached force close time: {self._format_time_et(current_time)}")
            self._force_close_reached = True
        
        # Get strategy signal
        try:
            signal_data, _ = self.strategy.get_signal(
                ticker=self.ticker,
                new_data=df,
                current_position=current_position,
                avg_cost=avg_cost,
                verbose=self.config.verbose,
                is_market_close=is_force_close,
                current_time_et=current_et
            )
            
            signal = signal_data.get('signal', 'HOLD')
            
        except Exception as e:
            print(f"❌ Strategy error: {e}")
            return {'status': 'error', 'error': str(e), 'iteration': self._iteration}
        
        # Execute trade if signal is actionable
        trade_executed = False
        if signal in ['BUY', 'SELL', 'SHORT', 'COVER']:
            self._signals_count += 1
            
            # Log signal
            emoji = {"BUY": "🟢", "SELL": "🔴", "SHORT": "🔻", "COVER": "🔺"}
            print(f"\n{emoji.get(signal, '⚪')} {self._format_time_et(current_time)} | "
                  f"{signal} @ ${current_price:.2f}")
            print(f"   {signal_data.get('reason', 'N/A')}")
            
            # Execute
            success = self.position_manager.execute_and_update(
                timestamp=current_time,
                signal=signal,
                current_price=current_price,
                ticker=self.ticker
            )
            
            if success:
                self._trades_count += 1
                trade_executed = True
                
                # Trade callback
                if self.on_trade_callback:
                    self.on_trade_callback(signal, current_price, current_time)
            
            # Signal callback
            if self.on_signal_callback:
                self.on_signal_callback(signal_data, current_price, current_time)
        
        # Update visualizer if available
        if self.visualizer:
            strategy_df = self.strategy.get_history_data(self.ticker)
            trade_log = self.position_manager.get_trade_log()
            
            if not strategy_df.empty:
                self.visualizer.update_data(
                    market_data=strategy_df,
                    trade_log=trade_log,
                    current_equity=current_equity,
                    current_position=current_position,
                    timestamp=current_time
                )
        
        # Record equity
        self._equity_history.append({
            'timestamp': current_time,
            'equity': current_equity,
            'position': current_position,
            'price': current_price
        })
        
        # Iteration callback
        if self.on_iteration_callback:
            self.on_iteration_callback(self._iteration, current_time, current_equity)
        
        return {
            'status': 'ok',
            'iteration': self._iteration,
            'signal': signal,
            'trade_executed': trade_executed,
            'price': current_price,
            'equity': current_equity,
            'position': current_position
        }
    
    # ==========================================
    # Final Position Check
    # ==========================================
    
    def _ensure_flat_position(self, end_time: datetime) -> bool:
        """
        Ensure all positions are closed at end of session.
        
        Args:
            end_time: Session end time
            
        Returns:
            True if positions are now flat
        """
        # Get final data
        df = self._fetch_data(end_dt=end_time if self.mode == 'backtest' else None)
        final_price = df.iloc[-1]['close'] if not df.empty else 0.0
        
        # Check position
        final_status = self.position_manager.get_account_status(final_price)
        final_position = final_status.get('position', 0.0)
        
        print(f"\n{'='*60}")
        print(f"🔍 Final Position Check")
        print(f"{'='*60}")
        print(f"   Time: {self._format_time_et(end_time)}")
        print(f"   Price: ${final_price:.2f}")
        print(f"   Position: {final_position:.0f} shares")
        
        if final_position != 0:
            print(f"\n⚠️ Open position detected! Forcing close...")
            
            close_signal = 'SELL' if final_position > 0 else 'COVER'
            
            try:
                self.position_manager.execute_and_update(
                    timestamp=end_time,
                    signal=close_signal,
                    current_price=final_price,
                    ticker=self.ticker
                )
                
                # Verify
                final_status = self.position_manager.get_account_status(final_price)
                final_position = final_status.get('position', 0.0)
                
                if final_position == 0:
                    print(f"   ✅ Position closed successfully")
                    return True
                else:
                    print(f"   ❌ Warning: Position still open ({final_position} shares)")
                    return False
                    
            except Exception as e:
                print(f"   ❌ Force close failed: {e}")
                return False
        else:
            print(f"   ✅ Position is flat")
            return True
    
    # ==========================================
    # Main Run Methods
    # ==========================================
    
    def run_backtest(self, 
                     start_time: datetime, 
                     end_time: datetime,
                     progress_interval: int = 10) -> Dict[str, Any]:
        """
        Run backtest from start_time to end_time.
        
        Args:
            start_time: Backtest start time (UTC)
            end_time: Backtest end time (UTC)
            progress_interval: Print progress every N iterations
            
        Returns:
            Backtest results dictionary
        """
        self._running = True
        self._start_time = datetime.now(timezone.utc)
        self._iteration = 0
        self._signals_count = 0
        self._trades_count = 0
        self._equity_history = []
        self._last_entry_reached = False
        self._force_close_reached = False
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting Backtest")
        print(f"{'='*60}")
        print(f"   Start: {self._format_time_et(start_time)}")
        print(f"   End: {self._format_time_et(end_time)}")
        print(f"   Step: {self.data_config.step_seconds} seconds")
        
        current_time = start_time
        step = timedelta(seconds=self.data_config.step_seconds)
        total_steps = int((end_time - start_time).total_seconds() / self.data_config.step_seconds)
        
        try:
            while current_time <= end_time and self._running:
                # Ensure timezone
                if current_time.tzinfo is None:
                    current_time = current_time.replace(tzinfo=timezone.utc)
                
                # Process iteration
                result = self._process_iteration(current_time)
                
                # Progress update
                if self._iteration % progress_interval == 0:
                    progress = (current_time - start_time) / (end_time - start_time) * 100
                    current_et = self._to_eastern(current_time)
                    equity = result.get('equity', 0)
                    position = result.get('position', 0)
                    print(f"\n📊 Progress: {progress:.1f}% | Time: {current_et.strftime('%H:%M')} ET | "
                          f"Equity: ${equity:,.0f} | Position: {position:.0f}")
                
                # Advance time
                current_time += step
                
        except KeyboardInterrupt:
            print("\n⚠️ Backtest interrupted by user")
        
        # Ensure flat position at end
        self._ensure_flat_position(end_time)
        
        return self._generate_report(end_time)
    
    def run_live(self, 
                 max_runtime_minutes: Optional[int] = None,
                 interval_seconds: Optional[int] = None) -> Dict[str, Any]:
        """
        Run live trading.
        
        Args:
            max_runtime_minutes: Maximum runtime in minutes (None = unlimited)
            interval_seconds: Override default step interval
            
        Returns:
            Trading results dictionary
        """
        import time as time_module
        
        self._running = True
        self._start_time = datetime.now(timezone.utc)
        self._iteration = 0
        self._signals_count = 0
        self._trades_count = 0
        self._equity_history = []
        self._last_entry_reached = False
        self._force_close_reached = False
        
        interval = interval_seconds or self.data_config.step_seconds
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting Live Trading")
        print(f"{'='*60}")
        print(f"   Ticker: {self.ticker}")
        print(f"   Interval: {interval} seconds")
        if max_runtime_minutes:
            print(f"   Max Runtime: {max_runtime_minutes} minutes")
        
        try:
            while self._running:
                current_time = datetime.now(timezone.utc)
                
                # Check max runtime
                if max_runtime_minutes:
                    runtime = (current_time - self._start_time).total_seconds() / 60
                    if runtime >= max_runtime_minutes:
                        print(f"\n⏰ Max runtime reached ({max_runtime_minutes} minutes)")
                        break
                
                # Check market hours
                if self.config.respect_market_hours:
                    if not self._is_market_hours(current_time):
                        current_et = self._to_eastern(current_time)
                        
                        # Check if market closed for the day
                        if current_et.time() >= self.time_config.market_close_time:
                            print(f"\n🔔 Market closed for today")
                            break
                        
                        # Wait for market open
                        print(f"⏳ [{current_et.strftime('%H:%M:%S')}] Waiting for market hours...")
                        time_module.sleep(60)
                        continue
                
                # Process iteration
                self._process_iteration(current_time)
                
                # Wait for next iteration
                time_module.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Live trading interrupted by user")
        
        # Final check
        end_time = datetime.now(timezone.utc)
        self._ensure_flat_position(end_time)
        
        return self._generate_report(end_time)
    
    def run(self, 
            start_time: Optional[datetime] = None, 
            end_time: Optional[datetime] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Main entry point - runs in appropriate mode.
        
        Args:
            start_time: Start time (required for backtest)
            end_time: End time (required for backtest)
            **kwargs: Additional arguments passed to run method
            
        Returns:
            Results dictionary
        """
        if self.mode == 'backtest':
            if start_time is None or end_time is None:
                raise ValueError("start_time and end_time required for backtest mode")
            return self.run_backtest(start_time, end_time, **kwargs)
        else:
            return self.run_live(**kwargs)
    
    def stop(self):
        """Stop the engine gracefully."""
        print("\n🛑 Stopping engine...")
        self._running = False
    
    # ==========================================
    # Reporting
    # ==========================================
    
    def _generate_report(self, end_time: datetime) -> Dict[str, Any]:
        """Generate final report."""
        
        # Get final status
        df = self._fetch_data(end_dt=end_time if self.mode == 'backtest' else None)
        final_price = df.iloc[-1]['close'] if not df.empty else 0.0
        final_status = self.position_manager.get_account_status(final_price)
        
        runtime_seconds = (datetime.now(timezone.utc) - self._start_time).total_seconds() if self._start_time else 0
        
        trade_log = self.position_manager.get_trade_log()
        
        report = {
            'mode': self.mode,
            'ticker': self.ticker,
            'strategy': type(self.strategy).__name__,
            'runtime_seconds': runtime_seconds,
            'iterations': self._iteration,
            'signals': self._signals_count,
            'trades_executed': self._trades_count,
            'initial_capital': self.finance_config.initial_capital,
            'final_equity': final_status.get('equity', 0),
            'final_position': final_status.get('position', 0),
            'final_cash': final_status.get('cash', 0),
            'pnl': final_status.get('equity', 0) - self.finance_config.initial_capital,
            'pnl_pct': ((final_status.get('equity', 0) - self.finance_config.initial_capital) / 
                       self.finance_config.initial_capital * 100) if self.finance_config.initial_capital > 0 else 0,
            'force_close_time_reached': self._force_close_reached,
            'last_entry_time_reached': self._last_entry_reached,
            'trade_log': trade_log,
            'equity_history': self._equity_history
        }
        
        # Calculate win rate
        if trade_log is not None and not trade_log.empty:
            completed_trades = trade_log[trade_log['type'].isin(['SELL', 'COVER'])]
            if not completed_trades.empty and 'net_pnl' in completed_trades.columns:
                winning_trades = completed_trades[completed_trades['net_pnl'] > 0]
                report['completed_trades'] = len(completed_trades)
                report['winning_trades'] = len(winning_trades)
                report['win_rate'] = len(winning_trades) / len(completed_trades) if len(completed_trades) > 0 else 0
        
        return report
    
    def print_report(self, report: Optional[Dict] = None):
        """Print formatted report."""
        if report is None:
            report = self._generate_report(datetime.now(timezone.utc))
        
        print(f"\n{'='*60}")
        print(f"📊 Trading Report - {report['mode'].upper()}")
        print(f"{'='*60}")
        print(f"   Ticker: {report['ticker']}")
        print(f"   Strategy: {report['strategy']}")
        print(f"   Runtime: {report['runtime_seconds'] / 60:.1f} minutes")
        print(f"   Iterations: {report['iterations']}")
        print(f"   Signals: {report['signals']}")
        print(f"   Trades: {report['trades_executed']}")
        
        print(f"\n💰 Performance:")
        print(f"   Initial: ${report['initial_capital']:,.2f}")
        print(f"   Final: ${report['final_equity']:,.2f}")
        print(f"   PnL: ${report['pnl']:,.2f} ({report['pnl_pct']:+.2f}%)")
        print(f"   Position: {report['final_position']:.0f} shares {'✅' if report['final_position'] == 0 else '⚠️'}")
        
        if 'win_rate' in report:
            print(f"\n📈 Trade Statistics:")
            print(f"   Completed: {report['completed_trades']}")
            print(f"   Winners: {report['winning_trades']}")
            print(f"   Win Rate: {report['win_rate']*100:.1f}%")
        
        print(f"\n⏰ Time Controls:")
        print(f"   Last Entry Reached: {'✅' if report['last_entry_time_reached'] else '❌'}")
        print(f"   Force Close Reached: {'✅' if report['force_close_time_reached'] else '❌'}")
        
        print(f"{'='*60}")