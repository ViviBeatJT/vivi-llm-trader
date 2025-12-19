# src/engine/trading_engine.py

"""
TradingEngine - Core trading logic shared between backtest and live trading.

使用集中配置系统，不再定义重复的配置类。

Usage:
    from src.config.trading_config import get_full_config
    from src.engine.trading_engine import TradingEngine
    
    config = get_full_config(ticker='TSLA', mode='paper')
    engine = TradingEngine.from_config(config)
    result = engine.run()
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta, time as dt_time
from typing import Dict, Any, Optional, Callable, List, Literal, TYPE_CHECKING
import pandas as pd
import pytz

if TYPE_CHECKING:
    from src.config.trading_config import TradingConfig


# ==========================================
# Engine Config (组件容器，不是参数配置)
# ==========================================

@dataclass
class EngineComponents:
    """
    Engine 运行所需的组件容器

    注意：这不是参数配置，只是组件的引用。
    所有参数配置都在 TradingConfig 中。
    """
    ticker: str
    strategy: Any
    position_manager: Any
    data_fetcher: Any
    executor: Any

    # Optional components
    cache: Any = None
    visualizer: Any = None

    # Callbacks
    on_signal_callback: Optional[Callable] = None
    on_trade_callback: Optional[Callable] = None
    on_iteration_callback: Optional[Callable] = None


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
    """

    def __init__(self,
                 components: EngineComponents,
                 config: 'TradingConfig',
                 mode: Literal['backtest', 'live'] = 'backtest'):
        """
        Initialize the trading engine.

        Args:
            components: Engine components (strategy, position_manager, etc.)
            config: TradingConfig configuration object
            mode: 'backtest' or 'live'
        """
        self.components = components
        self.config = config
        self.mode = mode

        # Shortcuts to components
        self.ticker = components.ticker
        self.strategy = components.strategy
        self.position_manager = components.position_manager
        self.data_fetcher = components.data_fetcher
        self.executor = components.executor
        self.cache = components.cache
        self.visualizer = components.visualizer

        # Callbacks
        self.on_signal_callback = components.on_signal_callback
        self.on_trade_callback = components.on_trade_callback
        self.on_iteration_callback = components.on_iteration_callback

        # Timezone
        self._tz = pytz.timezone(config.time.timezone)

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
        print(f"   Initial Capital: ${config.finance.initial_capital:,.2f}")

    @classmethod
    def from_config(cls,
                    config: 'TradingConfig',
                    components: Optional[EngineComponents] = None,
                    mode: Optional[str] = None) -> 'TradingEngine':
        """
        从 TradingConfig 创建 Engine

        Args:
            config: TradingConfig 配置对象
            components: 可选的预创建组件
            mode: 覆盖 config 中的 mode

        Returns:
            TradingEngine instance
        """
        from src.config.component_factory import ComponentFactory, TradingMode

        actual_mode = mode or config.system.mode
        trading_mode = TradingMode(actual_mode)

        if components is None:
            # Auto-create components
            finance_params = config.finance.to_dict()
            
            # Data fetcher
            data_fetcher = ComponentFactory.create_data_fetcher(trading_mode)
            
            # Executor
            executor = ComponentFactory.create_executor(trading_mode, finance_params)
            
            # Position manager
            position_manager = ComponentFactory.create_position_manager(
                executor,
                finance_params,
                data_fetcher=data_fetcher if trading_mode != TradingMode.SIMULATION else None
            )
            
            # Strategy
            strategy = ComponentFactory.create_strategy_from_config(config)
            
            # Visualizer
            visualizer = None
            if config.system.enable_chart:
                from pathlib import Path
                charts_dir = Path(config.system.output_dir) / "charts"
                charts_dir.mkdir(parents=True, exist_ok=True)
                chart_file = str(charts_dir / f"{config.system.ticker}_{config.system.strategy}_{actual_mode}.html")
                
                visualizer = ComponentFactory.create_visualizer(
                    ticker=config.system.ticker,
                    output_file=chart_file,
                    auto_open=config.system.auto_open_browser,
                    initial_capital=config.finance.initial_capital,
                    bb_narrow_threshold=config.up_trend_aware.bb_narrow_threshold
                )
            
            components = EngineComponents(
                ticker=config.system.ticker,
                strategy=strategy,
                position_manager=position_manager,
                data_fetcher=data_fetcher,
                executor=executor,
                visualizer=visualizer,
            )

        engine_mode = 'live' if actual_mode in [
            'paper', 'live'] else 'backtest'
        return cls(components, config, mode=engine_mode)

    # ==========================================
    # Time Utilities
    # ==========================================

    def _to_eastern(self, dt: datetime) -> datetime:
        """Convert datetime to Eastern time."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(self._tz)

    def _is_market_hours(self, dt: datetime) -> bool:
        """Check if time is within market hours."""
        et = self._to_eastern(dt)
        current_time = et.time()

        if et.weekday() >= 5:
            return False

        return self.config.time.market_open_time <= current_time < self.config.time.market_close_time

    def _is_in_no_entry_window(self, dt: datetime) -> bool:
        """Check if in no-new-entry window."""
        et = self._to_eastern(dt)
        return et.time() >= self.config.time.last_entry_time

    def _is_force_close_time(self, dt: datetime) -> bool:
        """Check if it's force close time."""
        et = self._to_eastern(dt)
        return et.time() >= self.config.time.force_close_time

    def _format_time_et(self, dt: datetime) -> str:
        """Format datetime as ET string."""
        et = self._to_eastern(dt)
        return et.strftime('%H:%M:%S ET')

    # ==========================================
    # Data Management
    # ==========================================

    def _fetch_data(self, end_dt: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch market data."""
        if self.data_fetcher is None:
            return pd.DataFrame()

        return self.data_fetcher.get_latest_bars(
            ticker=self.ticker,
            lookback_minutes=self.config.data.lookback_minutes,
            end_dt=end_dt,
            timeframe=self.config.data.timeframe
        )

    # ==========================================
    # Signal & Trade Execution
    # ==========================================

    def _process_iteration(self, current_time: datetime) -> Dict[str, Any]:
        """Process a single iteration (one time step)."""
        self._iteration += 1

        # Fetch data
        df = self._fetch_data(
            end_dt=current_time if self.mode == 'backtest' else None)

        if df.empty:
            return {'status': 'no_data', 'iteration': self._iteration}

        current_price = self.data_fetcher.get_latest_price(
            ticker=self.ticker,
            current_time=current_time,
        )
        
        current_et = self._to_eastern(current_time)

        # Get account status
        account_status = self.position_manager.get_account_status(
            current_price)
        current_position = account_status.get('position', 0.0)
        avg_cost = account_status.get('avg_cost', 0.0)
        current_equity = account_status.get(
            'equity', self.config.finance.initial_capital)

        # Check time windows
        is_force_close = self._is_force_close_time(current_time)
        is_no_entry = self._is_in_no_entry_window(current_time)

        # Log time window transitions
        if not self._last_entry_reached and is_no_entry:
            print(
                f"\n⏰ Reached last entry time: {self._format_time_et(current_time)}")
            self._last_entry_reached = True

        if not self._force_close_reached and is_force_close:
            print(
                f"\n🔔 Reached force close time: {self._format_time_et(current_time)}")
            self._force_close_reached = True

        # Get strategy signal
        try:
            signal_data, _ = self.strategy.get_signal(
                ticker=self.ticker,
                new_data=df,
                current_position=current_position,
                current_price=current_price,
                avg_cost=avg_cost,
                verbose=self.config.system.verbose,
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

                if self.on_trade_callback:
                    self.on_trade_callback(signal, current_price, current_time)

            if self.on_signal_callback:
                self.on_signal_callback(
                    signal_data, current_price, current_time)

        # Update visualizer
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

        if self.on_iteration_callback:
            self.on_iteration_callback(
                self._iteration, current_time, current_equity)

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
        """Ensure all positions are closed at end of session."""
        df = self._fetch_data(
            end_dt=end_time if self.mode == 'backtest' else None)
        final_price = df.iloc[-1]['close'] if not df.empty else 0.0

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

                final_status = self.position_manager.get_account_status(
                    final_price)
                final_position = final_status.get('position', 0.0)

                if final_position == 0:
                    print(f"   ✅ Position closed successfully")
                    return True
                else:
                    print(
                        f"   ❌ Warning: Position still open ({final_position} shares)")
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
        """Run backtest from start_time to end_time."""
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
        print(f"   Step: {self.config.data.step_seconds} seconds")

        current_time = start_time
        step = timedelta(seconds=self.config.data.step_seconds)

        try:
            while current_time <= end_time and self._running:
                if current_time.tzinfo is None:
                    current_time = current_time.replace(tzinfo=timezone.utc)

                result = self._process_iteration(current_time)

                if self._iteration % progress_interval == 0:
                    progress = (current_time - start_time) / \
                        (end_time - start_time) * 100
                    current_et = self._to_eastern(current_time)
                    equity = result.get('equity', 0)
                    position = result.get('position', 0)
                    print(f"\n📊 Progress: {progress:.1f}% | Time: {current_et.strftime('%H:%M')} ET | "
                          f"Equity: ${equity:,.0f} | Position: {position:.0f}")

                current_time += step

        except KeyboardInterrupt:
            print("\n⚠️ Backtest interrupted by user")

        self._ensure_flat_position(end_time)

        return self._generate_report(end_time)

    def run_live(self,
                 max_runtime_minutes: Optional[int] = None,
                 interval_seconds: Optional[int] = None) -> Dict[str, Any]:
        """Run live trading."""
        import time as time_module

        self._running = True
        self._start_time = datetime.now(timezone.utc)
        self._iteration = 0
        self._signals_count = 0
        self._trades_count = 0
        self._equity_history = []
        self._last_entry_reached = False
        self._force_close_reached = False

        interval = interval_seconds or self.config.data.step_seconds
        max_runtime = max_runtime_minutes or self.config.system.max_runtime_minutes

        print(f"\n{'='*60}")
        print(f"🚀 Starting Live Trading")
        print(f"{'='*60}")
        print(f"   Ticker: {self.ticker}")
        print(f"   Interval: {interval} seconds")
        if max_runtime:
            print(f"   Max Runtime: {max_runtime} minutes")

        try:
            while self._running:
                current_time = datetime.now(timezone.utc)

                if max_runtime:
                    runtime = (current_time -
                               self._start_time).total_seconds() / 60
                    if runtime >= max_runtime:
                        print(
                            f"\n⏰ Max runtime reached ({max_runtime} minutes)")
                        break

                if self.config.system.respect_market_hours:
                    if not self._is_market_hours(current_time):
                        current_et = self._to_eastern(current_time)

                        if current_et.time() >= self.config.time.market_close_time:
                            print(f"\n🔔 Market closed for today")
                            break

                        print(
                            f"⏳ [{current_et.strftime('%H:%M:%S')}] Waiting for market hours...")
                        time_module.sleep(60)
                        continue

                self._process_iteration(current_time)
                time_module.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n⚠️ Live trading interrupted by user")

        end_time = datetime.now(timezone.utc)
        self._ensure_flat_position(end_time)

        return self._generate_report(end_time)

    def run(self,
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None,
            **kwargs) -> Dict[str, Any]:
        """Main entry point - runs in appropriate mode."""
        if self.mode == 'backtest':
            if start_time is None or end_time is None:
                raise ValueError(
                    "start_time and end_time required for backtest mode")
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
        df = self._fetch_data(
            end_dt=end_time if self.mode == 'backtest' else None)
        final_price = df.iloc[-1]['close'] if not df.empty else 0.0
        final_status = self.position_manager.get_account_status(final_price)

        runtime_seconds = (datetime.now(
            timezone.utc) - self._start_time).total_seconds() if self._start_time else 0

        trade_log = self.position_manager.get_trade_log()
        initial_capital = self.config.finance.initial_capital

        report = {
            'mode': self.mode,
            'ticker': self.ticker,
            'strategy': type(self.strategy).__name__,
            'runtime_seconds': runtime_seconds,
            'iterations': self._iteration,
            'signals': self._signals_count,
            'trades_executed': self._trades_count,
            'initial_capital': initial_capital,
            'final_equity': final_status.get('equity', 0),
            'final_position': final_status.get('position', 0),
            'final_cash': final_status.get('cash', 0),
            'pnl': final_status.get('equity', 0) - initial_capital,
            'pnl_pct': ((final_status.get('equity', 0) - initial_capital) /
                        initial_capital * 100) if initial_capital > 0 else 0,
            'force_close_time_reached': self._force_close_reached,
            'last_entry_time_reached': self._last_entry_reached,
            'trade_log': trade_log,
            'equity_history': self._equity_history
        }

        if trade_log is not None and not trade_log.empty:
            completed_trades = trade_log[trade_log['type'].isin(
                ['SELL', 'COVER'])]
            if not completed_trades.empty and 'net_pnl' in completed_trades.columns:
                winning_trades = completed_trades[completed_trades['net_pnl'] > 0]
                report['completed_trades'] = len(completed_trades)
                report['winning_trades'] = len(winning_trades)
                report['win_rate'] = len(
                    winning_trades) / len(completed_trades) if len(completed_trades) > 0 else 0

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
        print(
            f"   Position: {report['final_position']:.0f} shares {'✅' if report['final_position'] == 0 else '⚠️'}")

        if 'win_rate' in report:
            print(f"\n📈 Trade Statistics:")
            print(f"   Completed: {report['completed_trades']}")
            print(f"   Winners: {report['winning_trades']}")
            print(f"   Win Rate: {report['win_rate']*100:.1f}%")

        print(f"\n⏰ Time Controls:")
        print(
            f"   Last Entry Reached: {'✅' if report['last_entry_time_reached'] else '❌'}")
        print(
            f"   Force Close Reached: {'✅' if report['force_close_time_reached'] else '❌'}")

        print(f"{'='*60}")
