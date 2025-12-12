# src/runner/live_runner.py

"""
Simplified Live Trading Runner

This runner is a thin wrapper around TradingEngine.
It handles:
1. Command-line argument parsing
2. Component initialization via factory
3. API position synchronization
4. Running the engine in live mode

Usage:
    python live_runner.py --strategy moderate --ticker TSLA --mode paper
    python live_runner.py --strategy trend_aware --ticker AAPL --mode simulation
    python live_runner.py --strategy mean_reversion --mode live  # ⚠️ Real money!
"""

from datetime import datetime, timezone
import argparse
from pathlib import Path
import threading
import time as time_module

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from src.engine.trading_engine import (
    TradingEngine, 
    EngineConfig, 
    TimeConfig, 
    FinanceConfig, 
    DataConfig
)
from src.factory.component_factory import (
    ComponentFactory, 
    StrategyRegistry, 
    TradingMode
)


# ==========================================
# Default Configuration
# ==========================================

DEFAULT_TICKER = "TSLA"
DEFAULT_STRATEGY = "moderate"
DEFAULT_MODE = "paper"

DEFAULT_INITIAL_CAPITAL = 1000.0
DEFAULT_INTERVAL_SECONDS = 30
DEFAULT_LOOKBACK_MINUTES = 300

SYNC_POSITION_ON_START = True
CHART_UPDATE_INTERVAL = 30


# ==========================================
# Chart Update Thread
# ==========================================

class ChartUpdater(threading.Thread):
    """Background thread for updating charts during live trading."""
    
    def __init__(self, 
                 visualizer,
                 strategy,
                 position_manager,
                 ticker: str,
                 update_interval: int = 30):
        super().__init__()
        self.visualizer = visualizer
        self.strategy = strategy
        self.position_manager = position_manager
        self.ticker = ticker
        self.update_interval = update_interval
        self._running = True
        self.daemon = True
    
    def run(self):
        """Run chart update loop."""
        print(f"\n📊 Chart updater started (every {self.update_interval}s)")
        
        while self._running:
            try:
                strategy_df = self.strategy.get_history_data(self.ticker)
                
                if strategy_df.empty:
                    time_module.sleep(self.update_interval)
                    continue
                
                current_price = strategy_df.iloc[-1]['close']
                account_status = self.position_manager.get_account_status(current_price)
                
                self.visualizer.update_data(
                    market_data=strategy_df,
                    trade_log=self.position_manager.get_trade_log(),
                    current_equity=account_status.get('equity', 0),
                    current_position=account_status.get('position', 0),
                    timestamp=datetime.now(timezone.utc)
                )
                
                time_module.sleep(self.update_interval)
                
            except Exception as e:
                print(f"⚠️ Chart update error: {e}")
                time_module.sleep(self.update_interval)
    
    def stop(self):
        """Stop the updater."""
        self._running = False


# ==========================================
# Signal Callback
# ==========================================

def default_signal_callback(signal_dict: dict, price: float, timestamp: datetime):
    """Default callback for signal notifications."""
    signal = signal_dict.get('signal', 'UNKNOWN')
    reason = signal_dict.get('reason', '')
    
    if signal in ['BUY', 'SELL', 'SHORT', 'COVER']:
        if '强制平仓' in reason or '收盘' in reason or 'force' in reason.lower():
            print(f"   🔔 Market Close Force Close")


# ==========================================
# Main Runner
# ==========================================

def run_live(
    ticker: str,
    strategy_name: str,
    mode: str,  # 'simulation', 'paper', 'live'
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    lookback_minutes: int = DEFAULT_LOOKBACK_MINUTES,
    max_runtime_minutes: int = None,
    enable_chart: bool = True,
    auto_open_browser: bool = True,
    sync_position: bool = SYNC_POSITION_ON_START,
    output_dir: str = "live_trading",
    verbose: bool = True
) -> dict:
    """
    Run live trading.
    
    Args:
        ticker: Stock ticker symbol
        strategy_name: Strategy key from registry
        mode: Trading mode ('simulation', 'paper', 'live')
        initial_capital: Starting capital
        interval_seconds: Update interval in seconds
        lookback_minutes: Data lookback period
        max_runtime_minutes: Maximum runtime (None = unlimited)
        enable_chart: Whether to generate chart
        auto_open_browser: Auto-open chart in browser
        sync_position: Sync position from API on start
        output_dir: Output directory for results
        verbose: Print detailed output
        
    Returns:
        Trading results dictionary
    """
    # Convert mode string to enum
    trading_mode = TradingMode(mode)
    
    print("\n" + "="*70)
    print(f"🚀 Live Trading Runner")
    print("="*70)
    print(f"   Mode: {mode.upper()}")
    print(f"   Ticker: {ticker}")
    print(f"   Strategy: {strategy_name}")
    print(f"   Initial Capital: ${initial_capital:,.2f}")
    print(f"   Interval: {interval_seconds}s")
    
    # Live mode warning
    if trading_mode == TradingMode.LIVE:
        print("\n" + "⚠️"*20)
        print("   WARNING: LIVE TRADING MODE!")
        print("   All trades will use REAL MONEY!")
        print("⚠️"*20)
        
        confirm = input("\nConfirm live trading? (type 'YES' to confirm): ")
        if confirm != 'YES':
            print("Cancelled.")
            return {}
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    charts_dir = Path(output_dir) / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================
    # 1. Create Configuration
    # ==========================================
    
    finance_config = FinanceConfig(
        initial_capital=initial_capital,
        commission_rate=0.0003,
        slippage_rate=0.0001,
        min_lot_size=1,
        max_allocation=0.95
    )
    
    time_config = TimeConfig()
    
    data_config = DataConfig(
        lookback_minutes=lookback_minutes,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        step_seconds=interval_seconds
    )
    
    # ==========================================
    # 2. Create Components
    # ==========================================
    
    print(f"\n🔧 Initializing Components...")
    
    # Data fetcher
    data_fetcher = ComponentFactory.create_data_fetcher(trading_mode)
    
    # Executor
    executor = ComponentFactory.create_executor(
        trading_mode, 
        finance_config.to_dict()
    )
    
    # Position manager (with data_fetcher for API sync)
    position_manager = ComponentFactory.create_position_manager(
        executor, 
        finance_config.to_dict(),
        data_fetcher=data_fetcher if trading_mode != TradingMode.SIMULATION else None
    )
    
    # Strategy
    strategy = StrategyRegistry.create(strategy_name)
    
    # Sync position from API
    if sync_position and trading_mode in [TradingMode.PAPER, TradingMode.LIVE]:
        print(f"\n🔄 Syncing position from API for {ticker}...")
        sync_success = position_manager.sync_from_api(ticker)
        if not sync_success:
            print("⚠️ Position sync failed, using local initial state")
    
    # Visualizer
    visualizer = None
    chart_updater = None
    chart_file = None
    
    if enable_chart:
        process_id = f"{ticker}_{strategy_name}_{mode}"
        chart_file = str(charts_dir / f"{process_id}.html")
        
        visualizer = ComponentFactory.create_visualizer(
            ticker=ticker,
            output_file=chart_file,
            auto_open=auto_open_browser,
            initial_capital=initial_capital
        )
        print(f"   Chart: {chart_file}")
    
    # ==========================================
    # 3. Create Engine
    # ==========================================
    
    engine_config = EngineConfig(
        ticker=ticker,
        strategy=strategy,
        position_manager=position_manager,
        data_fetcher=data_fetcher,
        executor=executor,
        time_config=time_config,
        finance_config=finance_config,
        data_config=data_config,
        visualizer=visualizer,
        on_signal_callback=default_signal_callback,
        respect_market_hours=True,
        verbose=verbose
    )
    
    engine = TradingEngine(engine_config, mode='live')
    
    # Start chart updater thread
    if visualizer:
        chart_updater = ChartUpdater(
            visualizer=visualizer,
            strategy=strategy,
            position_manager=position_manager,
            ticker=ticker,
            update_interval=CHART_UPDATE_INTERVAL
        )
        chart_updater.start()
    
    # ==========================================
    # 4. Run Engine
    # ==========================================
    
    try:
        report = engine.run(
            max_runtime_minutes=max_runtime_minutes,
            interval_seconds=interval_seconds
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        report = engine._generate_report(datetime.now(timezone.utc))
    finally:
        # Stop chart updater
        if chart_updater:
            print("\n🛑 Stopping chart updater...")
            chart_updater.stop()
            chart_updater.join(timeout=2)
    
    # Print report
    engine.print_report(report)
    
    # ==========================================
    # 5. Print Trade Log
    # ==========================================
    
    trade_log = position_manager.get_trade_log()
    
    if trade_log is not None and not trade_log.empty:
        print("\n📝 Trade Log:")
        display_log = trade_log[['time', 'type', 'qty', 'price', 'fee', 'net_pnl']].copy()
        display_log['time'] = display_log['time'].dt.strftime('%Y-%m-%d %H:%M')
        
        try:
            print(display_log.to_markdown(index=False, floatfmt=".2f"))
        except Exception:
            print(display_log.to_string(index=False))
    else:
        print("\n🤷 No trades executed.")
    
    # Final summary
    print(f"\n" + "="*70)
    print(f"✅ Live Trading Complete!")
    print("="*70)
    
    if chart_file:
        print(f"   📊 Chart: {chart_file}")
    
    print(f"   💰 Final Equity: ${report.get('final_equity', 0):,.2f}")
    print(f"   📈 PnL: ${report.get('pnl', 0):,.2f} ({report.get('pnl_pct', 0):+.2f}%)")
    print("="*70 + "\n")
    
    return report


def main():
    """Main entry point with CLI argument parsing."""
    
    parser = argparse.ArgumentParser(
        description='Simplified Live Trading Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python live_runner.py --strategy moderate --ticker TSLA --mode paper
    python live_runner.py --strategy trend_aware --ticker AAPL --mode simulation
    python live_runner.py --strategy mean_reversion --mode live  # ⚠️ Real money!

Modes:
    simulation  - Local simulation (no API calls for trades)
    paper       - Alpaca paper trading (simulated money)
    live        - Alpaca live trading (REAL MONEY!)

Available Strategies:
""" + "\n".join([f"    {k}: {v}" for k, v in StrategyRegistry.list_strategies().items()])
    )
    
    parser.add_argument(
        '--ticker', '-t',
        type=str,
        default=DEFAULT_TICKER,
        help=f'Stock ticker (default: {DEFAULT_TICKER})'
    )
    
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        default=DEFAULT_STRATEGY,
        choices=StrategyRegistry.get_all_keys(),
        help=f'Strategy name (default: {DEFAULT_STRATEGY})'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        default=DEFAULT_MODE,
        choices=['simulation', 'paper', 'live'],
        help=f'Trading mode (default: {DEFAULT_MODE})'
    )
    
    parser.add_argument(
        '--capital', '-c',
        type=float,
        default=DEFAULT_INITIAL_CAPITAL,
        help=f'Initial capital (default: {DEFAULT_INITIAL_CAPITAL})'
    )
    
    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
        help=f'Update interval in seconds (default: {DEFAULT_INTERVAL_SECONDS})'
    )
    
    parser.add_argument(
        '--lookback',
        type=int,
        default=DEFAULT_LOOKBACK_MINUTES,
        help=f'Data lookback in minutes (default: {DEFAULT_LOOKBACK_MINUTES})'
    )
    
    parser.add_argument(
        '--max-runtime',
        type=int,
        default=None,
        help='Maximum runtime in minutes (default: unlimited)'
    )
    
    parser.add_argument(
        '--no-chart',
        action='store_true',
        help='Disable chart generation'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Don\'t auto-open chart in browser'
    )
    
    parser.add_argument(
        '--no-sync',
        action='store_true',
        help='Don\'t sync position from API on start'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='live_trading',
        help='Output directory (default: live_trading)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    # Run live trading
    run_live(
        ticker=args.ticker,
        strategy_name=args.strategy,
        mode=args.mode,
        initial_capital=args.capital,
        interval_seconds=args.interval,
        lookback_minutes=args.lookback,
        max_runtime_minutes=args.max_runtime,
        enable_chart=not args.no_chart,
        auto_open_browser=not args.no_browser,
        sync_position=not args.no_sync,
        output_dir=args.output_dir,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()