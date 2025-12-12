# src/runner/backtest_runner.py

"""
Simplified Backtest Runner

This runner is a thin wrapper around TradingEngine.
It handles:
1. Command-line argument parsing
2. Component initialization via factory
3. Time range setup
4. Running the engine

Usage:
    python backtest_runner.py --strategy moderate --ticker TSLA --date 2024-12-05
    python backtest_runner.py --strategy trend_aware --ticker AAPL --date 2024-12-06 --no-chart --api_data
"""

from datetime import datetime, timezone, timedelta
import argparse
from pathlib import Path
import pytz

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
DEFAULT_DATE = None  # Will use yesterday if not specified

DEFAULT_INITIAL_CAPITAL = 1000.0
DEFAULT_STEP_SECONDS = 30
DEFAULT_LOOKBACK_MINUTES = 300


# ==========================================
# Main Runner
# ==========================================

def run_backtest(
    ticker: str,
    strategy_name: str,
    trading_date: str,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    step_seconds: int = DEFAULT_STEP_SECONDS,
    lookback_minutes: int = DEFAULT_LOOKBACK_MINUTES,
    enable_chart: bool = True,
    auto_open_browser: bool = True,
    output_dir: str = "backtest_results",
    verbose: bool = True,
    use_local_data: bool = False,
    local_data_dir: str = "data/"
) -> dict:
    """
    Run a single-day backtest.
    
    Args:
        ticker: Stock ticker symbol
        strategy_name: Strategy key from registry
        trading_date: Date string 'YYYY-MM-DD'
        initial_capital: Starting capital
        step_seconds: Time step in seconds
        lookback_minutes: Data lookback period
        enable_chart: Whether to generate chart
        auto_open_browser: Auto-open chart in browser
        output_dir: Output directory for results
        verbose: Print detailed output
        use_local_data: If True, use local CSV files instead of Alpaca API
        local_data_dir: Directory containing CSV files (when use_local_data=True)
        
    Returns:
        Backtest results dictionary
    """
    US_EASTERN = pytz.timezone('America/New_York')
    
    print("\n" + "="*70)
    print(f"🚀 Backtest Runner")
    print("="*70)
    
    # Parse date
    date_parts = [int(x) for x in trading_date.split('-')]
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # ==========================================
    # 1. Create Configuration
    # ==========================================
    
    # Finance config
    finance_config = FinanceConfig(
        initial_capital=initial_capital,
        commission_rate=0.0003,
        slippage_rate=0.0001,
        min_lot_size=1,
        max_allocation=0.95
    )
    
    # Time config
    time_config = TimeConfig()
    
    # Data config
    data_config = DataConfig(
        lookback_minutes=lookback_minutes,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        step_seconds=step_seconds
    )
    
    # ==========================================
    # 2. Create Components via Factory
    # ==========================================
    
    print(f"\n🔧 Initializing Components...")
    print(f"   Ticker: {ticker}")
    print(f"   Date: {trading_date}")
    print(f"   Initial Capital: ${initial_capital:,.2f}")
    
    # Data fetcher (Alpaca API or local CSV)
    if use_local_data:
        data_fetcher = ComponentFactory.create_local_data_fetcher(
            data_dir=local_data_dir,
            ticker=ticker,
            verbose=verbose
        )
    else:
        data_fetcher = ComponentFactory.create_data_fetcher(TradingMode.PAPER)
    
    # Executor (simulation for backtest)
    executor = ComponentFactory.create_executor(
        TradingMode.SIMULATION, 
        finance_config.to_dict()
    )
    
    # Position manager
    position_manager = ComponentFactory.create_position_manager(
        executor, 
        finance_config.to_dict()
    )
    
    # Strategy
    strategy = StrategyRegistry.create(strategy_name)
    
    # Visualizer (optional)
    visualizer = None
    chart_file = None
    if enable_chart:
        strategy_info = StrategyRegistry.get_info(strategy_name)
        chart_file = str(Path(output_dir) / f"{ticker}_{trading_date}_{strategy_name}.html")
        
        visualizer = ComponentFactory.create_visualizer(
            ticker=ticker,
            output_file=chart_file,
            auto_open=auto_open_browser,
            initial_capital=initial_capital
        )
        print(f"   Chart: {chart_file}")
    
    # ==========================================
    # 3. Create Engine Configuration
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
        verbose=verbose
    )
    
    # ==========================================
    # 4. Create and Run Engine
    # ==========================================
    
    engine = TradingEngine(engine_config, mode='backtest')
    
    # Set up time range
    start_time = US_EASTERN.localize(
        datetime(date_parts[0], date_parts[1], date_parts[2], 9, 30)
    ).astimezone(timezone.utc)
    
    end_time = US_EASTERN.localize(
        datetime(date_parts[0], date_parts[1], date_parts[2], 16, 0)
    ).astimezone(timezone.utc)
    
    # Run backtest
    report = engine.run(start_time, end_time, progress_interval=10)
    
    # Print report
    engine.print_report(report)
    
    # ==========================================
    # 5. Final Summary
    # ==========================================
    
    print(f"\n" + "="*70)
    print(f"✅ Backtest Complete!")
    print("="*70)
    
    if chart_file:
        print(f"   📊 Chart: {chart_file}")
    
    print(f"   💰 Final Equity: ${report['final_equity']:,.2f}")
    print(f"   📈 PnL: ${report['pnl']:,.2f} ({report['pnl_pct']:+.2f}%)")
    print("="*70 + "\n")
    
    return report


def main():
    """Main entry point with CLI argument parsing."""
    
    parser = argparse.ArgumentParser(
        description='Simplified Backtest Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python backtest_runner.py --strategy moderate --ticker TSLA --date 2024-12-05
    python backtest_runner.py --strategy trend_aware --ticker AAPL --date 2024-12-06 --no-chart
    python backtest_runner.py --strategy high_freq --capital 5000

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
        '--date', '-d',
        type=str,
        default=None,
        help='Trading date YYYY-MM-DD (default: yesterday)'
    )
    
    parser.add_argument(
        '--capital', '-c',
        type=float,
        default=DEFAULT_INITIAL_CAPITAL,
        help=f'Initial capital (default: {DEFAULT_INITIAL_CAPITAL})'
    )
    
    parser.add_argument(
        '--step',
        type=int,
        default=DEFAULT_STEP_SECONDS,
        help=f'Step interval in seconds (default: {DEFAULT_STEP_SECONDS})'
    )
    
    parser.add_argument(
        '--lookback',
        type=int,
        default=DEFAULT_LOOKBACK_MINUTES,
        help=f'Data lookback in minutes (default: {DEFAULT_LOOKBACK_MINUTES})'
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
        '--output-dir', '-o',
        type=str,
        default='backtest_results',
        help='Output directory (default: backtest_results)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    parser.add_argument(
        '--local-data',
        action='store_true',
        help='Use local CSV files instead of Alpaca API'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/',
        help='Directory containing CSV files (when using --local-data)'
    )
    
    args = parser.parse_args()
    
    # Default date to yesterday if not specified
    if args.date is None:
        from datetime import date
        yesterday = date.today() - timedelta(days=1)
        # Skip weekends
        while yesterday.weekday() >= 5:
            yesterday -= timedelta(days=1)
        args.date = yesterday.strftime('%Y-%m-%d')
    
    # Run backtest
    run_backtest(
        ticker=args.ticker,
        strategy_name=args.strategy,
        trading_date=args.date,
        initial_capital=args.capital,
        step_seconds=args.step,
        lookback_minutes=args.lookback,
        enable_chart=not args.no_chart,
        auto_open_browser=not args.no_browser,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        use_local_data=not args.api_data,
        local_data_dir=args.data_dir
    )


if __name__ == '__main__':
    main()