# src/runner/backtest_runner.py

"""
Simplified Backtest Runner

This runner uses the centralized TradingConfig system and TradingEngine.
It handles:
1. Command-line argument parsing
2. Configuration setup via TradingConfig
3. Time range setup
4. Running the engine

Usage:
    python -m src.runner.backtest_runner --strategy moderate --ticker TSLA --date 2025-01-01
    python -m src.runner.backtest_runner --strategy up_trend_aware --ticker SPLV --date 2025-12-05 --finance-preset small --monitor-frequency fast --local-data --data-dir "/Users/vivi/vivi-llm-trader/data/"
"""

from datetime import datetime, timezone, timedelta
import argparse
from pathlib import Path
import pytz

from src.config.trading_config import (
    TradingConfig,
    FinanceConfig,
    TimeConfig,
    DataConfig,
    SystemConfig,
    get_full_config,
)
from src.config.component_factory import (
    ComponentFactory,
    StrategyRegistry,
    TradingMode,
)
from src.engine.trading_engine import TradingEngine, EngineComponents


# ==========================================
# Default Configuration
# ==========================================

DEFAULT_TICKER = "TSLA"
DEFAULT_STRATEGY = "up_trend_aware"
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
    initial_capital: float = None,
    step_seconds: int = None,
    lookback_minutes: int = None,
    enable_chart: bool = True,
    auto_open_browser: bool = True,
    output_dir: str = "backtest_results",
    verbose: bool = True,
    use_local_data: bool = False,
    local_data_dir: str = "data/",
    strategy_preset: str = None,
    monitor_frequency: str = None,
    finance_preset: str = None,
) -> dict:
    """
    Run a single-day backtest.

    Args:
        ticker: Stock ticker symbol
        strategy_name: Strategy key from registry
        trading_date: Date string 'YYYY-MM-DD'
        initial_capital: Starting capital (None uses preset/default)
        step_seconds: Time step in seconds (None uses preset/default)
        lookback_minutes: Data lookback period (None uses preset/default)
        enable_chart: Whether to generate chart
        auto_open_browser: Auto-open chart in browser
        output_dir: Output directory for results
        verbose: Print detailed output
        use_local_data: If True, use local CSV files instead of Alpaca API
        local_data_dir: Directory containing CSV files (when use_local_data=True)
        strategy_preset: Strategy preset ('conservative', 'moderate', 'aggressive')
        monitor_frequency: Data frequency preset ('fast', 'medium', 'slow')
        finance_preset: Finance preset ('small', 'medium', 'large', 'paper')

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
    # 1. Create TradingConfig
    # ==========================================

    # Create config using the centralized system with presets
    config = get_full_config(
        initial_capital=initial_capital,
        ticker=ticker,
        strategy=strategy_name,
        mode='simulation',  # Backtest uses simulation mode
        strategy_preset=strategy_preset,
        monitor_frequency=monitor_frequency,
        finance_preset=finance_preset,
    )

    # Override data config parameters if explicitly provided
    if step_seconds is not None:
        config.data.step_seconds = step_seconds
    if lookback_minutes is not None:
        config.data.lookback_minutes = lookback_minutes

    # Get actual initial capital (from preset or explicit value)
    actual_initial_capital = config.finance.initial_capital

    # Override system config parameters
    config.system.enable_chart = enable_chart
    config.system.auto_open_browser = auto_open_browser
    config.system.output_dir = output_dir
    config.system.verbose = verbose

    # Print configuration summary
    print(f"\n🔧 Configuration:")
    print(f"   Ticker: {ticker}")
    print(f"   Date: {trading_date}")
    print(f"   Strategy: {strategy_name}" +
          (f" (preset: {strategy_preset})" if strategy_preset else ""))
    print(
        f"   Finance: {finance_preset or 'default'} (${actual_initial_capital:,.2f})")
    print(f"   Monitor Frequency: {monitor_frequency or 'default'}")
    print(f"   Step: {config.data.step_seconds} seconds")
    print(f"   Lookback: {config.data.lookback_minutes} minutes")

    # ==========================================
    # 2. Create Components
    # ==========================================

    print(f"\n🔧 Initializing Components...")

    mode = TradingMode.SIMULATION
    finance_params = config.finance.to_dict()

    # Data fetcher (Alpaca API or local CSV)
    if use_local_data:
        data_fetcher = ComponentFactory.create_data_fetcher(
            mode=mode,
            use_local=True,
            local_data_dir=local_data_dir
        )
    else:
        data_fetcher = ComponentFactory.create_data_fetcher(mode)

    # Executor (simulation for backtest)
    executor = ComponentFactory.create_executor(mode, finance_params)

    # Position manager
    position_manager = ComponentFactory.create_position_manager(
        executor,
        finance_params
    )

    # Strategy
    strategy = ComponentFactory.create_strategy_from_config(config)

    # Visualizer - 不在运行时更新，只在最后生成
    # 先不创建 visualizer，等回测完成后再生成图表
    chart_file = None
    if enable_chart:
        chart_file = str(Path(output_dir) /
                         f"{ticker}_{trading_date}_{strategy_name}.html")
        print(f"   Chart: {chart_file} (will generate after backtest)")

    # ==========================================
    # 3. Create Engine Components Container
    # ==========================================

    components = EngineComponents(
        ticker=ticker,
        strategy=strategy,
        position_manager=position_manager,
        data_fetcher=data_fetcher,
        executor=executor,
        visualizer=None,  # 不传入 visualizer，避免运行时更新
    )

    # ==========================================
    # 4. Create and Run Engine
    # ==========================================

    engine = TradingEngine(components, config, mode='backtest')

    # Set up time range
    start_time = US_EASTERN.localize(
        datetime(date_parts[0], date_parts[1], date_parts[2], 9, 30)
    ).astimezone(timezone.utc)

    end_time = US_EASTERN.localize(
        datetime(date_parts[0], date_parts[1], date_parts[2], 16, 0)
    ).astimezone(timezone.utc)

    # Run backtest
    report = engine.run(start_time, end_time, progress_interval=10)

    # ==========================================
    # 5. Generate Chart (after backtest completes)
    # ==========================================

    if enable_chart and chart_file:
        print(f"\n📊 Generating chart...")
        try:
            visualizer = ComponentFactory.create_visualizer(
                ticker=ticker,
                output_file=chart_file,
                auto_open=auto_open_browser,
                initial_capital=actual_initial_capital
            )

            # 获取最终数据
            strategy_df = strategy.get_history_data(ticker)
            trade_log = position_manager.get_trade_log()
            final_status = position_manager.get_account_status(
                strategy_df.iloc[-1]['close'] if not strategy_df.empty else 0)

            if not strategy_df.empty:
                visualizer.update_data(
                    market_data=strategy_df,
                    trade_log=trade_log,
                    current_equity=final_status.get(
                        'equity', actual_initial_capital),
                    current_position=final_status.get('position', 0),
                    timestamp=end_time
                )
                print(f"   ✅ Chart saved: {chart_file}")
            else:
                print(f"   ⚠️ No data for chart")
        except Exception as e:
            print(f"   ❌ Chart generation failed: {e}")

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
    python -m src.runner.backtest_runner --strategy moderate --ticker TSLA --date 2024-12-05
    python -m src.runner.backtest_runner --strategy up_trend_aware --ticker AAPL --date 2024-12-06 --no-chart
    python -m src.runner.backtest_runner --strategy mean_reversion --finance-preset medium
    python -m src.runner.backtest_runner --strategy up_trend_aware --preset conservative --monitor-frequency fast --finance-preset large

Presets:
    Strategy: conservative, moderate, aggressive
    Monitor Frequency: fast (1min/10s), medium (5min/30s), slow (15min/60s)
    Finance: small ($1k), medium ($5k), large ($25k), paper ($100k)

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
        '--preset', '-p',
        type=str,
        default=None,
        choices=['conservative', 'moderate', 'aggressive'],
        help='Strategy preset (conservative, moderate, aggressive)'
    )

    parser.add_argument(
        '--monitor-frequency',
        type=str,
        default=None,
        choices=['fast', 'medium', 'slow'],
        help='Monitor frequency preset (fast: 1min/10s, medium: 5min/30s, slow: 15min/60s)'
    )

    parser.add_argument(
        '--finance-preset',
        type=str,
        default=None,
        choices=['small', 'medium', 'large', 'paper'],
        help='Finance preset (small: $1k, medium: $5k, large: $25k, paper: $100k)'
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
        default=None,
        help=f'Initial capital (default: from preset or {DEFAULT_INITIAL_CAPITAL})'
    )

    parser.add_argument(
        '--step',
        type=int,
        default=None,
        help=f'Step interval in seconds (default: from preset or {DEFAULT_STEP_SECONDS})'
    )

    parser.add_argument(
        '--lookback',
        type=int,
        default=None,
        help=f'Data lookback in minutes (default: from preset or {DEFAULT_LOOKBACK_MINUTES})'
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
        step_seconds=args.step if args.step else None,
        lookback_minutes=args.lookback if args.lookback else None,
        enable_chart=not args.no_chart,
        auto_open_browser=not args.no_browser,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        use_local_data=args.local_data,
        local_data_dir=args.data_dir,
        strategy_preset=args.preset,
        monitor_frequency=args.monitor_frequency,
        finance_preset=args.finance_preset,
    )


if __name__ == '__main__':
    main()
