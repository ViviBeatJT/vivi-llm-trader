# src/runner/bulk_backtest_runner.py

"""
Bulk Backtest Runner - Multi-day, Multi-strategy Analysis

This runner reuses the single-day backtest logic from backtest_runner,
ensuring consistent behavior across all backtests.

Features:
1. Multi-day consecutive backtesting
2. Optional consecutive capital (Day 2 starts with Day 1's ending equity)
3. Multi-strategy comparison
4. Detailed logging per day
5. Summary reports (daily, monthly, yearly)

Usage:
    # Single strategy
    python -m src.runner.bulk_backtest_runner --strategy moderate --ticker TSLA --start 2024-12-01 --end 2024-12-31
    
    # With local data
    python -m src.runner.bulk_backtest_runner --strategy up_trend_aware --ticker SPLV --start 2024-12-01 --end 2024-12-31 --local-data --data-dir "/Users/vivi/vivi-llm-trader/data/"
    
    # Multiple strategies comparison
    python -m src.runner.bulk_backtest_runner --strategies moderate,up_trend_aware,mean_reversion --ticker TSLA --start 2024-12-01 --end 2024-12-31
"""

from datetime import datetime, timedelta
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import sys

# Reuse the core backtest logic
from src.runner.backtest_runner import run_backtest, DEFAULT_INITIAL_CAPITAL
from src.config.component_factory import StrategyRegistry


# ==========================================
# Helper Functions
# ==========================================

def get_trading_dates(start_date: str, end_date: str, trading_days_only: bool = True) -> List[str]:
    """Get list of trading dates."""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    dates = []
    current = start
    
    while current <= end:
        # Skip weekends
        if trading_days_only and current.weekday() >= 5:
            current += timedelta(days=1)
            continue
        
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    return dates


class OutputRedirector:
    """Context manager to redirect stdout to a file."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.original_stdout = None
        self.log_file = None
    
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.log_file = open(self.filepath, 'w', encoding='utf-8')
        sys.stdout = self.log_file
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        if self.log_file:
            self.log_file.close()
        return False


# ==========================================
# Single Day Wrapper (with logging)
# ==========================================

def run_single_day_with_logging(
    ticker: str,
    date_str: str,
    strategy_name: str,
    initial_capital: float,
    log_dir: Optional[str] = None,
    verbose: bool = False,
    use_local_data: bool = False,
    local_data_dir: str = "data/"
) -> Optional[Dict]:
    """
    Run backtest for a single day with optional log file output.
    
    This is a thin wrapper around run_backtest that adds:
    - Log file redirection
    - Consistent result formatting for bulk analysis
    
    Args:
        ticker: Stock ticker
        date_str: Date 'YYYY-MM-DD'
        strategy_name: Strategy key
        initial_capital: Starting capital
        log_dir: Directory for log files (None = no logging)
        verbose: Print detailed output
        use_local_data: If True, use local CSV files instead of Alpaca API
        local_data_dir: Directory containing CSV files
        
    Returns:
        Results dictionary or None on failure
    """
    
    # Determine if we should redirect to log file
    log_file_path = None
    if log_dir:
        log_file_path = str(Path(log_dir) / f"{date_str}_{strategy_name}.log")
    
    try:
        if log_file_path:
            # Run with output redirected to log file
            with OutputRedirector(log_file_path):
                result = run_backtest(
                    ticker=ticker,
                    strategy_name=strategy_name,
                    trading_date=date_str,
                    initial_capital=initial_capital,
                    enable_chart=False,  # No charts for bulk runs
                    auto_open_browser=False,
                    output_dir=log_dir,
                    verbose=verbose,
                    use_local_data=use_local_data,
                    local_data_dir=local_data_dir
                )
        else:
            # Run normally
            result = run_backtest(
                ticker=ticker,
                strategy_name=strategy_name,
                trading_date=date_str,
                initial_capital=initial_capital,
                enable_chart=False,
                auto_open_browser=False,
                verbose=verbose,
                use_local_data=use_local_data,
                local_data_dir=local_data_dir
            )
        
        if result is None:
            return None
        
        # Format result for bulk analysis
        return {
            'date': date_str,
            'ticker': ticker,
            'strategy': strategy_name,
            'initial_capital': initial_capital,
            'final_equity': result.get('final_equity', initial_capital),
            'pnl': result.get('pnl', 0),
            'pnl_pct': result.get('pnl_pct', 0),
            'total_trades': result.get('trades_executed', 0),
            'completed_trades': result.get('completed_trades', 0),
            'winning_trades': result.get('winning_trades', 0),
            'losing_trades': result.get('completed_trades', 0) - result.get('winning_trades', 0),
            'win_rate': result.get('win_rate', 0),
            'final_position': result.get('final_position', 0),
            'iterations': result.get('iterations', 0)
        }
        
    except Exception as e:
        # Log error
        if log_file_path:
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"\n\n❌ ERROR: {e}\n")
                import traceback
                traceback.print_exc(file=f)
        else:
            print(f"❌ Error running backtest for {date_str}: {e}")
        
        return None


# ==========================================
# Bulk Backtest
# ==========================================

def run_bulk_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    strategies: List[str],
    trading_days_only: bool = True,
    consecutive_capital: bool = True,
    output_dir: str = 'bulk_backtest_results',
    verbose: bool = False,
    use_local_data: bool = False,
    local_data_dir: str = "data/"
) -> pd.DataFrame:
    """
    Run bulk backtest across multiple dates and strategies.
    
    Uses run_backtest from backtest_runner for each day to ensure
    consistent behavior with single-day backtests.
    
    Args:
        ticker: Stock ticker
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        strategies: List of strategy keys
        trading_days_only: Skip weekends
        consecutive_capital: Use previous day's ending equity as next day's starting capital
        output_dir: Output directory
        verbose: Print detailed output per day
        use_local_data: If True, use local CSV files instead of Alpaca API
        local_data_dir: Directory containing CSV files
        
    Returns:
        DataFrame with all results
    """
    # Create directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get dates
    dates = get_trading_dates(start_date, end_date, trading_days_only)
    
    print(f"\n{'='*60}")
    print(f"🚀 Bulk Backtest")
    print(f"{'='*60}")
    print(f"   Ticker: {ticker}")
    print(f"   Dates: {start_date} to {end_date}")
    print(f"   Trading Days: {len(dates)}")
    print(f"   Strategies: {', '.join(strategies)}")
    print(f"   Consecutive Capital: {'Yes' if consecutive_capital else 'No'}")
    print(f"   Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Run backtests
    all_results = []
    total_runs = len(dates) * len(strategies)
    current_run = 0
    
    for strategy_name in strategies:
        try:
            strategy_info = StrategyRegistry.get_info(strategy_name)
            print(f"\n📊 Strategy: {strategy_info.name}")
        except ValueError:
            print(f"\n📊 Strategy: {strategy_name}")
        print(f"{'='*60}")
        
        # Track capital for this strategy
        current_capital = DEFAULT_INITIAL_CAPITAL
        
        for date_str in dates:
            current_run += 1
            progress = current_run / total_runs * 100
            
            print(f"[{progress:5.1f}%] {date_str} - {strategy_name} (${current_capital:,.0f})...",
                  end=' ', flush=True)
            
            # Run single day backtest using shared logic
            result = run_single_day_with_logging(
                ticker=ticker,
                date_str=date_str,
                strategy_name=strategy_name,
                initial_capital=current_capital,
                log_dir=str(log_dir),
                verbose=verbose,
                use_local_data=use_local_data,
                local_data_dir=local_data_dir
            )
            
            if result:
                all_results.append(result)
                status = "✅" if result['pnl'] >= 0 else "❌"
                print(f"{status} PnL: ${result['pnl']:+.2f} ({result['pnl_pct']:+.2f}%) → ${result['final_equity']:,.0f}")
                
                # Update capital for next day
                if consecutive_capital:
                    current_capital = result['final_equity']
            else:
                print("⚠️ Skipped (no data or error)")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save results
    if not df.empty:
        df.to_csv(f"{output_dir}/daily_results.csv", index=False)
        print(f"\n✅ Daily results saved: {output_dir}/daily_results.csv")
        print(f"✅ Log files saved: {log_dir}/ ({len(all_results)} files)")
        
        # Print final summary
        if consecutive_capital:
            print(f"\n💰 Final Capital Summary:")
            for strategy in df['strategy'].unique():
                strategy_df = df[df['strategy'] == strategy]
                if not strategy_df.empty:
                    initial = strategy_df.iloc[0]['initial_capital']
                    final = strategy_df.iloc[-1]['final_equity']
                    total_return = final - initial
                    total_return_pct = (total_return / initial * 100) if initial > 0 else 0
                    print(f"   {strategy:20s}: ${initial:,.2f} → ${final:,.2f} "
                          f"(${total_return:+,.2f}, {total_return_pct:+.2f}%)")
    
    return df


# ==========================================
# Summary Report Generation
# ==========================================

def generate_summary_reports(df: pd.DataFrame, output_dir: str):
    """Generate summary reports from bulk backtest results."""
    
    if df.empty:
        print("⚠️ No data for summary reports")
        return
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Add time dimensions
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['year_month'] = df['date'].dt.to_period('M')
    df['year_quarter'] = df['date'].dt.to_period('Q')
    
    print(f"\n{'='*60}")
    print(f"📊 Generating Summary Reports")
    print(f"{'='*60}")
    
    # Monthly summary
    monthly = df.groupby(['year_month', 'strategy']).agg({
        'pnl': 'sum',
        'pnl_pct': 'mean',
        'total_trades': 'sum',
        'win_rate': 'mean'
    }).reset_index()
    monthly['year_month'] = monthly['year_month'].astype(str)
    monthly.to_csv(f"{output_dir}/monthly_summary.csv", index=False)
    print(f"✅ Monthly summary: {output_dir}/monthly_summary.csv")
    
    # Strategy comparison
    strategy_comp = df.groupby('strategy').agg({
        'pnl': ['sum', 'mean', 'std', 'min', 'max'],
        'pnl_pct': ['mean', 'std'],
        'total_trades': 'sum',
        'win_rate': 'mean'
    }).reset_index()
    strategy_comp.columns = ['_'.join(col).strip('_') for col in strategy_comp.columns.values]
    strategy_comp.to_csv(f"{output_dir}/strategy_comparison.csv", index=False)
    print(f"✅ Strategy comparison: {output_dir}/strategy_comparison.csv")
    
    # Print strategy comparison
    print(f"\n🏆 Strategy Comparison:")
    print("-"*60)
    for _, row in strategy_comp.iterrows():
        print(f"\n{row['strategy']}:")
        print(f"   Total PnL: ${row['pnl_sum']:,.2f}")
        print(f"   Avg Daily PnL: ${row['pnl_mean']:,.2f} (±${row['pnl_std']:.2f})")
        print(f"   Best Day: ${row['pnl_max']:,.2f}")
        print(f"   Worst Day: ${row['pnl_min']:,.2f}")
        print(f"   Win Rate: {row['win_rate_mean']*100:.1f}%")


# ==========================================
# Main
# ==========================================

def main():
    """Main entry point with CLI argument parsing."""
    
    parser = argparse.ArgumentParser(
        description='Bulk Backtest Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.runner.bulk_backtest_runner --strategy moderate --ticker TSLA --start 2024-12-01 --end 2024-12-31
    python -m src.runner.bulk_backtest_runner --strategies moderate,up_trend_aware --ticker TSLA --start 2024-12-01 --end 2024-12-31
    python -m src.runner.bulk_backtest_runner --strategy moderate --no-consecutive-capital

Available Strategies:
""" + "\n".join([f"    {k}: {v}" for k, v in StrategyRegistry.list_strategies().items()])
    )
    
    parser.add_argument(
        '--ticker', '-t',
        type=str,
        default='TSLA',
        help='Stock ticker (default: TSLA)'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        required=True,
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        default=None,
        help='Single strategy to test'
    )
    
    parser.add_argument(
        '--strategies',
        type=str,
        default=None,
        help='Comma-separated list of strategies'
    )
    
    parser.add_argument(
        '--trading-days-only',
        action='store_true',
        default=True,
        help='Skip weekends (default: True)'
    )
    
    parser.add_argument(
        '--include-weekends',
        action='store_true',
        help='Include weekends in backtest dates'
    )
    
    parser.add_argument(
        '--no-consecutive-capital',
        action='store_true',
        help='Reset capital each day (don\'t use consecutive)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='bulk_backtest_results',
        help='Output directory (default: bulk_backtest_results)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output for each day'
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
    
    # Determine strategies
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(',')]
    elif args.strategy:
        strategies = [args.strategy]
    else:
        strategies = ['up_trend_aware']  # Default
    
    # Validate strategies
    available = StrategyRegistry.get_all_keys()
    for s in strategies:
        if s not in available:
            print(f"❌ Unknown strategy: {s}")
            print(f"   Available: {', '.join(available)}")
            return
    
    # Determine trading days setting
    trading_days_only = not args.include_weekends
    
    # Run bulk backtest
    df = run_bulk_backtest(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        strategies=strategies,
        trading_days_only=trading_days_only,
        consecutive_capital=not args.no_consecutive_capital,
        output_dir=args.output_dir,
        verbose=args.verbose,
        use_local_data=args.local_data,
        local_data_dir=args.data_dir
    )
    
    # Generate summaries
    if not df.empty:
        generate_summary_reports(df, args.output_dir)
        
        print(f"\n{'='*60}")
        print(f"✅ Bulk Backtest Complete!")
        print(f"   Results in: {args.output_dir}/")
        print(f"{'='*60}\n")
    else:
        print("\n❌ No valid results")


if __name__ == '__main__':
    main()