# src/runner/live_runner.py

"""
Live Trading Runner - 使用集中配置系统

极简版本，所有复杂逻辑都在 TradingEngine 中。

使用方式:
    # 命令行
    python -m src.runner.live_runner --ticker TSLA --mode paper --strategy up_trend_aware --finance-preset small --monitor_frequency fast
    python -m src.runner.live_runner --ticker AAPL --preset conservative --capital 5000
    
    # Python 代码
    from src.runner.live_runner import run_live
    from src.config.trading_config import quick_config
    
    config = quick_config(capital=5000, ticker='AAPL')
    result = run_live(config)
"""

from datetime import datetime, timezone
import argparse
from pathlib import Path
from typing import Optional

from src.config.trading_config import (
    TradingConfig,
    get_full_config,
    quick_config,
)
from src.engine.trading_engine import TradingEngine
from src.config.component_factory import TradingMode


# ==========================================
# Main Runner
# ==========================================

def run_live(config: TradingConfig) -> dict:
    """
    使用配置对象运行实盘交易

    Args:
        config: TradingConfig 配置对象

    Returns:
        交易结果字典
    """
    # 打印配置摘要
    print(config.summary())

    # Live 模式警告
    if config.system.mode == 'live':
        print("\n" + "⚠️" * 20)
        print("   WARNING: LIVE TRADING MODE!")
        print("   All trades will use REAL MONEY!")
        print("⚠️" * 20)

        confirm = input("\nConfirm live trading? (type 'YES' to confirm): ")
        if confirm != 'YES':
            print("Cancelled.")
            return {}

    # 创建输出目录
    Path(config.system.output_dir).mkdir(parents=True, exist_ok=True)

    # ==========================================
    # 创建 Engine (自动创建所有组件)
    # ==========================================

    engine = TradingEngine.from_config(config)

    # API 仓位同步
    if config.system.sync_position_on_start and config.system.mode in ['paper', 'live']:
        print(f"\n🔄 Syncing position from API for {config.system.ticker}...")
        sync_success = engine.position_manager.sync_from_api(
            config.system.ticker)
        if not sync_success:
            print("⚠️ Position sync failed, using local initial state")

    # ==========================================
    # Run
    # ==========================================

    try:
        report = engine.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        report = engine._generate_report(datetime.now(timezone.utc))

    # Print report
    engine.print_report(report)

    # Print trade log
    _print_trade_log(engine.position_manager.get_trade_log())

    # Final summary
    _print_final_summary(report, engine.visualizer)

    return report


def _print_trade_log(trade_log):
    """Print trade log."""
    if trade_log is not None and not trade_log.empty:
        print("\n📝 Trade Log:")
        display_log = trade_log[['time', 'type',
                                 'qty', 'price', 'fee', 'net_pnl']].copy()
        display_log['time'] = display_log['time'].dt.strftime('%Y-%m-%d %H:%M')

        try:
            print(display_log.to_markdown(index=False, floatfmt=".2f"))
        except Exception:
            print(display_log.to_string(index=False))
    else:
        print("\n🤷 No trades executed.")


def _print_final_summary(report: dict, visualizer):
    """Print final summary."""
    print(f"\n" + "=" * 70)
    print(f"✅ Live Trading Complete!")
    print("=" * 70)

    if visualizer:
        print(f"   📊 Chart: {visualizer.output_file}")

    print(f"   💰 Final Equity: ${report.get('final_equity', 0):,.2f}")
    print(
        f"   📈 PnL: ${report.get('pnl', 0):,.2f} ({report.get('pnl_pct', 0):+.2f}%)")
    print("=" * 70 + "\n")


# ==========================================
# CLI Entry Point
# ==========================================

def main():
    """命令行入口"""

    parser = argparse.ArgumentParser(
        description='Live Trading Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 基本使用
    python -m src.runner.live_runner --ticker TSLA --mode paper --finance-preset small
    
    # 使用预设
    python -m src.runner.live_runner --ticker AAPL --preset conservative --capital 5000
    
    # 自定义参数
    python -m src.runner.live_runner --ticker NVDA --stop-loss 0.015 --take-profit 0.025

Presets:
    Finance: small ($1k), medium ($5k), large ($25k), paper ($100k)
    Strategy: conservative, moderate, aggressive
    
Modes:
    simulation - Local simulation (no API)
    paper      - Alpaca paper trading
    live       - Real money trading ⚠️
"""
    )

    # 基本参数
    parser.add_argument('--ticker', '-t', type=str, default='TSLA',
                        help='Stock ticker (default: TSLA)')
    parser.add_argument('--mode', '-m', type=str, default='paper',
                        choices=['simulation', 'paper', 'live'],
                        help='Trading mode (default: paper)')
    parser.add_argument('--strategy', '-s', type=str, default='simple_trend',
                        choices=['up_trend_aware', 'trend_aware',
                                 'moderate', 'mean_reversion'],
                        help='Strategy name (default: simple_trend)')

    # 资金参数
    parser.add_argument('--capital', '-c', type=float, default=None,
                        help='Initial capital')
    parser.add_argument('--finance-preset', type=str, default=None,
                        choices=['small', 'medium', 'large', 'paper'],
                        help='Finance preset')

    # 策略参数
    parser.add_argument('--preset', '-p', type=str, default=None,
                        choices=['conservative', 'moderate', 'aggressive'],
                        help='Strategy preset')
    parser.add_argument('--monitor_frequency', type=str, default=None,
                        choices=['fast', 'medium', 'slow'],
                        help='How frequency monitor the stock')
    parser.add_argument('--stop-loss', type=float, default=None,
                        help='Stop loss percentage (e.g., 0.02 for 2%%)')
    parser.add_argument('--take-profit', type=float, default=None,
                        help='Take profit percentage (e.g., 0.03 for 3%%)')

    # 数据参数
    parser.add_argument('--interval', '-i', type=int, default=None,
                        help='Update interval in seconds (default: 30)')
    parser.add_argument('--lookback', type=int, default=None,
                        help='Data lookback in minutes (default: 300)')

    # 运行参数
    parser.add_argument('--max-runtime', type=int, default=None,
                        help='Maximum runtime in minutes')
    parser.add_argument('--no-chart', action='store_true',
                        help='Disable chart generation')
    parser.add_argument('--no-browser', action='store_true',
                        help='Don\'t auto-open chart in browser')
    parser.add_argument('--no-sync', action='store_true',
                        help='Don\'t sync position from API')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    # ==========================================
    # 构建配置
    # ==========================================

    config = get_full_config(
        ticker=args.ticker,
        strategy=args.strategy,
        mode=args.mode,
        finance_preset=args.finance_preset,
        strategy_preset=args.preset,
        monitor_frequency=args.monitor_frequency
    )

    # 覆盖参数
    if args.capital is not None:
        config.finance.initial_capital = args.capital

    if args.stop_loss is not None:
        config.simple_trend.normal_stop_loss = args.stop_loss
        config.trend_aware.stop_loss_threshold = args.stop_loss

    if args.take_profit is not None:
        config.simple_trend.uptrend_take_profit = args.take_profit
        config.trend_aware.trend_exit_profit = args.take_profit

    if args.interval is not None:
        config.data.step_seconds = args.interval

    if args.lookback is not None:
        config.data.lookback_minutes = args.lookback

    if args.max_runtime is not None:
        config.system.max_runtime_minutes = args.max_runtime

    if args.no_chart:
        config.system.enable_chart = False

    if args.no_browser:
        config.system.auto_open_browser = False

    if args.no_sync:
        config.system.sync_position_on_start = False

    if args.output_dir is not None:
        config.system.output_dir = args.output_dir

    if args.quiet:
        config.system.verbose = False

    # ==========================================
    # 运行
    # ==========================================

    run_live(config)


if __name__ == '__main__':
    main()
