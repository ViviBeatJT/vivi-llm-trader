# backtest_with_chart_multi_strategy_improved.py

"""
带图表的多策略回测运行器（改进版）

✨ 改进点：
1. 强制收盘时间检查（15:55）
2. 循环结束后的最终持仓验证
3. 确保16:00前持仓归零
4. 详细的时间窗口日志

支持策略：
1. conservative - 原始保守策略
2. moderate - 温和进取策略（推荐）
3. high_freq - 高频交易策略
4. ultra - 超激进策略

用法：
    python backtest_with_chart_multi_strategy_improved.py --strategy moderate
"""

from datetime import datetime, timezone, timedelta, time as dt_time
import os
from dotenv import load_dotenv
import pytz
import argparse

# --- Core Modules ---
from src.cache.trading_cache import TradingCache
from src.manager.position_manager import PositionManager
from src.data_fetcher.alpaca_data_fetcher import AlpacaDataFetcher
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# --- Simple Chart Visualizer ---
from src.visualization.simple_chart_visualizer import SimpleChartVisualizer

# --- Executors ---
from src.executor.simulation_executor import SimulationExecutor

# --- 所有策略 ---
from src.strategies.aggressive_mean_reversion_strategy import AggressiveMeanReversionStrategy
from src.strategies.moderate_aggressive_strategy import ModerateAggressiveStrategy
from src.strategies.high_frequency_strategy import HighFrequencyStrategy
from src.strategies.ultra_aggressive_strategy import UltraAggressiveStrategy
from src.strategies.trend_aware_strategy import TrendAwareStrategy

load_dotenv()

US_EASTERN = pytz.timezone('America/New_York')


# ==================== 策略配置 ====================

STRATEGY_CONFIGS = {
    'trend_aware': {
        'class': TrendAwareStrategy,
        'name': '趋势感知策略',
    'params': {
        'bb_period': 20,
        'bb_std_dev': 2.0,
        'adx_period': 14,
        'adx_trend_threshold': 25,
        'adx_range_threshold': 20,
        'ema_fast_period': 12,
        'ema_slow_period': 26,
        'mean_reversion_entry': 0.85,
        'mean_reversion_exit': 0.60,
        'trend_entry_pullback': 0.50,
        'trend_exit_profit': 0.03,
        'stop_loss_threshold': 0.01,  # ✨ 改为 1%
        'monitor_interval_seconds': 60,
        'max_history_bars': 500
    },
        'chart_file': 'backtest_trend_aware.html',
        'description': '接近布林带就交易，捕捉更多机会,TREND AWARE'
    },
    'conservative': {
        'class': AggressiveMeanReversionStrategy,
        'name': '原始保守策略',
        'params': {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'stop_loss_threshold': 0.10,
            'monitor_interval_seconds': 60,
        },
        'chart_file': 'backtest_conservative.html',
        'description': '只在完全突破布林带时交易'
    },
    'moderate': {
        'class': ModerateAggressiveStrategy,
        'name': '温和进取策略',
        'params': {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'entry_threshold': 0.85,      # 85% 开仓
            'exit_threshold': 0.60,       # 60% 平仓
            'stop_loss_threshold': 0.10,
            'monitor_interval_seconds': 60,
        },
        'chart_file': 'backtest_moderate.html',
        'description': '接近布林带就交易，捕捉更多机会'
    },
    'high_freq': {
        'class': HighFrequencyStrategy,
        'name': '高频交易策略',
        'params': {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'strong_entry': 0.90,
            'mild_entry': 0.75,
            'exit_threshold': 0.65,
            'stop_loss_threshold': 0.08,
            'monitor_interval_seconds': 60,
        },
        'chart_file': 'backtest_high_freq.html',
        'description': '在布林带内部也交易'
    },
    'ultra': {
        'class': UltraAggressiveStrategy,
        'name': '超激进动态策略',
        'params': {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'min_entry_threshold': 0.70,
            'max_entry_threshold': 0.90,
            'quick_exit_threshold': 0.55,
            'stop_loss_threshold': 0.06,
            'take_profit_threshold': 0.03,
            'monitor_interval_seconds': 60,
        },
        'chart_file': 'backtest_ultra.html',
        'description': '动态调整，快速止盈止损'
    }
}


# ==================== 回测配置 ====================

# 基本设置
TICKER = "TSLA"
TRADING_DATE = "2025-12-05"

# 回测设置
STEP_SECONDS = 30          # 每1分钟监控一次
LOOKBACK_MINUTES = 300    # 回看300分钟（5小时）

# ✨ 关键时间点（东部时间）
LAST_ENTRY_TIME = dt_time(15, 50)   # 最后开仓时间
FORCE_CLOSE_TIME = dt_time(15, 55)  # 强制平仓时间
MARKET_CLOSE_TIME = dt_time(16, 0)  # 市场收盘时间

# 交易设置
INITIAL_CAPITAL = 1000.0
SHARES_PER_TRADE = 1
COMMISSION_PER_TRADE = 0.0

# 图表设置
AUTO_OPEN_BROWSER = True


def create_strategy(strategy_name: str):
    """创建策略实例"""
    if strategy_name not in STRATEGY_CONFIGS:
        raise ValueError(
            f"未知策略: {strategy_name}. 可选: {list(STRATEGY_CONFIGS.keys())}")

    config = STRATEGY_CONFIGS[strategy_name]
    strategy_class = config['class']
    params = config['params']

    print(f"\n📊 策略: {config['name']}")
    print(f"   描述: {config['description']}")
    print(f"   参数:")
    for key, value in params.items():
        if isinstance(value, float):
            print(f"      {key}: {value:.2f}")
        else:
            print(f"      {key}: {value}")

    return strategy_class(**params)


def run_backtest(strategy_name: str = 'moderate'):
    """运行带图表的回测（改进版）"""

    print("\n" + "="*70)
    print(f"🚀 带图表的回测（改进版） - {TICKER}")
    print("="*70)

    strategy_config = STRATEGY_CONFIGS[strategy_name]
    chart_file = strategy_config['chart_file']

    print(f"\n📅 回测配置:")
    print(f"   策略: {strategy_config['name']}")
    print(f"   股票: {TICKER}")
    print(f"   日期: {TRADING_DATE}")
    print(f"   步进: {STEP_SECONDS} 秒")
    print(f"   初始资金: ${INITIAL_CAPITAL:,.0f}")

    print(f"\n⏰ 关键时间点（东部时间）:")
    print(f"   最后开仓: {LAST_ENTRY_TIME}")
    print(f"   强制平仓: {FORCE_CLOSE_TIME}")
    print(f"   市场收盘: {MARKET_CLOSE_TIME}")

    print(f"\n📊 图表配置:")
    print(f"   文件: {chart_file}")
    print(f"   自动打开: {'是' if AUTO_OPEN_BROWSER else '否'}")

    # 1. 初始化组件
    print(f"\n🔧 初始化组件...")

    # 财务参数
    FINANCE_PARAMS = {
        'INITIAL_CAPITAL': INITIAL_CAPITAL,
        'COMMISSION_RATE': 0.0003,
        'SLIPPAGE_RATE': 0.0001,
        'MIN_LOT_SIZE': SHARES_PER_TRADE,
        'MAX_ALLOCATION': 0.95,
    }

    cache = TradingCache()
    data_fetcher = AlpacaDataFetcher()
    executor = SimulationExecutor(FINANCE_PARAMS)
    position_manager = PositionManager(executor, FINANCE_PARAMS)

    # 2. 创建策略
    strategy = create_strategy(strategy_name)

    # 3. 初始化图表
    print(f"\n📊 初始化图表可视化...")
    visualizer = SimpleChartVisualizer(
        ticker=TICKER,
        output_file=chart_file,
        auto_open=AUTO_OPEN_BROWSER
    )
    visualizer.set_initial_capital(INITIAL_CAPITAL)

    # 4. 获取初始时间范围
    print(f"\n⏱️ 设置回测时间...")

    date_parts = [int(x) for x in TRADING_DATE.split('-')]

    # 市场时间: 9:30 - 16:00 ET
    start_time = US_EASTERN.localize(
        datetime(date_parts[0], date_parts[1], date_parts[2], 9, 30))
    end_time = US_EASTERN.localize(
        datetime(date_parts[0], date_parts[1], date_parts[2], 16, 0))

    # 转换为 UTC
    start_time = start_time.astimezone(timezone.utc)
    end_time = end_time.astimezone(timezone.utc)

    print(f"   开始: {start_time.strftime('%Y-%m-%d %H:%M')} UTC (9:30 ET)")
    print(f"   结束: {end_time.strftime('%Y-%m-%d %H:%M')} UTC (16:00 ET)")
    print(f"   步进: {STEP_SECONDS} 秒")

    # 5. 回测循环
    print(f"\n🏃 开始回测...")
    print(f"="*70)

    current_time = start_time
    iteration = 0
    update_count = 0

    # ✨ 追踪关键时间点
    last_entry_reached = False
    force_close_reached = False

    try:
        while current_time <= end_time:
            iteration += 1

            # 确保时区
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=timezone.utc)

            # 转换为东部时间
            current_et = current_time.astimezone(US_EASTERN)
            current_et_time = current_et.time()

            # ✨ 检测关键时间点
            if not last_entry_reached and current_et_time >= LAST_ENTRY_TIME:
                print(f"\n⏰ 到达最后开仓时间: {current_et.strftime('%H:%M')} ET")
                last_entry_reached = True

            if not force_close_reached and current_et_time >= FORCE_CLOSE_TIME:
                print(f"\n🔔 到达强制平仓时间: {current_et.strftime('%H:%M')} ET")
                force_close_reached = True

            # 获取数据
            df = data_fetcher.get_latest_bars(
                ticker=TICKER,
                lookback_minutes=LOOKBACK_MINUTES,
                end_dt=current_time,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute)
            )

            if df.empty:
                current_time += timedelta(seconds=STEP_SECONDS)
                continue

            current_price = df.iloc[-1]['close']

            # 获取当前账户状态
            account_status = position_manager.get_account_status(current_price)
            current_position = account_status.get('position', 0.0)
            avg_cost = account_status.get('avg_cost', 0.0)
            current_equity = account_status.get('equity', INITIAL_CAPITAL)

            # ✨ 判断是否需要强制平仓
            is_force_close = current_et_time >= FORCE_CLOSE_TIME

            # 获取信号
            try:
                signal_data, _ = strategy.get_signal(
                    ticker=TICKER,
                    new_data=df,
                    current_position=current_position,
                    avg_cost=avg_cost,
                    verbose=False,
                    is_market_close=is_force_close,  # ✨ 15:55后强制平仓
                    current_time_et=current_et       # ✨ 传入时间用于检查
                )

                signal = signal_data['signal']

                # 执行交易
                if signal in ['BUY', 'SELL', 'SHORT', 'COVER']:
                    emoji = {"BUY": "🟢", "SELL": "🔴",
                             "SHORT": "🔻", "COVER": "🔺"}
                    print(f"\n{emoji.get(signal, '⚪')} {current_et.strftime('%H:%M')} ET | "
                          f"{signal} @ ${current_price:.2f}")
                    print(f"   {signal_data.get('reason', 'N/A')}")

                    # 执行交易
                    position_manager.execute_and_update(
                        timestamp=current_time,
                        signal=signal,
                        current_price=current_price,
                        ticker=TICKER
                    )

            except Exception as e:
                print(f"❌ 策略错误: {e}")
                current_time += timedelta(seconds=STEP_SECONDS)
                continue

            # 更新图表
            strategy_df = strategy.get_history_data(TICKER)
            trade_log = position_manager.get_trade_log()

            if not strategy_df.empty:
                visualizer.update_data(
                    market_data=strategy_df,
                    trade_log=trade_log,
                    current_equity=current_equity,
                    current_position=current_position,
                    timestamp=current_time
                )
                update_count += 1

            # 进度显示
            if iteration % 10 == 0:
                progress = (current_time - start_time) / \
                    (end_time - start_time) * 100
                print(f"\n📊 进度: {progress:.1f}% | 时间: {current_et.strftime('%H:%M')} ET | "
                      f"权益: ${current_equity:,.0f} | 持仓: {current_position}")

            # 前进1分钟
            current_time += timedelta(seconds=STEP_SECONDS)

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断回测")

    # ===== ✨ 最终持仓检查（新增）=====
    print(f"\n" + "="*70)
    print(f"🔍 最终持仓检查")
    print("="*70)

    # 获取最终数据和价格
    df_final = data_fetcher.get_latest_bars(
        ticker=TICKER,
        lookback_minutes=LOOKBACK_MINUTES,
        end_dt=end_time,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute)
    )

    if not df_final.empty:
        final_price = df_final.iloc[-1]['close']
    else:
        final_price = account_status.get('last_price', 0.0)

    # 获取最终持仓状态
    final_status = position_manager.get_account_status(final_price)
    final_position = final_status.get('position', 0.0)

    print(
        f"   最终时间: {end_time.astimezone(US_EASTERN).strftime('%Y-%m-%d %H:%M')} ET")
    print(f"   最终价格: ${final_price:.2f}")
    print(f"   最终持仓: {final_position} 股")

    # ✨ 如果还有持仓，强制平仓！
    if final_position != 0:
        print(f"\n⚠️  检测到未平仓位！")
        print(f"   持仓: {final_position} 股")
        print(f"   执行强制平仓...")

        close_signal = 'SELL' if final_position > 0 else 'COVER'

        try:
            position_manager.execute_and_update(
                timestamp=end_time,
                signal=close_signal,
                current_price=final_price,
                ticker=TICKER
            )

            # 重新获取状态
            final_status = position_manager.get_account_status(final_price)
            final_position = final_status.get('position', 0.0)

            print(f"   ✅ 强制平仓完成")
            print(f"   最终持仓: {final_position} 股")

            if final_position != 0:
                print(f"   ❌ 警告：平仓后仍有持仓 {final_position} 股！")

        except Exception as e:
            print(f"   ❌ 强制平仓失败: {e}")
    else:
        print(f"   ✅ 持仓已归零")

    # 最终结果
    print(f"\n" + "="*70)
    print(f"📊 回测结果 - {strategy_config['name']}")
    print("="*70)

    trade_log = position_manager.get_trade_log()

    final_equity = final_status.get('equity', INITIAL_CAPITAL)
    total_pnl = final_status.get('total_pnl', 0)
    total_pnl_pct = final_status.get('total_pnl_pct', 0)

    print(f"\n💰 资金情况:")
    print(f"   初始资金: ${INITIAL_CAPITAL:,.2f}")
    print(f"   最终权益: ${final_equity:,.2f}")
    print(f"   盈亏: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)")
    print(f"   现金: ${final_status.get('cash', 0):,.2f}")
    print(
        f"   持仓: {final_status.get('position', 0)} 股 {'✅' if final_status.get('position', 0) == 0 else '❌'}")

    print(f"\n📈 交易统计:")

    if not trade_log.empty:
        print(f"   总交易数: {len(trade_log)}")

        if 'type' in trade_log.columns:
            completed_trades = trade_log[trade_log['type'].isin(
                ['SELL', 'COVER'])]
            if not completed_trades.empty and 'net_pnl' in completed_trades.columns:
                winning_trades = completed_trades[completed_trades['net_pnl'] > 0]
                win_rate = len(winning_trades) / len(completed_trades) * 100
                print(f"   完成交易: {len(completed_trades)}")
                print(f"   胜率: {win_rate:.1f}%")

                if len(winning_trades) > 0:
                    print(f"   平均盈利: ${winning_trades['net_pnl'].mean():.2f}")
                losing_trades = completed_trades[completed_trades['net_pnl'] < 0]
                if len(losing_trades) > 0:
                    print(f"   平均亏损: ${losing_trades['net_pnl'].mean():.2f}")
    else:
        print(f"   总交易数: 0")

    print(f"\n📊 图表:")
    print(f"   文件: {chart_file}")
    print(f"   更新: {update_count} 次")

    print(f"\n⏰ 时间窗口检查:")
    print(f"   最后开仓时间触发: {'✅' if last_entry_reached else '❌'}")
    print(f"   强制平仓时间触发: {'✅' if force_close_reached else '❌'}")

    print(f"\n" + "="*70)
    print(f"✅ 回测完成！查看图表: {chart_file}")
    print("="*70 + "\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='带图表的多策略回测（改进版）')

    parser.add_argument('--strategy', type=str, default='moderate',
                        choices=list(STRATEGY_CONFIGS.keys()),
                        help='选择策略 (conservative/moderate/high_freq/ultra/trend_aware)')

    parser.add_argument('--ticker', type=str, default=None,
                        help='股票代码 (默认: TSLA)')

    parser.add_argument('--date', type=str, default=None,
                        help='交易日期 (YYYY-MM-DD, 默认: 2024-12-05)')

    args = parser.parse_args()

    # 更新全局配置
    global TICKER, TRADING_DATE
    if args.ticker:
        TICKER = args.ticker
    if args.date:
        TRADING_DATE = args.date

    run_backtest(strategy_name=args.strategy)


if __name__ == '__main__':
    main()
