# backtest_with_chart_multi_strategy.py

"""
带图表的多策略回测运行器

支持策略：
1. conservative - 原始保守策略
2. moderate - 温和进取策略（推荐）
3. high_freq - 高频交易策略
4. ultra - 超激进策略

用法：
    python backtest_with_chart_multi_strategy.py --strategy moderate
    
特点：
- 命令行选择策略
- 实时图表更新
- 蜡烛图 + 布林带
- 交易标记
"""

from datetime import datetime, timezone, timedelta
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

load_dotenv()

US_EASTERN = pytz.timezone('America/New_York')


# ==================== 策略配置 ====================

STRATEGY_CONFIGS = {
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
TRADING_DATE = "2024-12-05"

# 回测设置（与原来保持一致）
STEP_MINUTES = 1          # 每1分钟监控一次
LOOKBACK_MINUTES = 120    # 每次获取过去120分钟的5分钟K线

# 交易设置
INITIAL_CAPITAL = 100000.0
SHARES_PER_TRADE = 50
COMMISSION_PER_TRADE = 1.0

# 图表设置
AUTO_OPEN_BROWSER = True


def create_strategy(strategy_name: str):
    """创建策略实例"""
    if strategy_name not in STRATEGY_CONFIGS:
        raise ValueError(f"未知策略: {strategy_name}. 可选: {list(STRATEGY_CONFIGS.keys())}")
    
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
    """运行带图表的回测"""
    
    print("\n" + "="*70)
    print(f"🚀 带图表的回测 - {TICKER}")
    print("="*70)
    
    strategy_config = STRATEGY_CONFIGS[strategy_name]
    chart_file = strategy_config['chart_file']
    
    print(f"\n📅 回测配置:")
    print(f"   策略: {strategy_config['name']}")
    print(f"   股票: {TICKER}")
    print(f"   日期: {TRADING_DATE}")
    print(f"   步进: {STEP_MINUTES} 分钟")
    print(f"   初始资金: ${INITIAL_CAPITAL:,.0f}")
    
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
        'MAX_ALLOCATION': 0.2,
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
    
    # 解析日期并设置时间范围
    from datetime import datetime, time as dt_time
    import pytz
    
    US_EASTERN = pytz.timezone('America/New_York')
    date_parts = [int(x) for x in TRADING_DATE.split('-')]
    
    # 市场时间: 9:30 - 16:00 ET
    start_time = US_EASTERN.localize(datetime(date_parts[0], date_parts[1], date_parts[2], 9, 30))
    end_time = US_EASTERN.localize(datetime(date_parts[0], date_parts[1], date_parts[2], 16, 0))
    
    # 转换为 UTC
    start_time = start_time.astimezone(timezone.utc)
    end_time = end_time.astimezone(timezone.utc)
    
    print(f"   开始: {start_time.strftime('%Y-%m-%d %H:%M')} UTC (9:30 ET)")
    print(f"   结束: {end_time.strftime('%Y-%m-%d %H:%M')} UTC (16:00 ET)")
    print(f"   步进: {STEP_MINUTES} 分钟")
    print(f"   回看: {LOOKBACK_MINUTES} 分钟（5分钟K线）")
    
    # 5. 回测循环（时间驱动）
    print(f"\n🏃 开始回测...")
    print(f"   策略: {strategy_config['name']}")
    print(f"   每 {STEP_MINUTES} 分钟监控一次")
    print(f"   每次获取过去 {LOOKBACK_MINUTES} 分钟的5分钟K线")
    print(f"="*70)
    
    current_time = start_time
    iteration = 0
    update_count = 0
    
    try:
        while current_time <= end_time:
            iteration += 1
            
            # 确保时区
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=timezone.utc)
            
            # 获取截至当前时间的数据（过去120分钟的5分钟K线）
            df = data_fetcher.get_latest_bars(
                ticker=TICKER,
                lookback_minutes=LOOKBACK_MINUTES,
                end_dt=current_time,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute)
            )
            
            if df.empty:
                current_time += timedelta(minutes=STEP_MINUTES)
                continue
            
            current_price = df.iloc[-1]['close']
            
            # 获取当前账户状态
            account_status = position_manager.get_account_status(current_price)
            current_position = account_status.get('position', 0.0)
            avg_cost = account_status.get('avg_cost', 0.0)
            current_equity = account_status.get('equity', INITIAL_CAPITAL)
            
            # 获取信号
            try:
                signal_data, _ = strategy.get_signal(
                    ticker=TICKER,
                    new_data=df,
                    current_position=current_position,
                    avg_cost=avg_cost,
                    verbose=False
                )
                
                signal = signal_data['signal']
                
                # 执行交易
                if signal in ['BUY', 'SELL', 'SHORT', 'COVER']:
                    emoji = {"BUY": "🟢", "SELL": "🔴", "SHORT": "🔻", "COVER": "🔺"}
                    print(f"\n{emoji.get(signal, '⚪')} {current_time.strftime('%H:%M')} | "
                          f"{signal} @ ${current_price:.2f}")
                    print(f"   {signal_data.get('reason', 'N/A')}")
                    
                    # 使用 position_manager 的方法执行交易
                    position_manager.execute_and_update(
                        timestamp=current_time,
                        signal=signal,
                        current_price=current_price,
                        ticker=TICKER
                    )
            
            except Exception as e:
                print(f"❌ 策略错误: {e}")
                current_time += timedelta(minutes=STEP_MINUTES)
                continue
            
            # 更新图表
            strategy_df = strategy.get_history_data(TICKER)
            trade_log = position_manager.get_trade_log()
            
            if not strategy_df.empty:
                # 首次更新检查数据
                if update_count == 0:
                    print(f"\n🔍 策略数据诊断:")
                    print(f"   数据行数: {len(strategy_df)}")
                    
                    bb_cols = ['SMA', 'BB_UPPER', 'BB_LOWER']
                    for col in bb_cols:
                        if col in strategy_df.columns:
                            valid_count = strategy_df[col].notna().sum()
                            print(f"   ✅ {col}: {valid_count} 有效值")
                        else:
                            print(f"   ❌ {col}: 不存在！")
                
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
                progress = (current_time - start_time) / (end_time - start_time) * 100
                print(f"\n📊 进度: {progress:.1f}% | 迭代: {iteration} | 图表更新: {update_count}")
                print(f"   权益: ${current_equity:,.0f} | 持仓: {current_position}")
            
            # 前进1分钟
            current_time += timedelta(minutes=STEP_MINUTES)
    
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断回测")
    
    # 最终更新
    print(f"\n✅ 回测循环完成！")
    print(f"\n" + "="*70)
    print(f"📊 回测结果 - {strategy_config['name']}")
    print("="*70)
    
    final_time = end_time
    strategy_df = strategy.get_history_data(TICKER)
    trade_log = position_manager.get_trade_log()
    
    # 获取最终价格
    df_final = data_fetcher.get_latest_bars(
        ticker=TICKER,
        lookback_minutes=LOOKBACK_MINUTES,
        end_dt=final_time,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute)
    )
    
    if not df_final.empty:
        final_price = df_final.iloc[-1]['close']
    else:
        final_price = current_price
    
    # 获取最终账户状态
    final_status = position_manager.get_account_status(final_price)
    trade_log = position_manager.get_trade_log()
    
    final_equity = final_status.get('equity', INITIAL_CAPITAL)
    total_pnl = final_status.get('total_pnl', 0)
    total_pnl_pct = final_status.get('total_pnl_pct', 0)
    
    print(f"\n💰 资金情况:")
    print(f"   初始资金: ${INITIAL_CAPITAL:,.2f}")
    print(f"   最终权益: ${final_equity:,.2f}")
    print(f"   盈亏: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)")
    print(f"   现金: ${final_status.get('cash', 0):,.2f}")
    print(f"   持仓: {final_status.get('position', 0)} 股")
    
    print(f"\n📈 交易统计:")
    
    if not trade_log.empty:
        print(f"   总交易数: {len(trade_log)}")
        
        # 计算完成的交易（检查列名）
        if 'type' in trade_log.columns:
            completed_trades = trade_log[trade_log['type'].isin(['SELL', 'COVER'])]
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
    
    print(f"\n" + "="*70)
    print(f"✅ 回测完成！查看图表: {chart_file}")
    print("="*70 + "\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='带图表的多策略回测')
    
    parser.add_argument('--strategy', type=str, default='moderate',
                       choices=list(STRATEGY_CONFIGS.keys()),
                       help='选择策略 (conservative/moderate/high_freq/ultra)')
    
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