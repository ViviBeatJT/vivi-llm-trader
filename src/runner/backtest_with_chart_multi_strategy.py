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
from src.strategies.aggresive_mean_reversion_strategy import AggressiveMeanReversionStrategy
from src.strategies.moderate_aggresive_strategy import ModerateAggressiveStrategy
from src.strategies.high_frequency_strategy import HighFrequencyStrategy
from src.strategies.ultra_aggresive_strategy import UltraAggressiveStrategy

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
STEP_MINUTES = 1

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
    
    # 4. 获取历史数据
    print(f"\n📥 获取历史数据...")
    
    # 解析日期并设置时间范围
    from datetime import datetime, time as dt_time
    import pytz
    
    US_EASTERN = pytz.timezone('America/New_York')
    date_parts = [int(x) for x in TRADING_DATE.split('-')]
    
    # 市场时间: 9:30 - 16:00 ET
    start_dt = US_EASTERN.localize(datetime(date_parts[0], date_parts[1], date_parts[2], 9, 30))
    end_dt = US_EASTERN.localize(datetime(date_parts[0], date_parts[1], date_parts[2], 16, 0))
    
    # 转换为 UTC
    start_dt = start_dt.astimezone(timezone.utc)
    end_dt = end_dt.astimezone(timezone.utc)
    
    print(f"   时间范围: {TRADING_DATE} 9:30-16:00 ET")
    
    # 获取整天的数据
    try:
        historical_bars = data_fetcher.get_latest_bars(
            ticker=TICKER,
            lookback_minutes=450,  # 从9:30到16:00约6.5小时 = 390分钟
            end_dt=end_dt,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute)
        )
    except Exception as e:
        print(f"❌ 获取数据失败: {e}")
        return
    
    if historical_bars.empty:
        print(f"❌ 没有数据！")
        return
    
    print(f"✅ 获取了 {len(historical_bars)} 根 5分钟 K线")
    print(f"   时间范围: {historical_bars.index[0]} 至 {historical_bars.index[-1]}")
    
    # 5. 回测循环
    print(f"\n🏃 开始回测...")
    print(f"   策略: {strategy_config['name']}")
    print(f"   图表每次迭代更新")
    print(f"="*70)
    
    total_bars = len(historical_bars)
    iteration = 0
    update_count = 0
    
    for i in range(0, total_bars, STEP_MINUTES):
        iteration += 1
        current_time_bars = historical_bars.iloc[:i + 1]
        
        if len(current_time_bars) < 20:
            continue
        
        current_time = current_time_bars.index[-1]
        current_price = current_time_bars.iloc[-1]['close']
        
        # 获取当前持仓和权益
        current_position = position_manager.get_position(TICKER)
        
        # 计算当前权益
        summary = position_manager.get_summary()
        current_equity = summary.get('total_value', INITIAL_CAPITAL)
        
        # 获取平均成本
        avg_cost = 0
        if current_position != 0:
            positions = summary.get('positions', {})
            if TICKER in positions:
                avg_cost = positions[TICKER].get('avg_price', 0)
        
        # 获取信号
        signal_data, _ = strategy.get_signal(
            ticker=TICKER,
            new_data=current_time_bars.tail(1),
            current_position=current_position,
            avg_cost=avg_cost,
            verbose=False
        )
        
        signal = signal_data['signal']
        
        # 执行交易
        if signal in ['BUY', 'SELL', 'SHORT', 'COVER']:
            try:
                result = executor.execute_order(
                    ticker=TICKER,
                    action=signal,
                    shares=SHARES_PER_TRADE,
                    current_price=current_price,
                    timestamp=current_time
                )
                
                if result and result.get('status') == 'success':
                    emoji = {"BUY": "🟢", "SELL": "🔴", "SHORT": "🔻", "COVER": "🔺"}
                    print(f"\n{emoji.get(signal, '⚪')} {current_time.strftime('%H:%M')} | "
                          f"{signal} @ ${current_price:.2f} x {SHARES_PER_TRADE}")
                    print(f"   {signal_data.get('reason', 'N/A')}")
            except Exception as e:
                print(f"⚠️ 交易执行失败: {e}")
        
        # 更新图表
        strategy_df = strategy.get_history_data(TICKER)
        trade_log = position_manager.get_all_trades()  # 使用正确的方法
        # current_equity 已在上面计算过了
        
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
        if iteration % 50 == 0:
            progress = (i / total_bars) * 100
            print(f"\n📊 进度: {progress:.0f}% | 迭代: {iteration}")
            print(f"   权益: ${current_equity:,.0f} | 持仓: {current_position}")
    
    # 6. 最终结果
    print(f"\n" + "="*70)
    print(f"📊 回测结果 - {strategy_config['name']}")
    print("="*70)
    
    final_price = historical_bars.iloc[-1]['close']
    
    # 获取最终账户状态
    summary = position_manager.get_summary()
    trade_log = position_manager.get_all_trades()
    
    final_equity = summary.get('total_value', INITIAL_CAPITAL)
    total_pnl = final_equity - INITIAL_CAPITAL
    total_pnl_pct = (total_pnl / INITIAL_CAPITAL) * 100
    
    print(f"\n💰 资金情况:")
    print(f"   初始资金: ${INITIAL_CAPITAL:,.2f}")
    print(f"   最终权益: ${final_equity:,.2f}")
    print(f"   盈亏: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)")
    print(f"   现金: ${summary.get('cash', 0):,.2f}")
    
    print(f"\n📈 交易统计:")
    print(f"   总交易数: {len(trade_log) if trade_log else 0}")
    
    if trade_log and len(trade_log) > 0:
        # 计算完成的交易
        sell_trades = [t for t in trade_log if t.get('action') in ['SELL', 'COVER']]
        if sell_trades:
            winning_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
            win_rate = len(winning_trades) / len(sell_trades) * 100
            print(f"   完成交易: {len(sell_trades)}")
            print(f"   胜率: {win_rate:.1f}%")
    
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