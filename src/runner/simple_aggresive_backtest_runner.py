# src/runner/simple_aggressive_backtest_runner.py

"""
简化版激进策略回测运行器

使用改进的 BacktestEngine，自动检测策略类型，无需手动扩展引擎。
"""

from datetime import datetime, timezone
import os
from dotenv import load_dotenv
import pytz

# --- Core Modules ---
from src.cache.trading_cache import TradingCache
from src.manager.position_manager import PositionManager
from src.data_fetcher.alpaca_data_fetcher import AlpacaDataFetcher
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# --- 使用改进的引擎 ---
from src.engine.improved_backtest_engine import ImprovedBacktestEngine

# --- Executors ---
from src.executor.simulation_executor import SimulationExecutor

# --- 激进策略 ---
from src.strategies.aggresive_mean_reversion_strategy import AggressiveMeanReversionStrategy

load_dotenv()

US_EASTERN = pytz.timezone('America/New_York')

# ==========================================
# 快速配置区（一目了然）
# ==========================================

# 🎯 基本设置
TICKER = "TSLA"
TRADING_DATE = "2024-12-05"  # 回测日期（YYYY-MM-DD）

# 💹 策略参数
BB_PERIOD = 20                  # 布林带周期
BB_STD_DEV = 2.0                # 标准差
STOP_LOSS_THRESHOLD = 0.10      # 止损 10%
MONITOR_INTERVAL = 60           # 监控间隔（秒）

# ⏱️ 回测设置
STEP_MINUTES = 1                # 每1分钟检查（模拟高频）
LOOKBACK_MINUTES = 120          # 数据回溯

# 💰 初始资金
INITIAL_CAPITAL = 100000.0

# ==========================================
# 自动初始化
# ==========================================

print("\n" + "="*60)
print(f"🚀 激进策略回测 - {TICKER} @ {TRADING_DATE}")
print("="*60)

# 解析日期并设置交易时间
date_parts = [int(x) for x in TRADING_DATE.split('-')]
START_TIME = US_EASTERN.localize(datetime(date_parts[0], date_parts[1], date_parts[2], 9, 30))
END_TIME = US_EASTERN.localize(datetime(date_parts[0], date_parts[1], date_parts[2], 16, 0))

START_TIME = START_TIME.astimezone(timezone.utc)
END_TIME = END_TIME.astimezone(timezone.utc)

print(f"\n📅 回测时间:")
print(f"   {START_TIME.astimezone(US_EASTERN).strftime('%Y-%m-%d %H:%M %Z')} →")
print(f"   {END_TIME.astimezone(US_EASTERN).strftime('%Y-%m-%d %H:%M %Z')}")

# 财务参数
FINANCE_PARAMS = {
    'INITIAL_CAPITAL': INITIAL_CAPITAL,
    'COMMISSION_RATE': 0.0003,
    'SLIPPAGE_RATE': 0.0001,
    'MIN_LOT_SIZE': 10,
    'MAX_ALLOCATION': 0.2,
}

# 初始化组件
data_fetcher = AlpacaDataFetcher()
cache = TradingCache(os.path.join('cache', f'{TICKER}_backtest_cache.json'))
executor = SimulationExecutor(FINANCE_PARAMS)
position_manager = PositionManager(executor, FINANCE_PARAMS)

# 创建策略
print(f"\n💹 策略参数:")
print(f"   布林带: 周期={BB_PERIOD}, 标准差={BB_STD_DEV}σ")
print(f"   止损阈值: {STOP_LOSS_THRESHOLD*100:.0f}%")
print(f"   步进间隔: {STEP_MINUTES}分钟")

strategy = AggressiveMeanReversionStrategy(
    bb_period=BB_PERIOD,
    bb_std_dev=BB_STD_DEV,
    stop_loss_threshold=STOP_LOSS_THRESHOLD,
    monitor_interval_seconds=MONITOR_INTERVAL
)

# 创建改进的回测引擎
backtest_engine = ImprovedBacktestEngine(
    ticker=TICKER,
    start_dt=START_TIME,
    end_dt=END_TIME,
    strategy=strategy,
    position_manager=position_manager,
    data_fetcher=data_fetcher,
    cache=cache,
    step_minutes=STEP_MINUTES,
    lookback_minutes=LOOKBACK_MINUTES,
    timeframe=TimeFrame(5, TimeFrameUnit.Minute)
)

# 运行回测
print("\n" + "="*60)
final_equity, trade_log = backtest_engine.run()

# ==========================================
# 结果分析
# ==========================================

net_pnl = final_equity - INITIAL_CAPITAL
return_pct = (net_pnl / INITIAL_CAPITAL) * 100

print("\n" + "="*60)
print("💰 回测结果")
print("="*60)
print(f"   初始资金:  ${INITIAL_CAPITAL:,.2f}")
print(f"   最终权益:  ${final_equity:,.2f}")
print(f"   净盈亏:    ${net_pnl:,.2f} ({return_pct:+.2f}%)")

if trade_log is not None and not trade_log.empty:
    print("\n📊 交易统计:")
    print("="*60)
    
    # 基础统计
    total_trades = len(trade_log)
    buy_sell_pairs = len(trade_log[trade_log['type'].isin(['SELL', 'COVER'])])
    
    # 盈亏统计
    winning = trade_log[trade_log['net_pnl'] > 0]
    losing = trade_log[trade_log['net_pnl'] < 0]
    
    print(f"   总交易次数:  {total_trades}")
    print(f"   完成交易对:  {buy_sell_pairs}")
    
    if len(winning) > 0:
        print(f"\n   ✅ 盈利: {len(winning)} 笔")
        print(f"      平均: ${winning['net_pnl'].mean():.2f}")
        print(f"      最大: ${winning['net_pnl'].max():.2f}")
    
    if len(losing) > 0:
        print(f"\n   ❌ 亏损: {len(losing)} 笔")
        print(f"      平均: ${losing['net_pnl'].mean():.2f}")
        print(f"      最大: ${losing['net_pnl'].min():.2f}")
    
    if buy_sell_pairs > 0:
        win_rate = len(winning) / buy_sell_pairs * 100
        print(f"\n   📈 胜率: {win_rate:.1f}%")
    
    total_fees = trade_log['fee'].sum()
    print(f"\n   💸 总手续费: ${total_fees:.2f}")
    
    # 显示交易日志
    print("\n📝 交易明细:")
    print("="*60)
    display = trade_log[['time', 'type', 'qty', 'price', 'net_pnl']].copy()
    display['time'] = display['time'].dt.strftime('%H:%M')
    print(display.to_markdown(index=False, floatfmt=".2f"))
else:
    print("\n⚠️ 无交易记录")
    print("   - 检查回测时间段是否有市场数据")
    print("   - 价格可能未触发交易信号")

print("\n" + "="*60)
print("✅ 回测完成")
print("="*60 + "\n")