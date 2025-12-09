# src/backtest/backtest_runner.py

from datetime import datetime, timezone, time as dt_time
import os
from dotenv import load_dotenv
import pytz

# --- Core Modules ---
from src.cache.trading_cache import TradingCache
from src.manager.position_manager import PositionManager
from src.data_fetcher.alpaca_data_fetcher import AlpacaDataFetcher
from src.engine.backtest_engine import BacktestEngine
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# --- Executors ---
from src.executor.simulation_executor import SimulationExecutor
from src.executor.alpaca_trade_executor import AlpacaExecutor

# --- Strategies (no data_fetcher dependency) ---
from src.strategies.mean_reversion_strategy import MeanReversionStrategy
from src.strategies.gemini_strategy import GeminiStrategy

load_dotenv()

# ==========================================
# US Market Hours Validation
# ==========================================

# 美股交易时间 (Eastern Time)
US_EASTERN = pytz.timezone('America/New_York')
MARKET_OPEN_TIME = dt_time(9, 30)   # 9:30 AM ET
MARKET_CLOSE_TIME = dt_time(16, 0)  # 4:00 PM ET


def validate_market_hours(dt: datetime, label: str = "Time") -> datetime:
    """
    验证时间是否在美股开盘时间内。
    
    Args:
        dt: 要验证的时间
        label: 用于错误消息的标签 (如 "Start time", "End time")
        
    Returns:
        datetime: 转换为 UTC 的有效时间
        
    Raises:
        ValueError: 如果时间不在美股交易时间内
    """
    # 确保有时区信息
    if dt.tzinfo is None:
        # 假设无时区的输入是 Eastern Time
        dt = US_EASTERN.localize(dt)
    
    # 转换到 Eastern Time 进行验证
    dt_eastern = dt.astimezone(US_EASTERN)
    market_time = dt_eastern.time()
    
    # 检查是否在交易时间内
    if market_time < MARKET_OPEN_TIME or market_time > MARKET_CLOSE_TIME:
        raise ValueError(
            f"❌ {label} {dt_eastern.strftime('%Y-%m-%d %H:%M %Z')} 不在美股交易时间内。\n"
            f"   美股交易时间: {MARKET_OPEN_TIME.strftime('%H:%M')} - {MARKET_CLOSE_TIME.strftime('%H:%M')} ET\n"
            f"   请调整时间到交易时段内。"
        )
    
    # 检查是否是周末
    weekday = dt_eastern.weekday()
    if weekday >= 5:  # 5=Saturday, 6=Sunday
        day_name = "Saturday" if weekday == 5 else "Sunday"
        raise ValueError(
            f"❌ {label} {dt_eastern.strftime('%Y-%m-%d %H:%M %Z')} 是 {day_name}，美股休市。\n"
            f"   请选择周一至周五的交易日。"
        )
    
    # 返回 UTC 时间
    return dt.astimezone(timezone.utc)


def print_market_hours_info(start_dt: datetime, end_dt: datetime):
    """打印市场时间信息。"""
    start_et = start_dt.astimezone(US_EASTERN)
    end_et = end_dt.astimezone(US_EASTERN)
    
    print(f"⏰ 回测时间范围:")
    print(f"   开始: {start_et.strftime('%Y-%m-%d %H:%M %Z')} ({start_dt.strftime('%H:%M UTC')})")
    print(f"   结束: {end_et.strftime('%Y-%m-%d %H:%M %Z')} ({end_dt.strftime('%H:%M UTC')})")
    print(f"   美股交易时间: {MARKET_OPEN_TIME.strftime('%H:%M')} - {MARKET_CLOSE_TIME.strftime('%H:%M')} ET")


# ==========================================
# 1. Configuration
# ==========================================

# Simulation / Finance Settings
FINANCE_PARAMS = {
    'INITIAL_CAPITAL': 100000.0,
    'COMMISSION_RATE': 0.0003,
    'SLIPPAGE_RATE': 0.0001,
    'MIN_LOT_SIZE': 10,
    'MAX_ALLOCATION': 0.2,
    'STAMP_DUTY_RATE': 0.001,
}

# Run Settings
TICKER = "TSLA"

# 时间设置 (使用 Eastern Time 更直观)
# 美股交易时间: 9:30 AM - 4:00 PM ET
START_TIME = US_EASTERN.localize(datetime(2025, 12, 8, 9, 30))   # 9:30 AM ET
END_TIME = US_EASTERN.localize(datetime(2025, 12, 8, 16, 0))     # 4:00 PM ET

STEP_MINUTES = 5
LOOKBACK_MINUTES = 120  # Data lookback for strategy

# Timeframe for K-line data
DATA_TIMEFRAME = TimeFrame(5, TimeFrameUnit.Minute)

# Mode: True = Backtest (Simulation), False = Live/Paper (Real API Trade)
IS_BACKTEST_MODE = True 

# Strategy Selection: 'mean_reversion' or 'gemini_ai'
SELECTED_STRATEGY = 'mean_reversion' 

# ==========================================
# 2. Validate Market Hours
# ==========================================

print(f"\n🚀 Initializing Runner for {TICKER}...")

try:
    START_TIME = validate_market_hours(START_TIME, "Start time")
    END_TIME = validate_market_hours(END_TIME, "End time")
    
    if END_TIME <= START_TIME:
        raise ValueError("❌ End time must be after start time.")
    
    print_market_hours_info(START_TIME, END_TIME)
    
except ValueError as e:
    print(str(e))
    exit(1)

# ==========================================
# 3. Initialization
# ==========================================

# A. Data Fetcher (used by BacktestEngine, not by Strategy)
data_fetcher = AlpacaDataFetcher()

# B. Cache System
cache_path = os.path.join('cache', f'{TICKER}_trading_cache.json')
cache = TradingCache(cache_path)

# C. Executor & Position Manager
if IS_BACKTEST_MODE:
    print("🔧 Mode: Simulation / Backtest")
    executor = SimulationExecutor(FINANCE_PARAMS)
else:
    print("⚠️ Mode: LIVE / PAPER TRADING")
    executor = AlpacaExecutor(paper=True, max_allocation_rate=FINANCE_PARAMS['MAX_ALLOCATION'])

position_manager = PositionManager(executor, FINANCE_PARAMS)

# D. Strategy (no data_fetcher dependency anymore!)
print(f"🧠 Strategy: {SELECTED_STRATEGY}")

if SELECTED_STRATEGY == 'mean_reversion':
    strategy = MeanReversionStrategy(
        bb_period=20, 
        bb_std_dev=2,
        rsi_window=14,
        rsi_oversold=30,
        rsi_overbought=70,
        max_history_bars=500
    )
elif SELECTED_STRATEGY == 'gemini_ai':
    strategy = GeminiStrategy(
        cache=cache,
        use_cache=True,
        temperature=0.2,
        delay_seconds=2,
        bb_period=20,
        rsi_window=14,
        max_history_bars=500
    )
else:
    raise ValueError(f"Invalid strategy selected: {SELECTED_STRATEGY}")

# ==========================================
# 4. Run Backtest Engine
# ==========================================

backtest_engine = BacktestEngine(
    ticker=TICKER,
    start_dt=START_TIME,
    end_dt=END_TIME,
    strategy=strategy,
    position_manager=position_manager,
    data_fetcher=data_fetcher,
    cache=cache,
    step_minutes=STEP_MINUTES,
    lookback_minutes=LOOKBACK_MINUTES,
    timeframe=DATA_TIMEFRAME
)

# Run
initial_cache_size = len(cache.data)
final_equity, trade_log = backtest_engine.run()

# ==========================================
# 5. Post-Run Processing
# ==========================================

# Save Cache if needed
if len(cache.data) > initial_cache_size:
    print(f"\n💾 Saving {len(cache.data) - initial_cache_size} new cache entries...")
    cache.save()

# Results
net_pnl = final_equity - FINANCE_PARAMS['INITIAL_CAPITAL']
return_pct = (net_pnl / FINANCE_PARAMS['INITIAL_CAPITAL']) * 100

print("\n" + "="*50)
print(f"💰 FINAL RESULT ({TICKER})")
print(f"   Strategy:        {strategy}")
print(f"   Initial Capital: ${FINANCE_PARAMS['INITIAL_CAPITAL']:,.2f}")
print(f"   Final Equity:    ${final_equity:,.2f}")
print(f"   Net P&L:         ${net_pnl:,.2f} ({return_pct:.2f}%)")
print("="*50)

# Print accumulated history info
print(f"\n📊 Strategy accumulated {strategy.get_history_size(TICKER)} bars of history for {TICKER}")

if trade_log is not None and not trade_log.empty:
    print("\n📝 Trade Log Summary:")
    display_log = trade_log[['time', 'type', 'qty', 'price', 'fee', 'net_pnl']].copy()
    display_log['time'] = display_log['time'].dt.strftime('%Y-%m-%d %H:%M')
    print(display_log.to_markdown(index=False, floatfmt=".2f"))
else:
    print("\n🤷 No trades executed.")