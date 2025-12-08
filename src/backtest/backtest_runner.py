# src/backtest/backtest_runner.py

from datetime import datetime, timezone
import os
from dotenv import load_dotenv

# --- Core Modules ---
from src.cache.trading_cache import TradingCache
from src.manager.position_manager import PositionManager
from src.data_fetcher.alpaca_data_fetcher import AlpacaDataFetcher
from src.backtest.backtest_engine import BacktestEngine
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# --- Executors ---
from src.executor.simulation_executor import SimulationExecutor
from src.executor.alpaca_trade_executor import AlpacaExecutor

# --- Strategies (no data_fetcher dependency) ---
from src.strategies.mean_reversion_strategy import MeanReversionStrategy
from src.strategies.gemini_strategy import GeminiStrategy

load_dotenv()

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
START_TIME = datetime(2025, 12, 3, tzinfo=timezone.utc)
END_TIME = datetime(2025, 12, 4, tzinfo=timezone.utc)
STEP_MINUTES = 5
LOOKBACK_MINUTES = 120  # Data lookback for strategy

# Timeframe for K-line data
# Options: TimeFrame.Minute, TimeFrame.Hour, TimeFrame.Day
#          or custom: TimeFrame(5, TimeFrameUnit.Minute), TimeFrame(15, TimeFrameUnit.Minute), etc.
DATA_TIMEFRAME = TimeFrame(5, TimeFrameUnit.Minute)

# Mode: True = Backtest (Simulation), False = Live/Paper (Real API Trade)
IS_BACKTEST_MODE = True 

# Strategy Selection: 'mean_reversion' or 'gemini_ai'
SELECTED_STRATEGY = 'mean_reversion' 

# ==========================================
# 2. Initialization
# ==========================================

print(f"\n🚀 Initializing Runner for {TICKER}...")

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
    # Initialize Mean Reversion Strategy
    strategy = MeanReversionStrategy(
        bb_period=20, 
        bb_std_dev=2,
        rsi_window=14,
        rsi_oversold=30,
        rsi_overbought=70,
        max_history_bars=500
    )
elif SELECTED_STRATEGY == 'gemini_ai':
    # Initialize Gemini AI Strategy
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
# 3. Run Backtest Engine
# ==========================================

# Initialize the Backtest Engine
# Note: data_fetcher is passed to BacktestEngine, NOT to Strategy
backtest_engine = BacktestEngine(
    ticker=TICKER,
    start_dt=START_TIME,
    end_dt=END_TIME,
    strategy=strategy,               # The brain (analyzes data)
    position_manager=position_manager, # The execution & wallet
    data_fetcher=data_fetcher,       # The eyes (fetches data for engine)
    cache=cache,                     # The memory
    step_minutes=STEP_MINUTES,
    lookback_minutes=LOOKBACK_MINUTES,
    timeframe=DATA_TIMEFRAME         # K-line timeframe
)

# Run
initial_cache_size = len(cache.data)
final_equity, trade_log = backtest_engine.run()

# ==========================================
# 4. Post-Run Processing
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