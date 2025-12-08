# src/live/live_runner.py

from datetime import datetime
import os
from dotenv import load_dotenv

# --- Core Modules ---
from src.cache.trading_cache import TradingCache
from src.manager.position_manager import PositionManager
from src.data_fetcher.alpaca_data_fetcher import AlpacaDataFetcher
from src.engine.live_engine import LiveEngine
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# --- Executors ---
from src.executor.simulation_executor import SimulationExecutor
from src.executor.alpaca_trade_executor import AlpacaExecutor

# --- Strategies ---
from src.strategies.mean_reversion_strategy import MeanReversionStrategy
from src.strategies.gemini_strategy import GeminiStrategy

load_dotenv()

# ==========================================
# 1. Configuration
# ==========================================

# 交易模式: 'paper' = 模拟盘, 'live' = 实盘, 'simulation' = 本地模拟（不连接 Alpaca）
TRADING_MODE = 'paper'  # ⚠️ 谨慎选择！'live' 会执行真实交易

# 财务参数（仅用于 simulation 模式）
FINANCE_PARAMS = {
    'INITIAL_CAPITAL': 100000.0,
    'COMMISSION_RATE': 0.0003,
    'SLIPPAGE_RATE': 0.0001,
    'MIN_LOT_SIZE': 10,
    'MAX_ALLOCATION': 0.2,
    'STAMP_DUTY_RATE': 0.001,
}

# 交易设置
TICKER = "TSLA"

# 运行参数
INTERVAL_SECONDS = 300       # 策略运行间隔（秒），300 = 5分钟
LOOKBACK_MINUTES = 120       # 数据回溯时间（分钟）
DATA_TIMEFRAME = TimeFrame(5, TimeFrameUnit.Minute)  # K线周期

# 交易时间控制
RESPECT_MARKET_HOURS = True  # 是否只在美股交易时间内运行
MAX_RUNTIME_MINUTES = None   # 最大运行时间（分钟），None = 无限制

# 策略选择: 'mean_reversion' or 'gemini_ai'
SELECTED_STRATEGY = 'mean_reversion'

# ==========================================
# 2. Signal Callback (可选)
# ==========================================

def on_signal_received(signal_dict: dict, price: float, timestamp: datetime):
    """
    信号回调函数 - 可用于发送通知、记录日志等。
    
    Args:
        signal_dict: 策略返回的信号字典
        price: 当前价格
        timestamp: 时间戳
    """
    signal = signal_dict.get('signal', 'UNKNOWN')
    confidence = signal_dict.get('confidence_score', 0)
    
    # 示例：只对高置信度信号发送通知
    if signal in ['BUY', 'SELL'] and confidence >= 7:
        print(f"📢 高置信度信号: {signal} @ ${price:.2f} (置信度: {confidence}/10)")
        
        # 这里可以添加：
        # - 发送邮件通知
        # - 发送 Telegram/Discord 消息
        # - 写入数据库
        # - 等等...

# ==========================================
# 3. Initialization
# ==========================================

def main():
    print("\n" + "="*60)
    print("🚀 实盘交易系统初始化")
    print("="*60)
    print(f"   交易标的: {TICKER}")
    print(f"   交易模式: {TRADING_MODE.upper()}")
    print(f"   策略: {SELECTED_STRATEGY}")
    print(f"   运行间隔: {INTERVAL_SECONDS} 秒")
    
    if TRADING_MODE == 'live':
        print("\n" + "⚠️"*20)
        print("   警告: 您正在使用实盘模式！")
        print("   所有交易将使用真实资金！")
        print("⚠️"*20)
        
        confirm = input("\n确认启动实盘交易? (输入 'YES' 确认): ")
        if confirm != 'YES':
            print("已取消启动。")
            return
    
    # A. Data Fetcher
    data_fetcher = AlpacaDataFetcher()
    
    # B. Cache System
    cache_path = os.path.join('cache', f'{TICKER}_live_cache.json')
    cache = TradingCache(cache_path)
    
    # C. Executor & Position Manager
    if TRADING_MODE == 'simulation':
        print("🔧 执行器: 本地模拟")
        executor = SimulationExecutor(FINANCE_PARAMS)
    elif TRADING_MODE == 'paper':
        print("🔧 执行器: Alpaca 模拟盘 (Paper)")
        executor = AlpacaExecutor(paper=True, max_allocation_rate=FINANCE_PARAMS['MAX_ALLOCATION'])
    elif TRADING_MODE == 'live':
        print("🔧 执行器: Alpaca 实盘 (Live)")
        executor = AlpacaExecutor(paper=False, max_allocation_rate=FINANCE_PARAMS['MAX_ALLOCATION'])
    else:
        raise ValueError(f"无效的交易模式: {TRADING_MODE}")
    
    position_manager = PositionManager(executor, FINANCE_PARAMS)
    
    # D. Strategy
    print(f"🧠 策略: {SELECTED_STRATEGY}")
    
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
        raise ValueError(f"无效的策略: {SELECTED_STRATEGY}")
    
    # ==========================================
    # 4. Create and Run Live Engine
    # ==========================================
    
    live_engine = LiveEngine(
        ticker=TICKER,
        strategy=strategy,
        position_manager=position_manager,
        data_fetcher=data_fetcher,
        cache=cache,
        interval_seconds=INTERVAL_SECONDS,
        lookback_minutes=LOOKBACK_MINUTES,
        timeframe=DATA_TIMEFRAME,
        respect_market_hours=RESPECT_MARKET_HOURS,
        max_runtime_minutes=MAX_RUNTIME_MINUTES,
        on_signal_callback=on_signal_received
    )
    
    # 运行引擎
    report = live_engine.run()
    
    # ==========================================
    # 5. Final Report
    # ==========================================
    
    print("\n" + "="*60)
    print("💰 最终结果")
    print("="*60)
    print(f"   运行时长: {report['runtime_seconds'] / 60:.1f} 分钟")
    print(f"   迭代次数: {report['iterations']}")
    print(f"   交易信号: {report['signals']}")
    print(f"   执行交易: {report['trades_executed']}")
    print(f"   最终权益: ${report['final_equity']:,.2f}")
    print("="*60)
    
    # 打印交易日志
    trade_log = position_manager.get_trade_log()
    if trade_log is not None and not trade_log.empty:
        print("\n📝 交易日志:")
        display_log = trade_log[['time', 'type', 'qty', 'price', 'fee', 'net_pnl']].copy()
        display_log['time'] = display_log['time'].dt.strftime('%Y-%m-%d %H:%M')
        print(display_log.to_markdown(index=False, floatfmt=".2f"))
    else:
        print("\n🤷 无交易记录。")


if __name__ == '__main__':
    main()