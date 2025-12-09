# src/runner/aggressive_live_runner.py

"""
激进均值回归策略 - 实盘运行器

特点：
- 每1分钟监控一次（可配置）
- 使用5分钟K线计算布林带
- 自动止损（亏损10%时平仓，可配置）
- 突破上轨做空，回归中线平空
- 跌破下轨做多,回归中线平多
"""

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

# --- 新的激进策略 ---
from src.strategies.aggressive_mean_reversion_strategy import AggressiveMeanReversionStrategy

load_dotenv()

# ==========================================
# 1. 交易模式配置
# ==========================================

# 交易模式: 'paper' = 模拟盘, 'live' = 实盘, 'simulation' = 本地模拟
TRADING_MODE = 'paper'  # ⚠️ 谨慎选择！'live' 会执行真实交易

# ==========================================
# 2. 策略参数配置（重点！）
# ==========================================

# 🎯 监控频率设置
MONITOR_INTERVAL_SECONDS = 60  # 监控间隔（秒），60 = 每分钟检查一次

# 📊 K线周期设置
DATA_TIMEFRAME = TimeFrame(5, TimeFrameUnit.Minute)  # 5分钟K线（用于计算技术指标）
LOOKBACK_MINUTES = 120  # 数据回溯时间（分钟），至少需要 BB周期 * 5分钟

# 💹 策略参数
BB_PERIOD = 20           # 布林带周期（需要20根K线）
BB_STD_DEV = 2.0         # 布林带标准差倍数
STOP_LOSS_THRESHOLD = 0.10  # 止损阈值（10% = 0.10）

# 💰 财务参数
FINANCE_PARAMS = {
    'INITIAL_CAPITAL': 100000.0,
    'COMMISSION_RATE': 0.0003,
    'SLIPPAGE_RATE': 0.0001,
    'MIN_LOT_SIZE': 10,
    'MAX_ALLOCATION': 0.2,  # 最大仓位比例（20%）
}

# ==========================================
# 3. 运行控制
# ==========================================

TICKER = "TSLA"  # 交易标的

# 交易时间控制
RESPECT_MARKET_HOURS = True   # 是否只在美股交易时间内运行
MAX_RUNTIME_MINUTES = None    # 最大运行时间（分钟），None = 无限制

# 是否在启动时从 API 同步仓位状态
SYNC_POSITION_ON_START = True

# ==========================================
# 4. 信号回调（可选）
# ==========================================

def on_signal_received(signal_dict: dict, price: float, timestamp: datetime):
    """
    信号回调函数 - 可用于发送通知、记录日志等。
    """
    signal = signal_dict.get('signal', 'UNKNOWN')
    confidence = signal_dict.get('confidence_score', 0)
    reason = signal_dict.get('reason', '')
    
    # 对所有交易信号发送通知
    if signal in ['BUY', 'SELL', 'SHORT', 'COVER']:
        print(f"\n{'='*60}")
        print(f"📢 交易信号通知")
        print(f"{'='*60}")
        print(f"   时间: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"   信号: {signal}")
        print(f"   价格: ${price:.2f}")
        print(f"   置信度: {confidence}/10")
        print(f"   原因: {reason}")
        print(f"{'='*60}\n")
        
        # 这里可以添加：
        # - 发送邮件通知
        # - 发送 Telegram/Discord 消息
        # - 写入数据库
        # - 触发其他自动化流程

# ==========================================
# 5. 主函数
# ==========================================

def main():
    print("\n" + "="*60)
    print("🚀 激进均值回归策略 - 实盘交易系统")
    print("="*60)
    print(f"   交易标的: {TICKER}")
    print(f"   交易模式: {TRADING_MODE.upper()}")
    print(f"   监控频率: 每 {MONITOR_INTERVAL_SECONDS} 秒")
    print(f"   K线周期: {DATA_TIMEFRAME.amount} 分钟")
    print(f"   布林带参数: 周期={BB_PERIOD}, 标准差={BB_STD_DEV}σ")
    print(f"   止损阈值: {STOP_LOSS_THRESHOLD*100:.1f}%")
    
    if TRADING_MODE == 'live':
        print("\n" + "⚠️"*20)
        print("   警告: 您正在使用实盘模式！")
        print("   所有交易将使用真实资金！")
        print("   策略会自动止损，但仍有风险！")
        print("⚠️"*20)
        
        confirm = input("\n确认启动实盘交易? (输入 'YES' 确认): ")
        if confirm != 'YES':
            print("已取消启动。")
            return
    
    # A. Data Fetcher（包含账户和持仓 API）
    is_paper = TRADING_MODE in ['paper', 'simulation']
    data_fetcher = AlpacaDataFetcher(paper=is_paper)
    
    # B. Cache System
    cache_path = os.path.join('cache', f'{TICKER}_aggressive_cache.json')
    cache = TradingCache(cache_path)
    
    # C. Executor & Position Manager
    if TRADING_MODE == 'simulation':
        print("🔧 执行器: 本地模拟")
        executor = SimulationExecutor(FINANCE_PARAMS)
        position_manager = PositionManager(executor, FINANCE_PARAMS)
    elif TRADING_MODE == 'paper':
        print("🔧 执行器: Alpaca 模拟盘 (Paper)")
        executor = AlpacaExecutor(paper=True, max_allocation_rate=FINANCE_PARAMS['MAX_ALLOCATION'])
        position_manager = PositionManager(executor, FINANCE_PARAMS, data_fetcher=data_fetcher)
    elif TRADING_MODE == 'live':
        print("🔧 执行器: Alpaca 实盘 (Live)")
        executor = AlpacaExecutor(paper=False, max_allocation_rate=FINANCE_PARAMS['MAX_ALLOCATION'])
        position_manager = PositionManager(executor, FINANCE_PARAMS, data_fetcher=data_fetcher)
    else:
        raise ValueError(f"无效的交易模式: {TRADING_MODE}")
    
    # D. 从 API 同步仓位状态（如果启用）
    if SYNC_POSITION_ON_START and TRADING_MODE in ['paper', 'live']:
        print(f"\n🔄 正在从 API 同步 {TICKER} 仓位状态...")
        sync_success = position_manager.sync_from_api(TICKER)
        if not sync_success:
            print("⚠️ 仓位同步失败，将使用本地初始状态")
    
    # E. 创建激进均值回归策略
    print(f"\n💹 初始化激进均值回归策略...")
    strategy = AggressiveMeanReversionStrategy(
        bb_period=BB_PERIOD,
        bb_std_dev=BB_STD_DEV,
        max_history_bars=500,
        stop_loss_threshold=STOP_LOSS_THRESHOLD,
        monitor_interval_seconds=MONITOR_INTERVAL_SECONDS
    )
    
    # ==========================================
    # 6. 创建并运行 Live Engine
    # ==========================================
    
    print(f"\n{'='*60}")
    print("🎯 策略规则:")
    print("="*60)
    print("   📈 价格突破上轨 → 做空（SHORT）")
    print("   📉 空仓时价格回到中线 → 平空（COVER）")
    print("   📉 价格跌破下轨 → 做多（BUY）")
    print("   📈 多仓时价格回到中线 → 平多（SELL）")
    print(f"   ⚠️ 单笔持仓亏损 {STOP_LOSS_THRESHOLD*100:.0f}% → 强制止损")
    print("="*60)
    
    live_engine = LiveEngine(
        ticker=TICKER,
        strategy=strategy,
        position_manager=position_manager,
        data_fetcher=data_fetcher,
        cache=cache,
        interval_seconds=MONITOR_INTERVAL_SECONDS,
        lookback_minutes=LOOKBACK_MINUTES,
        timeframe=DATA_TIMEFRAME,
        respect_market_hours=RESPECT_MARKET_HOURS,
        max_runtime_minutes=MAX_RUNTIME_MINUTES,
        on_signal_callback=on_signal_received
    )
    
    # 运行引擎
    print(f"\n🚀 启动实盘引擎...")
    print(f"   按 Ctrl+C 可随时安全停止\n")
    
    report = live_engine.run()
    
    # ==========================================
    # 7. 最终报告
    # ==========================================
    
    print("\n" + "="*60)
    print("💰 运行结果")
    print("="*60)
    print(f"   运行时长: {report['runtime_seconds'] / 60:.1f} 分钟")
    print(f"   迭代次数: {report['iterations']}")
    print(f"   交易信号: {report['signals']}")
    print(f"   执行交易: {report['trades_executed']}")
    print(f"   最终权益: ${report['final_equity']:,.2f}")
    print(f"   最终持仓: {report['final_position']:.0f} 股")
    if report['final_position'] != 0:
        print(f"   最终价格: ${report['final_price']:.2f}")
    print("="*60)
    
    # 打印交易日志
    trade_log = position_manager.get_trade_log()
    if trade_log is not None and not trade_log.empty:
        print("\n📝 交易日志:")
        display_log = trade_log[['time', 'type', 'qty', 'price', 'fee', 'net_pnl']].copy()
        display_log['time'] = display_log['time'].dt.strftime('%Y-%m-%d %H:%M')
        print(display_log.to_markdown(index=False, floatfmt=".2f"))
        
        # 统计
        total_pnl = trade_log['net_pnl'].sum()
        win_trades = len(trade_log[trade_log['net_pnl'] > 0])
        loss_trades = len(trade_log[trade_log['net_pnl'] < 0])
        
        print(f"\n📊 交易统计:")
        print(f"   总盈亏: ${total_pnl:,.2f}")
        print(f"   盈利次数: {win_trades}")
        print(f"   亏损次数: {loss_trades}")
        if win_trades + loss_trades > 0:
            win_rate = win_trades / (win_trades + loss_trades) * 100
            print(f"   胜率: {win_rate:.1f}%")
    else:
        print("\n🤷 无交易记录。")
    
    print(f"\n{'='*60}")
    print("✅ 程序结束")
    print("="*60)


if __name__ == '__main__':
    main()