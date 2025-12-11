# src/runner/live_runner_with_chart.py

"""
实盘交易运行器 - 带实时图表（简化版）

✨ 特点：
1. 保持原 live_runner.py 的简洁逻辑
2. 添加实时图表可视化
3. 支持命令行参数选择策略和股票
4. 自动收盘平仓保护（15:55）
5. 防止重复开仓的安全检查

用法：
    python live_runner_with_chart.py --strategy mean_reversion --ticker TSLA --mode paper
    python live_runner_with_chart.py --strategy moderate --ticker AAPL --mode simulation
"""

from datetime import datetime, timezone, time as dt_time
import os
from dotenv import load_dotenv
import argparse
import time
import threading
import pytz
from pathlib import Path

# --- Core Modules ---
from src.cache.trading_cache import TradingCache
from src.manager.position_manager import PositionManager
from src.data_fetcher.alpaca_data_fetcher import AlpacaDataFetcher
from src.engine.live_engine import LiveEngine
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# --- Time utilities ---
from src.utils.market_time_utils import DEFAULT_FORCE_CLOSE_TIME, format_time_et

# --- Chart Visualizer ---
from src.visualization.simple_chart_visualizer import SimpleChartVisualizer

# --- Executors ---
from src.executor.simulation_executor import SimulationExecutor
from src.executor.alpaca_trade_executor import AlpacaExecutor

# --- Strategies ---
from src.strategies.mean_reversion_strategy import MeanReversionStrategy
from src.strategies.moderate_aggressive_strategy import ModerateAggressiveStrategy
from src.strategies.trend_aware_strategy import TrendAwareStrategy

load_dotenv()

# ==========================================
# 1. 策略配置
# ==========================================

STRATEGY_CONFIGS = {
    'mean_reversion': {
        'class': MeanReversionStrategy,
        'name': '均值回归策略',
        'params': {
            'bb_period': 20,
            'bb_std_dev': 2,
            'rsi_window': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'max_history_bars': 500
        },
        'description': '基于布林带和RSI的经典均值回归策略'
    },
    'moderate': {
        'class': ModerateAggressiveStrategy,
        'name': '温和进取策略',
        'params': {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'entry_threshold': 0.95,
            'exit_threshold': 0.60,
            'stop_loss_threshold': 0.10,
            'monitor_interval_seconds': 60,
        },
        'description': '接近布林带就交易，捕捉更多机会'
    },
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
}

# ==========================================
# 2. 默认配置
# ==========================================

# 财务参数
FINANCE_PARAMS = {
    'INITIAL_CAPITAL': 1000.0,      # 🔥 改为 1000 美元
    'COMMISSION_RATE': 0.0003,
    'SLIPPAGE_RATE': 0.0001,
    'MIN_LOT_SIZE': 1,              # 🔥 改为 1 股（最小交易单位）
    'MAX_ALLOCATION': 0.95,         # 🔥 改为 95%（几乎全仓，因为资金少）
    'STAMP_DUTY_RATE': 0.001,
}

# 运行参数
DEFAULT_INTERVAL_SECONDS = 30    # 策略运行间隔（秒）
DEFAULT_LOOKBACK_MINUTES = 300    # 数据回溯时间（分钟）🔥 增加到300确保有足够数据
DATA_TIMEFRAME = TimeFrame(5, TimeFrameUnit.Minute)  # K线周期：5分钟

# 交易时间控制
RESPECT_MARKET_HOURS = True  # 是否只在美股交易时间内运行
MAX_RUNTIME_MINUTES = None   # 最大运行时间（分钟），None = 无限制

# 强制平仓时间（默认15:55）
FORCE_CLOSE_TIME = DEFAULT_FORCE_CLOSE_TIME

# 是否在启动时从 API 同步仓位状态（仅 paper/live 模式有效）
SYNC_POSITION_ON_START = True

# 图表设置
CHART_UPDATE_INTERVAL = 30  # 图表更新间隔（秒）
AUTO_OPEN_BROWSER = True


# ==========================================
# 3. 图表更新线程
# ==========================================

class ChartUpdater(threading.Thread):
    """图表更新线程 - 定期更新图表"""
    
    def __init__(self, 
                 visualizer: SimpleChartVisualizer,
                 strategy,
                 position_manager: PositionManager,
                 ticker: str,
                 update_interval: int = 30):
        super().__init__()
        self.visualizer = visualizer
        self.strategy = strategy
        self.position_manager = position_manager
        self.ticker = ticker
        self.update_interval = update_interval
        self._running = True
        self.daemon = True
    
    def run(self):
        """运行图表更新循环"""
        print(f"\n📊 图表更新线程启动 (每 {self.update_interval} 秒更新)")
        
        while self._running:
            try:
                # 获取策略数据
                strategy_df = self.strategy.get_history_data(self.ticker)
                
                if strategy_df.empty:
                    time.sleep(self.update_interval)
                    continue
                
                # 获取当前价格
                current_price = strategy_df.iloc[-1]['close'] if not strategy_df.empty else 0.0
                
                # 获取账户状态
                account_status = self.position_manager.get_account_status(current_price)
                current_equity = account_status.get('equity', 0.0)
                current_position = account_status.get('position', 0.0)
                
                # 获取交易记录
                trade_log = self.position_manager.get_trade_log()
                
                # 更新图表
                self.visualizer.update_data(
                    market_data=strategy_df,
                    trade_log=trade_log,
                    current_equity=current_equity,
                    current_position=current_position,
                    timestamp=datetime.now(timezone.utc)
                )
                
                # 等待
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"⚠️ 图表更新错误: {e}")
                time.sleep(self.update_interval)
    
    def stop(self):
        """停止图表更新"""
        self._running = False


# ==========================================
# 4. 信号回调函数
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
    reason = signal_dict.get('reason', '')
    
    # 只对交易信号发送通知
    if signal in ['BUY', 'SELL', 'SHORT', 'COVER']:
        time_str = format_time_et(timestamp)
        print(f"📢 [{time_str}] 交易信号: {signal} @ ${price:.2f} (置信度: {confidence}/10)")
        if '强制平仓' in reason or '收盘' in reason:
            print(f"   🔔 收盘强制平仓")


# ==========================================
# 5. 主函数
# ==========================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='实盘交易运行器 - 带实时图表（简化版）')
    
    parser.add_argument('--strategy', type=str, default='mean_reversion',
                       choices=list(STRATEGY_CONFIGS.keys()),
                       help='选择策略 (默认: mean_reversion)')
    
    parser.add_argument('--ticker', type=str, default='TSLA',
                       help='股票代码 (默认: TSLA)')
    
    parser.add_argument('--mode', type=str, default='paper',
                       choices=['paper', 'live', 'simulation'],
                       help='交易模式: paper(模拟盘)/live(实盘)/simulation(本地模拟)')
    
    parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL_SECONDS,
                       help=f'策略运行间隔（秒，默认: {DEFAULT_INTERVAL_SECONDS}）')
    
    parser.add_argument('--no-chart', action='store_true',
                       help='禁用实时图表')
    
    args = parser.parse_args()
    
    # 获取配置
    TICKER = args.ticker
    TRADING_MODE = args.mode
    SELECTED_STRATEGY = args.strategy
    INTERVAL_SECONDS = args.interval
    ENABLE_CHART = not args.no_chart
    
    # 文件路径
    process_id = f"{TICKER}_{SELECTED_STRATEGY}_{TRADING_MODE}"
    base_dir = Path("live_trading")
    cache_dir = base_dir / "cache"
    charts_dir = base_dir / "charts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    chart_file = str(charts_dir / f"{process_id}.html")
    cache_file = str(cache_dir / f"{process_id}_cache.json")
    
    strategy_config = STRATEGY_CONFIGS[SELECTED_STRATEGY]
    
    print("\n" + "="*60)
    print("🚀 实盘交易系统初始化")
    print("="*60)
    print(f"   股票代码: {TICKER}")
    print(f"   交易模式: {TRADING_MODE.upper()}")
    print(f"   策略: {strategy_config['name']}")
    print(f"   运行间隔: {INTERVAL_SECONDS} 秒")
    print(f"   K线周期: {DATA_TIMEFRAME.amount} {DATA_TIMEFRAME.unit.name}")
    print(f"   实时图表: {'开启' if ENABLE_CHART else '关闭'}")
    if ENABLE_CHART:
        print(f"   图表文件: {chart_file}")
    print(f"   缓存文件: {cache_file}")
    print(f"   强制平仓时间: {FORCE_CLOSE_TIME.strftime('%H:%M')} ET")
    
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
    is_paper = TRADING_MODE in ['paper', 'simulation']
    data_fetcher = AlpacaDataFetcher(paper=is_paper) if TRADING_MODE != 'simulation' else None
    
    # B. Cache System
    cache = TradingCache(cache_file)
    
    # C. Executor & Position Manager
    if TRADING_MODE == 'simulation':
        print("🔧 执行器: 本地模拟")
        executor = SimulationExecutor(FINANCE_PARAMS)
        position_manager = PositionManager(executor, FINANCE_PARAMS)
        data_fetcher = AlpacaDataFetcher(paper=True)
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
    
    # D. 从 API 同步仓位状态
    if SYNC_POSITION_ON_START and TRADING_MODE in ['paper', 'live']:
        print(f"\n🔄 正在从 API 同步 {TICKER} 仓位状态...")
        sync_success = position_manager.sync_from_api(TICKER)
        if not sync_success:
            print("⚠️ 仓位同步失败，将使用本地初始状态")
    
    # E. Strategy
    print(f"\n🧠 策略初始化...")
    strategy_class = strategy_config['class']
    strategy_params = strategy_config['params']
    strategy = strategy_class(**strategy_params)
    
    print(f"   策略: {strategy_config['name']}")
    print(f"   描述: {strategy_config['description']}")
    
    # F. 初始化图表可视化
    visualizer = None
    chart_updater = None
    
    if ENABLE_CHART:
        print(f"\n📊 初始化实时图表...")
        visualizer = SimpleChartVisualizer(
            ticker=TICKER,
            output_file=chart_file,
            auto_open=AUTO_OPEN_BROWSER
        )
        visualizer.set_initial_capital(FINANCE_PARAMS['INITIAL_CAPITAL'])
        
        # 启动图表更新线程
        chart_updater = ChartUpdater(
            visualizer=visualizer,
            strategy=strategy,
            position_manager=position_manager,
            ticker=TICKER,
            update_interval=CHART_UPDATE_INTERVAL
        )
        chart_updater.start()
        print(f"   图表更新间隔: {CHART_UPDATE_INTERVAL} 秒")
        print(f"   浏览器打开: {chart_file}")
    
    # ==========================================
    # G. Create and Run Live Engine
    # ==========================================
    
    try:
        live_engine = LiveEngine(
            ticker=TICKER,
            strategy=strategy,
            position_manager=position_manager,
            data_fetcher=data_fetcher,
            cache=cache,
            interval_seconds=INTERVAL_SECONDS,
            lookback_minutes=DEFAULT_LOOKBACK_MINUTES,
            timeframe=DATA_TIMEFRAME,
            respect_market_hours=RESPECT_MARKET_HOURS,
            max_runtime_minutes=MAX_RUNTIME_MINUTES,
            on_signal_callback=on_signal_received,
            force_close_time=FORCE_CLOSE_TIME
        )
        
        # 运行引擎
        report = live_engine.run()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 收到 Ctrl+C，正在停止...")
    finally:
        # 停止图表更新线程
        if chart_updater:
            print("\n🛑 停止图表更新...")
            chart_updater.stop()
            chart_updater.join(timeout=2)
    
    # ==========================================
    # H. Final Report
    # ==========================================
    
    print("\n" + "="*60)
    print("💰 最终结果")
    print("="*60)
    print(f"   运行时长: {report.get('runtime_seconds', 0) / 60:.1f} 分钟")
    print(f"   迭代次数: {report.get('iterations', 0)}")
    print(f"   交易信号: {report.get('signals', 0)}")
    print(f"   执行交易: {report.get('trades_executed', 0)}")
    print(f"   强制平仓: {'是' if report.get('force_close_executed', False) else '否'}")
    print(f"   最终权益: ${report.get('final_equity', 0):,.2f}")
    print(f"   最终持仓: {report.get('final_position', 0):.0f} 股 {'✅' if report.get('final_position', 0) == 0 else '⚠️'}")
    print("="*60)
    
    # 打印交易日志
    trade_log = position_manager.get_trade_log()
    if trade_log is not None and not trade_log.empty:
        print("\n📝 交易日志:")
        display_log = trade_log[['time', 'type', 'qty', 'price', 'fee', 'net_pnl']].copy()
        display_log['time'] = display_log['time'].dt.strftime('%Y-%m-%d %H:%M')
        print(display_log.to_markdown(index=False, floatfmt=".2f"))
        
        # 打印交易统计
        print(f"\n📈 交易统计:")
        completed_trades = trade_log[trade_log['type'].isin(['SELL', 'COVER'])]
        if not completed_trades.empty and 'net_pnl' in completed_trades.columns:
            winning_trades = completed_trades[completed_trades['net_pnl'] > 0]
            losing_trades = completed_trades[completed_trades['net_pnl'] < 0]
            
            win_rate = len(winning_trades) / len(completed_trades) * 100 if len(completed_trades) > 0 else 0
            total_pnl = completed_trades['net_pnl'].sum()
            
            print(f"   完成交易: {len(completed_trades)}")
            print(f"   盈利交易: {len(winning_trades)}")
            print(f"   亏损交易: {len(losing_trades)}")
            print(f"   胜率: {win_rate:.1f}%")
            print(f"   总盈亏: ${total_pnl:,.2f}")
            
            if len(winning_trades) > 0:
                print(f"   平均盈利: ${winning_trades['net_pnl'].mean():.2f}")
            if len(losing_trades) > 0:
                print(f"   平均亏损: ${losing_trades['net_pnl'].mean():.2f}")
    else:
        print("\n🤷 无交易记录。")
    
    if ENABLE_CHART:
        print(f"\n📊 最终图表已保存: {chart_file}")
    
    print("\n✅ 程序结束")


if __name__ == '__main__':
    main()