# src/runner/live_runner.py

"""
实盘交易运行器 - 支持多策略和实时图表

支持策略：
1. conservative - 原始保守策略
2. moderate - 温和进取策略（推荐）
3. moderate_dynamic - 动态阈值温和进取策略
4. high_freq - 高频交易策略
5. ultra - 超激进策略
6. mean_reversion - 均值回归策略

用法：
    python live_runner.py --strategy moderate --ticker TSLA --mode paper
    python live_runner.py --strategy moderate_dynamic --ticker AAPL --mode simulation
    
特点：
- 命令行选择策略和股票
- 实时图表更新
- 支持模拟盘/实盘/本地模拟
- 自动刷新图表
"""

from datetime import datetime, timezone
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

# --- Simple Chart Visualizer ---
from src.visualization.simple_chart_visualizer import SimpleChartVisualizer

# --- Executors ---
from src.executor.simulation_executor import SimulationExecutor
from src.executor.alpaca_trade_executor import AlpacaExecutor

# --- 所有策略 ---
from src.strategies.mean_reversion_strategy import MeanReversionStrategy
from src.strategies.aggressive_mean_reversion_strategy import AggressiveMeanReversionStrategy
from src.strategies.moderate_aggressive_strategy import ModerateAggressiveStrategy
from src.strategies.high_frequency_strategy import HighFrequencyStrategy
from src.strategies.ultra_aggressive_strategy import UltraAggressiveStrategy
from src.strategies.moderate_aggressive_dynamic_strategy import ModerateAggressiveDynamicStrategy

load_dotenv()

US_EASTERN = pytz.timezone('America/New_York')

# ==========================================
# 1. 策略配置
# ==========================================

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
        'chart_file': 'live_conservative.html',
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
        'chart_file': 'live_moderate' +  '.html' ,
        'description': '接近布林带就交易，捕捉更多机会（推荐）'
    },
    'moderate_dynamic': {
        'class': ModerateAggressiveDynamicStrategy,
        'name': '动态阈值温和进取策略',
        'params': {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'base_entry_threshold': 0.85,
            'aggressive_entry_threshold': 0.70,
            'exit_threshold': 0.60,
            'stop_loss_threshold': 0.10,
            'high_volatility_threshold': 0.02,
            'low_volatility_threshold': 0.01,
            'monitor_interval_seconds': 60,
        },
        'chart_file': 'live_moderate_dynamic.html',
        'description': '动态调整阈值，横盘期也能交易'
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
        'chart_file': 'live_high_freq.html',
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
        'chart_file': 'live_ultra.html',
        'description': '动态调整，快速止盈止损'
    },
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
        'chart_file': 'live_mean_reversion.html',
        'description': '基于布林带和RSI的均值回归策略'
    }
}

# ==========================================
# 2. 默认配置
# ==========================================

# 财务参数
FINANCE_PARAMS = {
    'INITIAL_CAPITAL': 200000.00,
    'COMMISSION_RATE': 0.0003,
    'SLIPPAGE_RATE': 0.0001,
    'MIN_LOT_SIZE': 10,
    'MAX_ALLOCATION': 0.01,  # 💰 提高到95%，最大化资金利用率
    'STAMP_DUTY_RATE': 0.001,
}

# 运行参数
DEFAULT_INTERVAL_SECONDS = 30     # 策略运行间隔（秒）
DEFAULT_LOOKBACK_MINUTES = 300    # 数据回溯时间（分钟）
DATA_TIMEFRAME = TimeFrame(5, TimeFrameUnit.Minute)  # K线周期：5分钟

# 交易时间控制
RESPECT_MARKET_HOURS = True  # 是否只在美股交易时间内运行
MAX_RUNTIME_MINUTES = None   # 最大运行时间（分钟），None = 无限制

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
        """
        初始化图表更新器
        
        Args:
            visualizer: 可视化工具
            strategy: 策略实例
            position_manager: 仓位管理器
            ticker: 股票代码
            update_interval: 更新间隔（秒）
        """
        super().__init__()
        self.visualizer = visualizer
        self.strategy = strategy
        self.position_manager = position_manager
        self.ticker = ticker
        self.update_interval = update_interval
        self._running = True
        self.daemon = True  # 设置为守护线程
    
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
# 4. 策略创建函数
# ==========================================

def create_strategy(strategy_name: str, cache: TradingCache = None):
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


# ==========================================
# 5. 信号回调函数
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
    
    # 只对交易信号发送通知
    if signal in ['BUY', 'SELL', 'SHORT', 'COVER']:
        print(f"📢 交易信号: {signal} @ ${price:.2f} (置信度: {confidence}/10)")
        
        # 这里可以添加：
        # - 发送邮件通知
        # - 发送 Telegram/Discord 消息
        # - 写入数据库
        # - 等等...


# ==========================================
# 6. 主函数
# ==========================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='实盘交易运行器 - 支持多策略')
    
    parser.add_argument('--strategy', type=str, default='moderate',
                       choices=list(STRATEGY_CONFIGS.keys()),
                       help='选择策略 (默认: moderate)')
    
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

    if TRADING_MODE == 'live':
        print("\n" + "⚠️"*20)
        print("   警告: 您正在使用实盘模式！")
        print("   所有交易将使用真实资金！")
        print("⚠️"*20)
        
        confirm = input("\n确认启动实盘交易? (输入 'YES' 确认): ")
        if confirm != 'YES':
            print("已取消启动。")
            return
    
    # A. Data Fetcher（包含账户和持仓 API）
    is_paper = TRADING_MODE in ['paper', 'simulation']
    data_fetcher = AlpacaDataFetcher(paper=is_paper) if TRADING_MODE != 'simulation' else None
    
    # B. Cache System
    # cache_path = os.path.join('cache', f'{TICKER}_live_cache.json')
    cache = TradingCache(cache_file)
    
    # C. Executor & Position Manager
    if TRADING_MODE == 'simulation':
        print("🔧 执行器: 本地模拟")
        executor = SimulationExecutor(FINANCE_PARAMS)
        position_manager = PositionManager(executor, FINANCE_PARAMS)
        # 本地模拟模式创建一个假的 data_fetcher 用于获取数据
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
    
    # D. 从 API 同步仓位状态（如果启用）
    if SYNC_POSITION_ON_START and TRADING_MODE in ['paper', 'live']:
        print(f"\n🔄 正在从 API 同步 {TICKER} 仓位状态...")
        sync_success = position_manager.sync_from_api(TICKER)
        if not sync_success:
            print("⚠️ 仓位同步失败，将使用本地初始状态")
    
    # E. Strategy
    print(f"\n🧠 策略初始化...")
    strategy = create_strategy(SELECTED_STRATEGY, cache)
    
    # F. 初始化图表可视化（如果启用）
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
            on_signal_callback=on_signal_received
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
    print(f"   最终权益: ${report.get('final_equity', 0):,.2f}")
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