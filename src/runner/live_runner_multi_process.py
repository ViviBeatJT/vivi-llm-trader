# src/runner/live_runner.py

"""
实盘交易运行器 - 支持多策略和多进程

支持策略：
1. conservative - 原始保守策略
2. moderate - 温和进取策略（推荐）
3. moderate_dynamic - 动态阈值温和进取策略
4. high_freq - 高频交易策略
5. ultra - 超激进策略
6. mean_reversion - 均值回归策略

用法：
    # 单进程
    python live_runner.py --strategy moderate --ticker TSLA --mode paper
    
    # 多进程（推荐）
    python live_runner.py --strategy moderate --ticker TSLA --mode paper &
    python live_runner.py --strategy moderate --ticker AAPL --mode paper &
    python live_runner.py --strategy moderate --ticker NVDA --mode paper &
    
特点：
- 🔀 多进程安全：每个 ticker 使用独立的日志、缓存和图表文件
- 📊 实时图表更新
- 🔒 文件锁防止冲突
- 💾 独立的缓存文件
"""

from datetime import datetime, timezone
import os
from dotenv import load_dotenv
import argparse
import time
import threading
import pytz
import sys
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
        'description': '只在完全突破布林带时交易'
    },
    'moderate': {
        'class': ModerateAggressiveStrategy,
        'name': '温和进取策略',
        'params': {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'entry_threshold': 0.85,
            'exit_threshold': 0.60,
            'stop_loss_threshold': 0.10,
            'monitor_interval_seconds': 60,
        },
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
        'description': '基于布林带和RSI的均值回归策略'
    }
}

# ==========================================
# 2. 默认配置
# ==========================================

# 财务参数
FINANCE_PARAMS = {
    'INITIAL_CAPITAL': 100000.0,
    'COMMISSION_RATE': 0.0003,
    'SLIPPAGE_RATE': 0.0001,
    'MIN_LOT_SIZE': 10,
    'MAX_ALLOCATION': 0.01,
    'STAMP_DUTY_RATE': 0.001,
}

# 运行参数
DEFAULT_INTERVAL_SECONDS = 30
DEFAULT_LOOKBACK_MINUTES = 300
DATA_TIMEFRAME = TimeFrame(5, TimeFrameUnit.Minute)

# 交易时间控制
RESPECT_MARKET_HOURS = True
MAX_RUNTIME_MINUTES = None

# 是否在启动时从 API 同步仓位状态
SYNC_POSITION_ON_START = True

# 图表设置
CHART_UPDATE_INTERVAL = 30
AUTO_OPEN_BROWSER = True


# ==========================================
# 3. 多进程资源管理器
# ==========================================

class ProcessResourceManager:
    """
    多进程资源管理器 - 为每个进程分配独立的资源
    
    确保不同进程之间的资源不冲突：
    - 独立的日志文件
    - 独立的缓存文件
    - 独立的图表文件
    - 独立的 PID 文件
    """
    
    def __init__(self, ticker: str, strategy_name: str, mode: str):
        """
        初始化资源管理器
        
        Args:
            ticker: 股票代码
            strategy_name: 策略名称
            mode: 交易模式
        """
        self.ticker = ticker
        self.strategy_name = strategy_name
        self.mode = mode
        self.process_id = f"{ticker}_{strategy_name}_{mode}"
        
        # 创建独立的目录结构
        self.base_dir = Path("live_trading")
        self.logs_dir = self.base_dir / "logs"
        self.cache_dir = self.base_dir / "cache"
        self.charts_dir = self.base_dir / "charts"
        self.pids_dir = self.base_dir / "pids"
        
        # 创建目录
        for dir_path in [self.logs_dir, self.cache_dir, self.charts_dir, self.pids_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 文件路径
        self.log_file = self.logs_dir / f"{self.process_id}.log"
        self.cache_file = self.cache_dir / f"{self.process_id}_cache.json"
        self.chart_file = self.charts_dir / f"{self.process_id}.html"
        self.pid_file = self.pids_dir / f"{self.process_id}.pid"
    
    def setup_logging(self):
        """设置独立的日志系统"""
        import logging
        
        # 创建 logger
        logger = logging.getLogger(f"live_runner_{self.process_id}")
        logger.setLevel(logging.INFO)
        
        # 清除现有的 handlers
        logger.handlers = []
        
        # 文件 handler
        file_handler = logging.FileHandler(self.log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        
        # 控制台 handler（带进程标识）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 格式化器（包含进程标识）
        formatter = logging.Formatter(
            f'[{self.ticker}] %(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def write_pid(self):
        """写入进程 ID"""
        import os
        self.pid_file.write_text(str(os.getpid()))
    
    def remove_pid(self):
        """删除进程 ID 文件"""
        if self.pid_file.exists():
            self.pid_file.unlink()
    
    def is_running(self) -> bool:
        """检查该配置是否已经在运行"""
        if not self.pid_file.exists():
            return False
        
        try:
            pid = int(self.pid_file.read_text().strip())
            # 检查进程是否存在
            import os
            import signal
            os.kill(pid, 0)  # 发送信号 0 检查进程
            return True
        except (ProcessLookupError, ValueError):
            # 进程不存在，清理 PID 文件
            self.remove_pid()
            return False
    
    def get_resources(self) -> dict:
        """获取所有资源路径"""
        return {
            'log_file': str(self.log_file),
            'cache_file': str(self.cache_file),
            'chart_file': str(self.chart_file),
            'pid_file': str(self.pid_file),
            'process_id': self.process_id
        }
    
    def cleanup(self):
        """清理资源"""
        self.remove_pid()


# ==========================================
# 4. 图表更新线程（带进程隔离）
# ==========================================

class ChartUpdater(threading.Thread):
    """图表更新线程 - 定期更新图表"""
    
    def __init__(self, 
                 visualizer: SimpleChartVisualizer,
                 strategy,
                 position_manager: PositionManager,
                 ticker: str,
                 logger,
                 update_interval: int = 30):
        """
        初始化图表更新器
        
        Args:
            visualizer: 可视化工具
            strategy: 策略实例
            position_manager: 仓位管理器
            ticker: 股票代码
            logger: 日志记录器
            update_interval: 更新间隔（秒）
        """
        super().__init__()
        self.visualizer = visualizer
        self.strategy = strategy
        self.position_manager = position_manager
        self.ticker = ticker
        self.logger = logger
        self.update_interval = update_interval
        self._running = True
        self.daemon = True
    
    def run(self):
        """运行图表更新循环"""
        self.logger.info(f"图表更新线程启动 (每 {self.update_interval} 秒更新)")
        
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
                self.logger.warning(f"图表更新错误: {e}")
                time.sleep(self.update_interval)
    
    def stop(self):
        """停止图表更新"""
        self._running = False


# ==========================================
# 5. 策略创建函数
# ==========================================

def create_strategy(strategy_name: str, cache: TradingCache = None, logger=None):
    """创建策略实例"""
    if strategy_name not in STRATEGY_CONFIGS:
        raise ValueError(f"未知策略: {strategy_name}. 可选: {list(STRATEGY_CONFIGS.keys())}")
    
    config = STRATEGY_CONFIGS[strategy_name]
    strategy_class = config['class']
    params = config['params']
    
    if logger:
        logger.info(f"策略: {config['name']}")
        logger.info(f"描述: {config['description']}")
        logger.info(f"参数: {params}")
    
    return strategy_class(**params)


# ==========================================
# 6. 信号回调函数
# ==========================================

def create_signal_callback(logger):
    """创建信号回调函数"""
    def on_signal_received(signal_dict: dict, price: float, timestamp: datetime):
        """信号回调函数"""
        signal = signal_dict.get('signal', 'UNKNOWN')
        confidence = signal_dict.get('confidence_score', 0)
        
        if signal in ['BUY', 'SELL', 'SHORT', 'COVER']:
            logger.info(f"交易信号: {signal} @ ${price:.2f} (置信度: {confidence}/10)")
    
    return on_signal_received


# ==========================================
# 7. 主函数
# ==========================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='实盘交易运行器 - 支持多进程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 单进程
  python live_runner.py --strategy moderate --ticker TSLA --mode paper
  
  # 多进程（同时交易多个股票）
  python live_runner.py --strategy moderate --ticker TSLA --mode paper &
  python live_runner.py --strategy moderate --ticker AAPL --mode paper &
  python live_runner.py --strategy moderate --ticker NVDA --mode paper &
  
  # 查看所有运行中的进程
  ls live_trading/pids/
  
  # 查看特定股票的日志
  tail -f live_trading/logs/TSLA_moderate_paper.log
        """
    )
    
    parser.add_argument('--strategy', type=str, default='moderate',
                       choices=list(STRATEGY_CONFIGS.keys()),
                       help='选择策略 (默认: moderate)')
    
    parser.add_argument('--ticker', type=str, required=True,
                       help='股票代码 (必填)')
    
    parser.add_argument('--mode', type=str, default='paper',
                       choices=['paper', 'live', 'simulation'],
                       help='交易模式: paper(模拟盘)/live(实盘)/simulation(本地模拟)')
    
    parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL_SECONDS,
                       help=f'策略运行间隔（秒，默认: {DEFAULT_INTERVAL_SECONDS}）')
    
    parser.add_argument('--no-chart', action='store_true',
                       help='禁用实时图表')
    
    parser.add_argument('--force', action='store_true',
                       help='强制启动（即使该配置已在运行）')
    
    args = parser.parse_args()
    
    # 获取配置
    TICKER = args.ticker.upper()
    TRADING_MODE = args.mode
    SELECTED_STRATEGY = args.strategy
    INTERVAL_SECONDS = args.interval
    ENABLE_CHART = not args.no_chart
    FORCE_START = args.force
    
    # ==========================================
    # A. 初始化多进程资源管理器
    # ==========================================
    
    resource_mgr = ProcessResourceManager(TICKER, SELECTED_STRATEGY, TRADING_MODE)
    resources = resource_mgr.get_resources()
    
    # 检查是否已经在运行
    if resource_mgr.is_running() and not FORCE_START:
        print(f"\n⚠️ 错误: {TICKER} (策略: {SELECTED_STRATEGY}, 模式: {TRADING_MODE}) 已经在运行！")
        print(f"   PID 文件: {resources['pid_file']}")
        print(f"   如需强制启动，请使用 --force 参数")
        print(f"   或先停止现有进程: kill $(cat {resources['pid_file']})")
        sys.exit(1)
    
    # 写入 PID
    resource_mgr.write_pid()
    
    # 设置日志
    logger = resource_mgr.setup_logging()
    
    strategy_config = STRATEGY_CONFIGS[SELECTED_STRATEGY]
    
    logger.info("="*60)
    logger.info("🚀 实盘交易系统初始化")
    logger.info("="*60)
    logger.info(f"进程 ID: {resources['process_id']}")
    logger.info(f"股票代码: {TICKER}")
    logger.info(f"交易模式: {TRADING_MODE.upper()}")
    logger.info(f"策略: {strategy_config['name']}")
    logger.info(f"运行间隔: {INTERVAL_SECONDS} 秒")
    logger.info(f"实时图表: {'开启' if ENABLE_CHART else '关闭'}")
    logger.info(f"日志文件: {resources['log_file']}")
    logger.info(f"缓存文件: {resources['cache_file']}")
    if ENABLE_CHART:
        logger.info(f"图表文件: {resources['chart_file']}")
    
    if TRADING_MODE == 'live':
        logger.warning("⚠️"*20)
        logger.warning("警告: 您正在使用实盘模式！")
        logger.warning("所有交易将使用真实资金！")
        logger.warning("⚠️"*20)
        
        confirm = input(f"\n确认启动 {TICKER} 实盘交易? (输入 'YES' 确认): ")
        if confirm != 'YES':
            logger.info("已取消启动")
            resource_mgr.cleanup()
            sys.exit(0)
    
    try:
        # B. Data Fetcher
        is_paper = TRADING_MODE in ['paper', 'simulation']
        data_fetcher = AlpacaDataFetcher(paper=is_paper) if TRADING_MODE != 'simulation' else None
        
        # C. Cache System（使用独立的缓存文件）
        cache = TradingCache(str(resources['cache_file']))
        
        # D. Executor & Position Manager
        if TRADING_MODE == 'simulation':
            logger.info("执行器: 本地模拟")
            executor = SimulationExecutor(FINANCE_PARAMS)
            position_manager = PositionManager(executor, FINANCE_PARAMS)
            data_fetcher = AlpacaDataFetcher(paper=True)
        elif TRADING_MODE == 'paper':
            logger.info("执行器: Alpaca 模拟盘 (Paper)")
            executor = AlpacaExecutor(paper=True, max_allocation_rate=FINANCE_PARAMS['MAX_ALLOCATION'])
            position_manager = PositionManager(executor, FINANCE_PARAMS, data_fetcher=data_fetcher)
        elif TRADING_MODE == 'live':
            logger.info("执行器: Alpaca 实盘 (Live)")
            executor = AlpacaExecutor(paper=False, max_allocation_rate=FINANCE_PARAMS['MAX_ALLOCATION'])
            position_manager = PositionManager(executor, FINANCE_PARAMS, data_fetcher=data_fetcher)
        else:
            raise ValueError(f"无效的交易模式: {TRADING_MODE}")
        
        # E. 从 API 同步仓位状态
        if SYNC_POSITION_ON_START and TRADING_MODE in ['paper', 'live']:
            logger.info(f"正在从 API 同步 {TICKER} 仓位状态...")
            sync_success = position_manager.sync_from_api(TICKER)
            if not sync_success:
                logger.warning("仓位同步失败，将使用本地初始状态")
        
        # F. Strategy
        logger.info("策略初始化...")
        strategy = create_strategy(SELECTED_STRATEGY, cache, logger)
        
        # G. 图表可视化（使用独立的图表文件）
        visualizer = None
        chart_updater = None
        
        if ENABLE_CHART:
            logger.info("初始化实时图表...")
            visualizer = SimpleChartVisualizer(
                ticker=TICKER,
                output_file=str(resources['chart_file']),
                auto_open=AUTO_OPEN_BROWSER
            )
            visualizer.set_initial_capital(FINANCE_PARAMS['INITIAL_CAPITAL'])
            
            # 启动图表更新线程
            chart_updater = ChartUpdater(
                visualizer=visualizer,
                strategy=strategy,
                position_manager=position_manager,
                ticker=TICKER,
                logger=logger,
                update_interval=CHART_UPDATE_INTERVAL
            )
            chart_updater.start()
            logger.info(f"图表更新间隔: {CHART_UPDATE_INTERVAL} 秒")
        
        # ==========================================
        # H. Create and Run Live Engine
        # ==========================================
        
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
            on_signal_callback=create_signal_callback(logger)
        )
        
        # 运行引擎
        report = live_engine.run()
        
    except KeyboardInterrupt:
        logger.warning("收到 Ctrl+C，正在停止...")
    except Exception as e:
        logger.error(f"运行时错误: {e}", exc_info=True)
    finally:
        # 停止图表更新线程
        if chart_updater:
            logger.info("停止图表更新...")
            chart_updater.stop()
            chart_updater.join(timeout=2)
        
        # 清理资源
        resource_mgr.cleanup()
    
    # ==========================================
    # I. Final Report
    # ==========================================
    
    logger.info("="*60)
    logger.info("💰 最终结果")
    logger.info("="*60)
    logger.info(f"运行时长: {report.get('runtime_seconds', 0) / 60:.1f} 分钟")
    logger.info(f"迭代次数: {report.get('iterations', 0)}")
    logger.info(f"交易信号: {report.get('signals', 0)}")
    logger.info(f"执行交易: {report.get('trades_executed', 0)}")
    logger.info(f"最终权益: ${report.get('final_equity', 0):,.2f}")
    logger.info("="*60)
    
    # 打印交易日志
    trade_log = position_manager.get_trade_log()
    if trade_log is not None and not trade_log.empty:
        logger.info("📝 交易日志:")
        for _, row in trade_log.iterrows():
            logger.info(f"  {row['time'].strftime('%Y-%m-%d %H:%M')} | "
                       f"{row['type']:6s} | {row['qty']:3.0f} 股 @ ${row['price']:.2f} | "
                       f"盈亏: ${row['net_pnl']:+.2f}")
        
        # 交易统计
        completed_trades = trade_log[trade_log['type'].isin(['SELL', 'COVER'])]
        if not completed_trades.empty and 'net_pnl' in completed_trades.columns:
            winning_trades = completed_trades[completed_trades['net_pnl'] > 0]
            losing_trades = completed_trades[completed_trades['net_pnl'] < 0]
            
            win_rate = len(winning_trades) / len(completed_trades) * 100 if len(completed_trades) > 0 else 0
            total_pnl = completed_trades['net_pnl'].sum()
            
            logger.info(f"📈 交易统计:")
            logger.info(f"  完成交易: {len(completed_trades)}")
            logger.info(f"  盈利交易: {len(winning_trades)}")
            logger.info(f"  亏损交易: {len(losing_trades)}")
            logger.info(f"  胜率: {win_rate:.1f}%")
            logger.info(f"  总盈亏: ${total_pnl:,.2f}")
            
            if len(winning_trades) > 0:
                logger.info(f"  平均盈利: ${winning_trades['net_pnl'].mean():.2f}")
            if len(losing_trades) > 0:
                logger.info(f"  平均亏损: ${losing_trades['net_pnl'].mean():.2f}")
    else:
        logger.info("🤷 无交易记录")
    
    if ENABLE_CHART:
        logger.info(f"📊 最终图表: {resources['chart_file']}")
    
    logger.info("✅ 程序结束")


if __name__ == '__main__':
    main()