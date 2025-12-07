from datetime import datetime, timezone
# 导入 TradingCache 类
from src.cache.trading_cache import TradingCache 
# 导入 PositionManager 类 (假设它在 manager 目录下)
from src.manager.position_manager import PositionManager 
from src.test.backtest import backtest_arbitrary_period
from src.executor.simulation_executor import SimulationExecutor # 模拟执行器（仓位管理）
from src.executor.alpaca_trade_executor import AlpacaExecutor # 实盘/纸盘执行器
import pandas as pd
import os
from dotenv import load_dotenv

# 导入 AlpacaDataFetcher 类
from src.data.alpaca_data_fetcher import AlpacaDataFetcher 

load_dotenv() # 确保加载了 .env 文件中的 Alpaca API 密钥

## --- 1. 财务参数设置（供 SimulationExecutor 使用） ---
# 注意：这些参数仅在 IS_BACKTEST_MODE = True 时生效
INITIAL_CAPITAL = 100000.0  # 初始资金 (USD)
COMMISSION_RATE = 0.0003    # 单边手续费率 (万分之三)
SLIPPAGE_RATE = 0.0001      # 模拟滑点 (万分之一)
MIN_LOT_SIZE = 100          # 最小交易单位（股/手）
MAX_ALLOCATION = 0.2        # 每次交易最大动用资金比例（例如总资产的20%）
STAMP_DUTY_RATE = 0.001     # 印花税率 (仅卖出时收取，假设为 A 股标准)

# 将所有财务参数打包
FINANCE_PARAMS = {
    'INITIAL_CAPITAL': INITIAL_CAPITAL,
    'COMMISSION_RATE': COMMISSION_RATE,
    'SLIPPAGE_RATE': SLIPPAGE_RATE,
    'MIN_LOT_SIZE': MIN_LOT_SIZE,
    'MAX_ALLOCATION': MAX_ALLOCATION,
    'STAMP_DUTY_RATE': STAMP_DUTY_RATE,
}

## --- 2. 运行配置 ---
TICKER = "TSLA"  # 交易标的
START_TIME = datetime(2023, 10, 1, tzinfo=timezone.utc)
END_TIME = datetime(2023, 10, 31, tzinfo=timezone.utc)
STEP_MINUTES = 5

# 设置运行模式：True 为回测模式，False 为实盘/纸盘模式
IS_BACKTEST_MODE = True 
# IS_BACKTEST_MODE = False # 启用 Alpaca 实时运行

# ----------------------------------------------------
# 3. 初始化执行器和仓位管理器 (UPDATED LOGIC)
# ----------------------------------------------------

# 根据模式选择交易执行器
if IS_BACKTEST_MODE:
    print("\n--- 🔧 运行模式: 回测 (SimulationExecutor) ---")
    # SimulationExecutor 需要 FINANCE_PARAMS 来计算交易细节
    executor = SimulationExecutor(FINANCE_PARAMS)
else:
    print("\n--- 🚀 运行模式: 实盘/纸盘 (AlpacaExecutor) ---")
    # AlpacaExecutor 需要 paper 标志和最大分配比例
    executor = AlpacaExecutor(paper=True, max_allocation_rate=MAX_ALLOCATION)
    
# 使用选定的 executor 和财务参数初始化 PositionManager
# PositionManager 成为状态管理和交易执行的统一入口
position_manager = PositionManager(executor, FINANCE_PARAMS) 

# 初始化数据获取器
data_fetcher = AlpacaDataFetcher()

# ----------------------------------------------------
# 4. 执行回测/运行 (UPDATED CALL)
# ----------------------------------------------------

# 自动处理缓存
cache = TradingCache(ticker, os.path.join('cache', f'{ticker}_trading_cache.json'))
initial_cache_size = len(cache.data)

final_equity, trade_log_df = backtest_arbitrary_period(
    cache=cache,
    ticker=TICKER,
    start_dt=START_TIME,
    end_dt=END_TIME,
    # 将 PositionManager 实例传入
    position_manager=position_manager, 
    data_fetcher=data_fetcher,
    step_minutes=STEP_MINUTES,
    is_live_run=not IS_BACKTEST_MODE
)

# ----------------------------------------------------
# 5. 缓存处理
# ----------------------------------------------------

if len(cache.data) > initial_cache_size:
    print(f"\n--- 💾 发现 {len(cache.data) - initial_cache_size} 个新缓存条目。正在保存... ---")
    cache.save()
else:
    print("\n--- 📝 未发现新缓存条目，跳过文件保存。 ---")

# ----------------------------------------------------
# 6. 结果打印与总结
# ----------------------------------------------------

total_net_pnl = final_equity - INITIAL_CAPITAL

print("\n--- 💰 回测/运行结果摘要 ---")
# 打印 PositionManager 内部的执行器类型
print(f"执行模式: {position_manager.executor.__class__.__name__}") 
print(f"初始资产: {INITIAL_CAPITAL:,.2f} USD")
print(f"最终资产: {final_equity:,.2f} USD")
print(f"总净收益: {total_net_pnl:,.2f} USD")
print("-" * 30)

if trade_log_df is not None and not trade_log_df.empty:
    print("\n详细交易日志:")
    # 只显示关键列，并格式化输出
    log_display = trade_log_df[['time', 'type', 'qty', 'price', 'fee', 'net_pnl', 'current_pos']]
    log_display['time'] = log_display['time'].dt.strftime('%Y-%m-%d %H:%M')
    print(log_display.to_markdown(index=False, floatfmt=".2f"))
    
else:
    print("未发生任何交易。")

# 最终状态
# 在这里传入 0.0 作为...
# ... [rest of the file content]