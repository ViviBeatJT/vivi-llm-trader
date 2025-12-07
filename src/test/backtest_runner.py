from datetime import datetime, timezone
from src.cache.trading_cache import load_cache
from src.test.backtest import backtest_arbitrary_period
from src.executor.simulation_executor import SimulationExecutor # 模拟执行器（仓位管理）
from src.executor.alpaca_trade_executor import AlpacaExecutor # 实盘/纸盘执行器
import pandas as pd
import os
from dotenv import load_dotenv

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

# --- 2. 模式切换开关（核心） ---
# True: 使用 SimulationExecutor 进行本地回测/模拟
# False: 使用 AlpacaExecutor 进行实盘 (需确保 Alpaca 客户端配置正确)
IS_BACKTEST_MODE = True 

# 如果使用 AlpacaExecutor，可以选择是否使用 paper 账户
ALPACA_PAPER_MODE = True


if __name__ == '__main__':
    # ----------------------------------------------------
    # 模式选择和执行器初始化
    # ----------------------------------------------------
    if IS_BACKTEST_MODE:
        print("💡 模式选择: 回测模拟 (SimulationExecutor)")
        # 使用 SimulationExecutor 进行回测
        executor = SimulationExecutor(FINANCE_PARAMS)
        
        # 回测需要明确的开始和结束时间
        START_DATE = datetime(2025, 12, 4, 19, 0, 0, tzinfo=timezone.utc)
        END_DATE = datetime(2025, 12, 4, 20, 0, 0, tzinfo=timezone.utc)
        
        # 初始资金从 FINANCE_PARAMS 中获取，用于最终 P&L 计算
        initial_capital = FINANCE_PARAMS.get('INITIAL_CAPITAL', 0.0)
        STEP_MINUTES = 5

    else:
        print(f"🚀 模式选择: Alpaca {'纸盘' if ALPACA_PAPER_MODE else '实盘'} (AlpacaExecutor)")
        
        # 检查必要的环境变量
        if not os.getenv('ALPACA_API_KEY_ID') or not os.getenv('ALPACA_SECRET_KEY'):
            print("❌ 错误：未配置 ALPACA_API_KEY_ID 或 ALPACA_SECRET_KEY。请检查 .env 文件。")
            exit()
            
        # 使用 AlpacaExecutor 进行实盘/纸盘交易
        executor = AlpacaExecutor(paper=ALPACA_PAPER_MODE, max_allocation_rate=MAX_ALLOCATION)
        
        # 实盘运行：通常只运行一次策略，或在一个无限循环中运行
        START_DATE = datetime.now(timezone.utc)
        # 仅测试一次，所以结束时间设为开始时间，backtest_arbitrary_period 会处理边界条件
        END_DATE = START_DATE 
        
        # 获取 Alpaca 账户的初始权益作为 P&L 计算基准
        # 注意：这里需要 API 调用来获取实时权益
        initial_status = executor.get_account_status(current_price=0.0) 
        initial_capital = initial_status.get('equity', 0.0)
        STEP_MINUTES = 1 # 实时交易可以更频繁

    # ----------------------------------------------------
    # 设置回测/运行参数
    # ----------------------------------------------------\
    cache = load_cache()
    TICKER = "TSLA"

    # 执行回测或实时运行
    # backtest_arbitrary_period 现在接受一个 executor 实例
    all_signals, trade_log_df, final_equity = backtest_arbitrary_period(
        cache,
        ticker=TICKER,
        start_dt=START_DATE,
        end_dt=END_DATE,
        executor=executor,  # 传入执行器实例
        step_minutes=STEP_MINUTES,
        is_live_run=not IS_BACKTEST_MODE, 
    )

    # ----------------------------------------------------
    # 结果打印与总结
    # ----------------------------------------------------
    
    total_net_pnl = final_equity - initial_capital
    
    print("\n--- 💰 回测/运行结果摘要 ---")
    print(f"执行模式: {executor.__class__.__name__}")
    print(f"初始资产: {initial_capital:,.2f} USD")
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
    # 在这里传入 0.0 作为 price 是因为我们只关心现金和持仓股数，最终权益已在上一步计算
    final_status = executor.get_account_status(current_price=0.0) 
    print(f"\n最终持仓概览: 现金 ${final_status['cash']:,.2f} | 剩余持仓 {final_status['position']:,.0f} 股")