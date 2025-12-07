# src/test/backtest.py

from datetime import datetime, timezone, timedelta
from src.cache.trading_cache import TradingCache # 导入 TradingCache 类
from src.strategies.mean_reversion_strategy import get_mean_reversion_signal
# 从 manager 导入 PositionManager (新增)
from src.manager.position_manager import PositionManager 
from src.executor.base_executor import BaseExecutor # 导入 BaseExecutor (用于旧代码兼容，但不再直接使用)
from src.data.alpaca_data_fetcher import AlpacaDataFetcher # 导入 AlpacaDataFetcher 类
from typing import Optional
import pandas as pd # 确保导入 pandas


# 注意：旧的 executor: BaseExecutor 参数已替换为 position_manager: PositionManager
def backtest_arbitrary_period(cache: TradingCache, 
                              ticker: str,
                              start_dt: datetime,
                              end_dt: datetime,
                              position_manager: PositionManager, # 核心改动：接收 PositionManager
                              data_fetcher: AlpacaDataFetcher, 
                              step_minutes: int = 5,
                              is_live_run: bool = False, 
                              delay_seconds: int = 15):
    """
    自动回测/运行指定时间段内的交易状态。
    在回测模式下，使用时间戳模拟历史数据；在实时模式下，获取实时价格。
    交易执行和状态管理现在完全通过 PositionManager 进行。

    Args:
        cache: Gemini 响应缓存 (TradingCache 实例)。
        ticker: 股票代码。
        start_dt: 运行的起始时间。
        end_dt: 运行的结束时间。
        position_manager: 仓位管理器实例，负责状态管理和交易执行。
        data_fetcher: AlpacaDataFetcher 实例，用于获取实时价格。
        step_minutes: 每次循环的时间步长（分钟）。
        is_live_run: 如果为 True，则为实时运行模式。
        delay_seconds: 实时模式下的等待时间（秒）。
    """
    
    current_time = start_dt
    results = [] # 记录所有信号
    
    # 假设 PositionManager 已经初始化，获取其初始状态
    initial_status = position_manager.get_account_status(current_price=0.0) 
    initial_cash = initial_status.get('cash', 0.0)
    
    print(f"📈 开始运行: {start_dt} 至 {end_dt} (步长: {step_minutes} 分钟) | 初始现金: ${initial_cash:,.2f}")
    
    # 获取 PositionManager 内部的 executor 类型，用于打印
    executor_type = position_manager.executor.__class__.__name__

    while current_time <= end_dt:
        time_for_signal = current_time.astimezone(timezone.utc)
        
        # 1. 获取最新价格 (回测使用缓存/历史，实时使用API)
        current_price = data_fetcher.get_price_data(
            ticker=ticker,
            timestamp=time_for_signal,
            cache=cache,
            is_live_run=is_live_run,
            delay_seconds=delay_seconds
        )
        
        if current_price is None or current_price <= 0:
            print(f"❌ 警告: 在 {time_for_signal} 无法获取有效价格，跳过此时间点。")
            current_time += timedelta(minutes=step_minutes)
            continue
            
        # 2. 获取当前账户状态 (使用 PositionManager)
        current_status = position_manager.get_account_status(current_price=current_price)
        current_cash = current_status['cash']
        current_position = current_status['position']
        avg_cost = current_status['avg_cost']
        
        # 3. 生成交易信号
        signal, confidence, reason = get_mean_reversion_signal(
            timestamp=time_for_signal,
            current_price=current_price,
            current_position=current_position,
            current_cash=current_cash,
            avg_cost=avg_cost,
            executor_type=executor_type # 传递执行器类型
        )
        
        # 4. 执行交易 (通过 PositionManager 统一处理)
        if signal in ["BUY", "SELL"]:
            print(f"🔥 交易信号: {signal:4} | 价格: ${current_price:.2f} | 理由: {reason}")
            
            # **核心改动：让 PositionManager 来执行交易并更新自己的状态**
            # PositionManager 会内部调用 BaseExecutor.execute_trade，然后更新自身的 cash/position/avg_cost。
            trade_result = position_manager.execute_trade_and_update_state(
                timestamp=time_for_signal,
                signal=signal,
                current_price=current_price,
            )
            
            success = trade_result['executed']
            
            if success:
                print(f"    ✅ 交易执行成功。{trade_result['log_message']}")
            else:
                print(f"    ❌ 交易执行失败。{trade_result['log_message']}")
            
            # 记录信号和执行结果
            results.append({
                'timestamp_utc': time_for_signal,
                'signal': signal,
                'confidence': confidence,
                'reason': reason,
                'executed': success,
                'price': current_price
            })
        else:
            print(f"    交易信号: {signal:4} | 价格: ${current_price:.2f} | 观望 (HOLD)")
            
            # 记录 HOLD 信号
            results.append({
                'timestamp_utc': time_for_signal,
                'signal': signal,
                'confidence': confidence,
                'reason': reason,
                'executed': False,
                'price': current_price
            })

        if is_live_run:
            # 实时模式下只运行一次，然后退出循环
            break
        
        # 回测模式下，前进到下一个时间点
        current_time += time_step

    # --- 最终总结 ---
    # 获取最终的账户状态 (使用 PositionManager)
    final_status = position_manager.get_account_status(current_price=current_price)
    final_equity = final_status.get('equity', 0.0)
    trade_log_df = position_manager.get_trade_log() # 从 PositionManager 获取交易日志

    print("\n--- ✅ 运行完成。结果总结 ---")
    
    # 打印格式化后的结果 (保持总结逻辑不变)
    total_signals = len(results)
    buy_count = sum(1 for r in results if r['signal'] == 'BUY')
    sell_count = sum(1 for r in results if r['signal'] == 'SELL')

    print(f"总测试点数: {total_signals}")
    print(f"买入信号 ({buy_count}), 卖出信号 ({sell_count})")
    
    return final_equity, trade_log_df