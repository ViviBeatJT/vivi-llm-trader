# src/test/backtest.py

from datetime import datetime, timezone, timedelta
from src.cache.trading_cache import load_cache, save_cache
from src.strategies.mean_reversion_strategy import get_mean_reversion_signal
from src.executor.base_executor import BaseExecutor
from src.data.alpaca_data_fetcher import get_latest_price # 导入实时价格获取函数
from typing import Optional

def backtest_arbitrary_period(cache: dict,
                              ticker: str,
                              start_dt: datetime,
                              end_dt: datetime,
                              executor: BaseExecutor,
                              step_minutes: int = 5,
                              is_live_run: bool = False, # 新增参数：是否为实时运行模式
                              delay_seconds: int = 15):
    """
    自动回测/运行指定时间段内的交易状态。
    在回测模式下，使用时间戳模拟历史数据；在实时模式下，获取实时价格。

    Args:
        cache: Gemini 响应缓存。
        ticker: 股票代码。
        start_dt: 运行的起始时间。
        end_dt: 运行的结束时间。
        executor: 交易执行器实例 (SimulationExecutor 或 AlpacaExecutor)。
        step_minutes: 每次循环的时间步长（分钟）。
        is_live_run: 如果为 True，则调用 Alpaca API 获取实时价格。
        delay_seconds: 每次 LLM 调用后的延迟时间，用于遵守速率限制。
    """
    results = []

    # 确保起始时间小于等于结束时间
    if start_dt >= end_dt and not is_live_run:
        print("❌ 错误：起始时间必须早于结束时间（回测模式）。")
        return results, pd.DataFrame(), executor.get_account_status(0.0)['equity']

    # 确保时间对象带有 UTC 时区信息
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    current_time = start_dt
    time_step = timedelta(minutes=step_minutes)

    # 打印运行范围
    run_mode = "实时运行 (Live/Paper)" if is_live_run else "历史回测"
    print(f"\n--- 🚀 开始 {run_mode} ({ticker}) ---")
    print(
        f"运行范围: {start_dt.strftime('%Y-%m-%d %H:%M UTC')} 至 {end_dt.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"步长: {step_minutes} 分钟")
    print("-" * 30)

    while current_time <= end_dt or is_live_run:
        if is_live_run:
            # 实时模式下，使用当前时间作为策略分析时间点
            time_for_signal = datetime.now(timezone.utc).astimezone(timezone.utc)
            # 实时获取最新价格
            current_price = get_latest_price(ticker)
        else:
            # 回测模式下，使用循环时间
            time_for_signal = current_time
            
            # 回测模式下，假设价格数据存储在缓存中，通过时间戳查找
            # 注意：这里需要一个机制来从缓存中获取当前时间点的价格
            # 简单回测场景：直接从 LLM 调用的 K线数据中提取最后一个收盘价（近似处理）
            current_price = 0.0 # 稍后从信号结果中更新
            
        print(f"--- 📊 正在处理时间点: {time_for_signal.strftime('%Y-%m-%d %H:%M UTC')} ---")
        
        # 1. 策略调用（获取信号）
        # time_for_signal 决定了 LLM 分析的 K线数据的结束时间点
        signal_result, current_price = get_mean_reversion_signal(
            cache, ticker, time_for_signal, delay_seconds)
        
        signal = signal_result.get('signal')
        confidence = signal_result.get('confidence_score', 0)
        reason = signal_result.get('reason', 'N/A')

        # 尝试从信号结果中提取价格 (仅用于回测模式的近似价格)
        if not is_live_run and 'price' in signal_result:
             # 假设 LLM 结果中可以包含当前收盘价
             current_price = signal_result.get('price', 0.0) 
        elif not is_live_run:
             # 如果是回测模式，并且没有价格，则跳过
             # 实际项目中，这里应该从历史数据中精确查找
             print("⚠️ 回测模式下，无法从信号结果中获取当前价格。跳过本周期。")
             current_time += time_step
             continue

        if current_price <= 0.0:
            print("⚠️ 价格无效，跳过本周期。")
        elif signal in ["BUY", "SELL"]:
            # 2. 执行交易
            success = executor.execute_trade(
                timestamp=time_for_signal,
                signal=signal,
                current_price=current_price
            )
            print(f"    交易信号: {signal:4} | 价格: ${current_price:.2f} | 执行结果: {'成功' if success else '失败'}")
            
            # 记录交易信号和结果
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
    # 获取最终的账户状态
    final_status = executor.get_account_status(current_price=current_price)
    final_equity = final_status.get('equity', 0.0)
    trade_log_df = executor.get_trade_log() # 从 Executor 获取交易日志

    print("\n--- ✅ 运行完成。结果总结 ---")
    # ... (后继打印逻辑保持不变)

    # 打印格式化后的结果 (保持总结逻辑不变)
    total_signals = len(results)
    buy_count = sum(1 for r in results if r['signal'] == 'BUY')
    sell_count = sum(1 for r in results if r['signal'] == 'SELL')

    print(f"总测试点数: {total_signals}")
    print(f"买入信号 (BUY): {buy_count} 次")
    print(f"卖出信号 (SELL): {sell_count} 次")
    print("-" * 30)

    return results, trade_log_df, final_equity


if __name__ == '__main__':
    # ----------------------------------------------------\n
    # 示例运行 (如果需要一个单独的测试入口)
    # ----------------------------------------------------\n
    # 运行此文件需要 SimulationExecutor 的定义，此处仅保留函数定义
    print("请通过 backtest_runner.py 运行完整的交易系统。")