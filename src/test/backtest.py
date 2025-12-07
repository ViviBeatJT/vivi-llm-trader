# src/test/backtest.py

from datetime import datetime, timezone, timedelta
from src.cache.trading_cache import TradingCache # 导入 TradingCache 类
from src.strategies.mean_reversion_strategy import get_mean_reversion_signal
from src.executor.base_executor import BaseExecutor
from src.data.alpaca_data_fetcher import AlpacaDataFetcher # 导入 AlpacaDataFetcher 类
from typing import Optional
import pandas as pd # 确保导入 pandas

def backtest_arbitrary_period(cache: TradingCache, # 更改参数类型为 TradingCache
                              ticker: str,
                              start_dt: datetime,
                              end_dt: datetime,
                              executor: BaseExecutor,
                              data_fetcher: AlpacaDataFetcher, # 新增参数：数据获取器实例
                              step_minutes: int = 5,
                              is_live_run: bool = False, # 新增参数：是否为实时运行模式
                              delay_seconds: int = 15):
    """
    自动回测/运行指定时间段内的交易状态。
    在回测模式下，使用时间戳模拟历史数据；在实时模式下，获取实时价格。

    Args:
        cache: Gemini 响应缓存 (TradingCache 实例)。
        ticker: 股票代码。
        start_dt: 运行的起始时间。
        end_dt: 运行的结束时间。
        executor: 交易执行器实例 (SimulationExecutor 或 AlpacaExecutor)。
        data_fetcher: AlpacaDataFetcher 实例，用于获取实时价格。
        step_minutes: 每次循环的时间步长（分钟）。
        is_live_run: 如果为 True，则调用 Alpaca API 获取实时价格。
        delay_seconds: 每次 LLM 调用后的延迟时间，用于遵守速率限制。
    """
    results = []

    # 确保起始时间小于等于结束时间
    if start_dt >= end_dt and not is_live_run:
        print("❌ 错误：起始时间必须早于结束时间（回测模式）。")
        # 返回空的 results, None 日志, 和当前的 equity
        return results, None, executor.get_account_status(0.0)['equity'] 

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

    # 确保 current_price 在循环开始前有值
    current_price = 0.0

    while current_time <= end_dt or is_live_run:
        if is_live_run:
            # 实时模式下，使用当前时间作为策略分析时间点
            time_for_signal = datetime.now(timezone.utc).astimezone(timezone.utc)
            # **更新：使用 data_fetcher 实例调用 get_latest_price**
            current_price = data_fetcher.get_latest_price(ticker) 
        else:
            # 回测模式下，使用循环时间
            time_for_signal = current_time
            # 回测模式下，初始价格为 0.0，策略函数会返回对应时间点的收盘价
            current_price = 0.0 
            
        print(f"--- 📊 正在处理时间点: {time_for_signal.strftime('%Y-%m-%d %H:%M UTC')} ---")
        
        # 1. 策略调用（获取信号）
        # time_for_signal 决定了 LLM 分析的 K线数据的结束时间点
        # get_mean_reversion_signal 函数返回 (signal_result, latest_price)
        signal_result, current_price = get_mean_reversion_signal(
            cache, ticker, time_for_signal, lookback_minutes=60, delay_seconds=delay_seconds) # lookback_minutes 默认值 60
        
        signal = signal_result.get('signal')
        confidence = signal_result.get('confidence_score', 0)
        reason = signal_result.get('reason', 'N/A')

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
    # 运行此文件需要 SimulationExecutor 的定义，此处仅保留函数定义
    print("请通过 backtest_runner.py 运行完整的交易系统。")