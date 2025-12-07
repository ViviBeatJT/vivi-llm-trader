from datetime import datetime, timezone, timedelta
from src.cache.trading_cache import load_cache, save_cache
from src.strategies.mean_reversion_strategy import get_mean_reversion_signal

def backtest_arbitrary_period(cache: dict, ticker: str, start_dt: datetime, end_dt: datetime, step_minutes: int = 5, delay_seconds: int = 15):
    """
    自动回测指定时间段内的交易状态，每 5 分钟执行一次 LLM 策略。

    Args:
        ticker: 股票代码 (e.g., 'TSLA')
        start_dt: 回测的起始时间 (必须是带时区的 datetime 对象，推荐 UTC)。
        end_dt: 回测的结束时间 (必须是带时区的 datetime 对象，推荐 UTC)。
    """
    results = []

    # 确保起始时间小于等于结束时间
    if start_dt >= end_dt:
        print("❌ 错误：起始时间必须早于结束时间。")
        return results

    # 确保时间对象带有 UTC 时区信息
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    current_time = start_dt

    # 打印回测范围
    print(f"\n--- 🚀 开始回测 ({ticker}) ---")
    print(
        f"回测范围: {start_dt.strftime('%Y-%m-%d %H:%M UTC')} 至 {end_dt.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"执行步长: {step_minutes} 分钟 | API 延迟: {delay_seconds} 秒")  # 打印新的延迟信息
    print("-" * 50)

    # 无限循环直到达到结束时间
    time_step = timedelta(minutes=step_minutes)

    while current_time <= end_dt:
        print(
            f"\n[TIME: {current_time.strftime('%Y-%m-%d %H:%M UTC')}] 正在获取信号...")

        # 调用核心策略函数，使用当前时间点作为数据结束时间
        # lookback_minutes 参数在 get_mean_reversion_signal 内部调用 get_latest_bars 时使用
        signal_result = get_mean_reversion_signal(
            cache=cache,
            ticker=ticker, end_dt=current_time, delay_seconds=delay_seconds)

        # 记录结果
        results.append({
            'timestamp_utc': current_time,
            'signal': signal_result.get('signal', 'ERROR'),
            'confidence': signal_result.get('confidence_score', 0),
            'reason': signal_result.get('reason', 'N/A'),
            'error': signal_result.get('error', None)
        })

        # 步进 5 分钟 (与 K 线周期一致)
        current_time += time_step

    print("\n--- ✅ 回测完成。结果总结 ---")

    # 打印格式化后的结果 (保持总结逻辑不变)
    total_signals = len(results)
    buy_count = sum(1 for r in results if r['signal'] == 'BUY')
    sell_count = sum(1 for r in results if r['signal'] == 'SELL')

    print(f"总测试点数: {total_signals}")
    print(f"买入信号 (BUY): {buy_count} 次")
    print(f"卖出信号 (SELL): {sell_count} 次")
    print("-" * 30)

    action_signals = [r for r in results if r['signal'] in ['BUY', 'SELL']]

    if action_signals:
        print("详细交易信号列表:")
        for r in action_signals:
            print(
                f"  {r['timestamp_utc'].strftime('%Y-%m-%d %H:%M UTC')} | {r['signal']:4} | 置信度: {r['confidence']}/10 | 原因: {r['reason']}")
    else:
        print("全时间段内无有效 BUY/SELL 信号。")

    return results


if __name__ == '__main__':
    # ----------------------------------------------------
    # 设置回测日期和股票
    # ----------------------------------------------------
    # 尝试加载缓存
    cache = load_cache()

    TICKER = "TSLA"

    # 回测起始时间
    START_DATE = datetime(2025, 12, 4, 19, 0, 0, tzinfo=timezone.utc)

    # 回测结束时间
    END_DATE = datetime(2025, 12, 4, 20, 0, 0,
                        tzinfo=timezone.utc)  # 仅测试 12 月 4 日收盘前

    # 设置步长：例如，每 15 分钟执行一次策略
    STEP_MINUTES = 5

    # 执行指定时间段回测
    all_signals = backtest_arbitrary_period(
        cache,
        ticker=TICKER,
        start_dt=START_DATE,
        end_dt=END_DATE,
        step_minutes=STEP_MINUTES  # 传入新的参数
    )

    save_cache(cache)
