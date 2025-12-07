# src/strategies/mean_reversion_strategy.py

import json
import os # 用于文件操作
from google import genai
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
import hashlib # 用于生成唯一的缓存键

# 导入 Alpaca 数据获取函数 (假设已修复 TimeFrame 导入问题)
from src.data.alpaca_data_fetcher import get_latest_bars
from google import genai
import time  # 新增：用于暂停执行

# 初始化 Gemini 客户端
load_dotenv()
client = genai.Client()

CACHE_FILE = 'gemini_cache.json'


def load_cache():
    """从本地文件加载 Gemini 响应缓存。"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                # 文件损坏时返回空字典
                return {}
    return {}


def save_cache(cache_data):
    """将 Gemini 响应缓存保存到本地文件。"""
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=4, ensure_ascii=False)

# 定义 LLM 输出结构 (沿用之前的情绪信号)


class TradingSignal(BaseModel):
    """交易信号模型"""
    signal: Literal["BUY", "SELL", "HOLD"] = Field(
        description="基于技术分析，给出买入、卖出或观望的交易信号。")
    confidence_score: int = Field(..., ge=1, le=10,
                                  description="对信号的自信程度评分，10为最高自信。")
    reason: str = Field(description="简要说明给出此信号的原因，必须基于布林带分析。")


# 定义 LLM 系统指令
SYSTEM_PROMPT = (
    "你是一位专业的量化交易员，专注于区间反转（Mean Reversion）策略。你的任务是分析提供的包含布林带（BB）和 RSI 指数（RSI(14)）的 K线数据表，"
    "并严格按照以下区间反转规则给出交易信号："

    "1. **强力买入 (BUY):** 满足以下至少两个条件时：\n"
    "   a. 收盘价连续触及或跌破布林带下轨 (Lower Band)。\n"
    "   b. RSI(14) 指数低于 30 (严重超卖区域)。\n"
    "   c. 最新价格相比前一个周期开始反弹（收盘价高于前一周期收盘价）。\n"

    "2. **强力卖出 (SELL):** 满足以下至少两个条件时：\n"
    "   a. 收盘价连续触及或突破布林带上轨 (Upper Band)。\n"
    "   b. RSI(14) 指数高于 70 (严重超买区域)。\n"
    "   c. 最新价格相比前一个周期开始下跌（收盘价低于前一周期收盘价）。\n"

    "3. **观望 (HOLD):** 当价格在布林带内，或RSI在30-70之间，或趋势不明确时。请务必在强力反转信号出现时才给出 BUY/SELL，否则给出 HOLD。"

    "输出必须是有效的 JSON 格式。"
)


def get_mean_reversion_signal(ticker: str = "TSLA", lookback_minutes: int = 60, end_dt: datetime = None, delay_seconds: int = 15) -> dict:
    """
    获取 K 线数据，计算布林带，并让 Gemini 给出区间反转信号。
    """
    # 1. 获取和格式化数据 (Data Fetching and Indicator Calculation)
    kline_data_text = get_latest_bars(
        ticker=ticker, lookback_minutes=lookback_minutes, end_dt=end_dt)

    if "没有找到可用的" in kline_data_text:
        print(f"🔴 错误：未能获取 {ticker} 的有效数据。")
        return {"error": "No data", "signal": "HOLD"}

    print(f"--- 正在使用 Gemini 2.5 Flash 分析 {ticker} 的布林带模式... ---")

    # 2. 构造 LLM 用户输入
    user_prompt = (
        f"请根据以下 {lookback_minutes} 分钟内 {ticker} 的 K 线和布林带数据 (最近 10 条数据)，"
        f"分析是否存在区间反转机会，并给出交易信号。\n\n"
        f"K 线数据表:\n{kline_data_text}"
    )

    # --- 缓存逻辑开始 ---
    # 3. 生成唯一的缓存键 (基于 ticker, timestamp, 和 prompt 的 SHA256 哈希值)
    # 我们将时间戳和 prompt 结合起来
    cache_key_input = f"{ticker}|{end_dt}|{user_prompt}"
    cache_key = hashlib.sha256(cache_key_input.encode('utf-8')).hexdigest()

    # 4. 尝试加载缓存
    cache = load_cache()

    if cache_key in cache:
        print(f"✅ 缓存命中！返回 {end_dt.strftime('%Y-%m-%d %H:%M UTC')} 的缓存结果。")
        return cache[cache_key]

    print(f"--- 缓存未命中。正在调用 Gemini 2.5 Flash 分析 {ticker} 的布林带模式... ---")
    # --- 缓存逻辑结束 ---

    # 5. 调用 Gemini API (如果缓存未命中)
    # 只有当后面还有时间点需要测试时才暂停
    print(f"⏸️ 暂停 {delay_seconds} 秒以遵守 Gemini API 速率限制...")
    time.sleep(delay_seconds)

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[SYSTEM_PROMPT, user_prompt],
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TradingSignal,
                temperature=0.2  # 略微增加温度，允许模型进行更灵活的模式识别
            )
        )

        signal_result = json.loads(response.text)

        # 6. 将结果存入缓存并保存文件
        cache[cache_key] = signal_result
        save_cache(cache)

        return signal_result

    except Exception as e:
        print(f"调用 Gemini API 发生错误: {e}")
        return {"error": str(e), "signal": "HOLD"}

# src/strategies/mean_reversion_strategy.py (替换 backtest_full_day 函数)


def backtest_arbitrary_period(ticker: str, start_dt: datetime, end_dt: datetime, step_minutes: int = 5, delay_seconds: int = 15):
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

# src/strategies/mean_reversion_strategy.py (更新后的运行块)


if __name__ == '__main__':
    # ----------------------------------------------------
    # 设置回测日期和股票
    # ----------------------------------------------------
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
        ticker=TICKER,
        start_dt=START_DATE,
        end_dt=END_DATE,
        step_minutes=STEP_MINUTES  # 传入新的参数
    )
