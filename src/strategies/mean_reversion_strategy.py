# src/strategies/mean_reversion_strategy.py

import json
from google import genai
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
from datetime import datetime, timezone
import hashlib  # 用于生成唯一的缓存键
from src.cache.trading_cache import load_cache, save_cache

# 导入 Alpaca 数据获取函数 (假设已修复 TimeFrame 导入问题)
from src.data.alpaca_data_fetcher import get_latest_bars
from google import genai
import time  # 新增：用于暂停执行

# 初始化 Gemini 客户端
load_dotenv()
client = genai.Client()

GEMINI_MODEL = "gemini-2.5-flash-lite"

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


def get_mean_reversion_signal(cache: dict, ticker: str = "TSLA", lookback_minutes: int = 60, end_dt: datetime = None, delay_seconds: int = 15) -> dict:
    """
    获取 K 线数据，计算布林带，并让 Gemini 给出区间反转信号。
    """
    # 1. 获取和格式化数据 (Data Fetching and Indicator Calculation)
    kline_data_text = get_latest_bars(
        ticker=ticker, lookback_minutes=lookback_minutes, end_dt=end_dt)

    if "" in kline_data_text:
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

    if cache_key in cache:
        print(f"✅ 缓存命中！返回 {end_dt.strftime('%Y-%m-%d %H:%M UTC')} 的缓存结果。")
        return cache[cache_key]

    print(f"--- 缓存未命中。正在调用 Gemini 2.5 Flash 分析 {ticker} 的布林带模式... ---")
    # --- 缓存逻辑结束 ---

    # 4. 调用 Gemini API (如果缓存未命中)
    # 只有当后面还有时间点需要测试时才暂停
    print(f"⏸️ 暂停 {delay_seconds} 秒以遵守 Gemini API 速率限制...")
    time.sleep(delay_seconds)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[SYSTEM_PROMPT, user_prompt],
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TradingSignal,
                temperature=0.2  # 略微增加温度，允许模型进行更灵活的模式识别
            )
        )

        signal_result = json.loads(response.text)

        # 5. 将结果存入缓存并保存文件
        cache[cache_key] = signal_result

        return signal_result

    except Exception as e:
        print(f"调用 Gemini API 发生错误: {e}")
        return {"error": str(e), "signal": "HOLD"}


if __name__ == '__main__':
    TICKER = 'TSLA'
    # 回测结束时间
    END_DATE = datetime(2025, 12, 4, 20, 0, 0,
                        tzinfo=timezone.utc)  # 仅测试 12 月 4 日收盘前

    cache = load_cache()
    get_mean_reversion_signal(
        cache, ticker=TICKER, lookback_minutes=60, end_dt=END_DATE, delay_seconds=15)
    save_cache(cache)
