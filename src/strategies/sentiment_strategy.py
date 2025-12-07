# src/strategies/sentiment_strategy.py (使用 Gemini 2.5 Flash 重构)

import os
import json
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from typing import Literal

# 1. 定义结构化输出的 Pydantic 模型
# Pydantic 模型是 GenAI 强制结构化输出的最佳方式


class TradingSignal(BaseModel):
    """交易信号模型"""
    signal: Literal["BUY", "SELL", "HOLD"] = Field(
        description="基于新闻情绪，给出买入、卖出或观望的交易信号。"
    )
    confidence_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="对信号的自信程度评分，10为最高自信。"
    )
    reason: str = Field(
        description="简要说明给出此信号的原因，必须基于新闻内容。"
    )


# 2. 初始化 Gemini 客户端
load_dotenv()
# 客户端将自动从 GEMINI_API_KEY 环境变量中读取密钥
try:
    client = genai.Client()
except Exception as e:
    print(f"❌ 初始化 Gemini 客户端失败：{e}")
    print("请检查 .env 文件中的 GEMINI_API_KEY 是否设置正确。")
    client = None

# 3. 定义 LLM 系统指令 (System Prompt)
SYSTEM_PROMPT = (
    "你是一位专业的、谨慎的金融市场情绪分析师。你的任务是分析提供的财经新闻摘要，"
    "判断其对股票的短期影响，并严格按照提供的 JSON 格式输出交易信号。"
    "输出必须是有效的 JSON，且信号必须是 BUY, SELL, 或 HOLD 之一。请侧重于短期影响。"
)


def get_trading_signal_from_text(news_text: str, ticker: str = "TSLA") -> dict:
    """
    使用 Gemini API 分析新闻文本，并返回结构化的交易信号。
    """
    if not client:
        return {"error": "Client not initialized", "signal": "HOLD"}

    print(f"--- 正在使用 Gemini 2.5 Flash 分析 {ticker} 的新闻情绪... ---")

    # 构造用户输入
    user_prompt = f"分析以下关于 {ticker} 的新闻，并给出交易信号：\n\n新闻内容：{news_text}"

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[SYSTEM_PROMPT, user_prompt],
            config=genai.types.GenerateContentConfig(
                # 强制要求 JSON 输出，并使用 Pydantic schema
                response_mime_type="application/json",
                response_schema=TradingSignal,
                temperature=0.1  # 调低温度，输出更稳定
            )
        )

        # response.text 是一个符合 Pydantic 模型的 JSON 字符串
        # 我们将其解析为 Python 字典方便后续处理
        return json.loads(response.text)

    except Exception as e:
        print(f"调用 Gemini API 发生错误: {e}")
        return {"error": str(e), "signal": "HOLD"}


# 4. 测试用例 (与之前相同，验证输出格式)
if __name__ == '__main__':
    # 模拟一条利好新闻
    bullish_news = (
        "Tesla's Q3 delivery numbers unexpectedly exceeded analyst expectations, "
        "hitting a new all-time high driven by strong demand for Model Y. "
        "The company is also announcing a new, lower-cost battery technology next week."
    )

    print("\n[测试案例 1: 模拟利好消息]")
    signal_bullish = get_trading_signal_from_text(bullish_news, ticker="TSLA")
    print(json.dumps(signal_bullish, indent=4, ensure_ascii=False))

    # 模拟一条利空新闻
    bearish_news = (
        "The NHTSA announced a formal investigation into Tesla's Autopilot system "
        "following several recent crashes. Additionally, a major bank downgraded "
        "TSLA stock from Buy to Neutral due to valuation concerns."
    )

    print("\n[测试案例 2: 模拟利空消息]")
    signal_bearish = get_trading_signal_from_text(bearish_news, ticker="TSLA")
    print(json.dumps(signal_bearish, indent=4, ensure_ascii=False))
