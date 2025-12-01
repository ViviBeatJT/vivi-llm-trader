# src/strategies/mean_reversion_strategy.py

import json
from google import genai
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
from datetime import datetime, timezone

# 导入 Alpaca 数据获取函数 (假设已修复 TimeFrame 导入问题)
from src.data.alpaca_data_fetcher import get_latest_bars
from google import genai

# 初始化 Gemini 客户端
load_dotenv()
client = genai.Client()

# 定义 LLM 输出结构 (沿用之前的情绪信号)
class TradingSignal(BaseModel):
    """交易信号模型"""
    signal: Literal["BUY", "SELL", "HOLD"] = Field(description="基于技术分析，给出买入、卖出或观望的交易信号。")
    confidence_score: int = Field(..., ge=1, le=10, description="对信号的自信程度评分，10为最高自信。")
    reason: str = Field(description="简要说明给出此信号的原因，必须基于布林带分析。")

# 定义 LLM 系统指令
SYSTEM_PROMPT = (
    "你是一位专业的量化交易员，专注于区间反转（Mean Reversion）策略。你的任务是分析提供的布林带（Bollinger Bands, BB）K线数据表，"
    "并严格按照以下区间反转规则给出交易信号："
    "1. **买入 (BUY):** 当收盘价连续触及或跌破布林带下轨 (Lower Band) 时，且短期有反弹迹象，预期价格将反转回均线 (SMA) 时，给出买入信号。"
    "2. **卖出 (SELL):** 当收盘价连续触及或突破布林带上轨 (Upper Band) 时，预期价格将反转回均线 (SMA) 时，给出卖出信号。"
    "3. **观望 (HOLD):** 当价格在布林带内，或趋势不明确，或无法判断反转信号时。"
    "输出必须是有效的 JSON 格式。"
)

def get_mean_reversion_signal(end_time: datetime = datetime.now(timezone.utc), ticker: str = "TSLA", lookback_minutes: int = 60) -> dict:
    """
    获取 K 线数据，计算布林带，并让 Gemini 给出区间反转信号。
    """
    # 1. 获取和格式化数据 (Data Fetching and Indicator Calculation)
    kline_data_text = get_latest_bars(ticker=ticker, end_time=end_time, lookback_minutes=lookback_minutes)
    
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

    # 3. 调用 Gemini API
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[SYSTEM_PROMPT, user_prompt],
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TradingSignal,
                temperature=0.2 # 略微增加温度，允许模型进行更灵活的模式识别
            )
        )
        
        return json.loads(response.text)

    except Exception as e:
        print(f"调用 Gemini API 发生错误: {e}")
        return {"error": str(e), "signal": "HOLD"}


if __name__ == '__main__':
    # 运行测试
    end_time = datetime(2025, 11, 28, 16, 0, 0, tzinfo=timezone.utc)    
    signal = get_mean_reversion_signal(end_time, ticker="TSLA")
    print("\n--- 策略分析结果 ---")
    print(json.dumps(signal, indent=4, ensure_ascii=False))