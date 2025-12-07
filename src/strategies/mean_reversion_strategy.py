# src/strategies/mean_reversion_strategy.py

import json
from google import genai
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
from datetime import datetime, timezone
import hashlib  # 用于生成唯一的缓存键
# 导入 TradingCache 类
from src.cache.trading_cache import TradingCache 
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# 导入 Alpaca 数据获取函数
from src.data.alpaca_data_fetcher import get_latest_bars
from google import genai
import time

# 初始化 Gemini 客户端
load_dotenv()
client = genai.Client()

GEMINI_MODEL = "gemini-2.5-flash-lite"

# 定义 LLM 输出结构 (不变)


class TradingSignal(BaseModel):
    """交易信号模型"""
    signal: Literal["BUY", "SELL", "HOLD"] = Field(
        description="基于技术分析，给出买入、卖出或观望的交易信号。")
    confidence_score: int = Field(..., ge=1, le=10,
                                  description="对信号的自信程度评分，10为最高自信。")
    reason: str = Field(description="简要说明给出此信号的原因，必须基于布林带分析。")


# 定义 LLM 系统指令 (不变)
SYSTEM_PROMPT = (
    "你是一位专业的量化交易员，专注于区间反转（Mean Reversion）策略。你的任务是分析提供的包含布林带（BB）和 RSI 指数（RSI(14)）的 K线数据表，"
    "并严格按照以下区间反转规则给出交易信号：1. BUY (买入)：当价格跌破布林带下轨，同时 RSI 低于 30 (超卖区) 时。2. SELL (卖出/平仓)：当价格触及布林带上轨，或从超卖区反弹至均线附近时。3. HOLD (观望)：其他情况。你的回复必须是有效的 JSON 格式。"
)


def get_mean_reversion_signal(cache: TradingCache, # 更改参数类型为 TradingCache
                              ticker: str, 
                              end_dt: datetime, 
                              lookback_minutes: int, 
                              timeframe: TimeFrame = TimeFrame(5, TimeFrameUnit.Minute), 
                              delay_seconds: int = 15):
    """
    获取指定时间点的均值回归策略信号。

    Returns:
        tuple[dict, float]: 交易信号字典和最新的收盘价格。
    """

    # 1. 获取 K 线数据 (包含 BB 和 RSI)
    # get_latest_bars 现在返回 (格式化文本, DataFrame)
    kline_data_text, df_bars = get_latest_bars(
        ticker=ticker, lookback_minutes=lookback_minutes, timeframe=timeframe, end_dt=end_dt)

    if df_bars.empty or kline_data_text == "没有找到可用的 K 线数据。":
        print(
            f"❌ 错误：在 {end_dt.strftime('%Y-%m-%d %H:%M UTC')} 找不到 {ticker} 的 K 线数据。")
        # 返回 HOLD 信号和 0 价格
        return {"signal": "HOLD", "confidence_score": 0, "reason": "No data"}, 0.0

    # 获取最新的收盘价作为执行价格
    current_price = df_bars['close'].iloc[-1]

    user_prompt = f"请分析 {ticker} 截止到 {end_dt.strftime('%Y-%m-%d %H:%M UTC')} 的 K线数据:\n\n{kline_data_text}"

    # 2. 缓存检查
    cache_key_input = f"{ticker}|{end_dt}|{user_prompt}"
    cache_key = hashlib.sha256(cache_key_input.encode('utf-8')).hexdigest()

    # 使用 cache.get() 检查缓存
    signal_result = cache.get(cache_key)
    
    if signal_result:
        print(f"✅ 缓存命中！返回 {end_dt.strftime('%Y-%m-%d %H:%M UTC')} 的缓存结果。")
        # 返回缓存结果和当前价格
        return signal_result, current_price

    print(f"--- 缓存未命中。正在调用 Gemini 2.5 Flash 分析 {ticker} 的布林带模式... ---")
    # --- 缓存逻辑结束 ---

    # 4. 调用 Gemini API (如果缓存未命中)
    print(f"⏸️ 暂停 {delay_seconds} 秒以遵守 Gemini API 速率限制...")
    time.sleep(delay_seconds)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[SYSTEM_PROMPT, user_prompt],
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TradingSignal,
                temperature=0.2
            )
        )

        signal_result = json.loads(response.text)

        # 5. 将结果存入缓存并保存文件
        # 使用 cache.add() 添加缓存
        cache.add(cache_key, signal_result) 
        # Note: 缓存保存现在由 backtest_runner 在运行结束后统一调用。
        # 如果需要立即保存，可以在这里调用 cache.save()
        # 但为避免频繁 I/O，我们依赖统一的保存机制。

        # 返回信号结果和当前价格
        return signal_result, current_price

    except Exception as e:
        print(f"调用 Gemini API 发生错误: {e}")
        # 返回错误信号和当前价格
        return {"error": str(e), "signal": "HOLD"}, current_price