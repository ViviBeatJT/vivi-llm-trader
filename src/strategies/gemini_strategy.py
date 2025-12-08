# src/strategies/gemini_strategy.py

import os
import json
import hashlib
import time
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from typing import Literal, Tuple, Dict, Optional
from datetime import datetime, timezone
import pandas as pd
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# 导入数据获取器、缓存和基类
from src.data_fetcher.alpaca_data_fetcher import AlpacaDataFetcher
from src.cache.trading_cache import TradingCache
from src.strategies.base_strategy import BaseStrategy # 导入基类

# 加载环境变量
load_dotenv()

# Gemini 模型配置
GEMINI_MODEL = "gemini-2.0-flash-exp"  # 使用最新的 Flash 模型


class TradingSignal(BaseModel):
    """交易信号模型 - 用于强制 Gemini 输出结构化数据"""
    signal: Literal["BUY", "SELL", "HOLD"] = Field(
        description="基于技术分析和市场数据，给出买入、卖出或观望的交易信号。"
    )
    confidence_score: int = Field(
        ..., 
        ge=1, 
        le=10,
        description="对信号的自信程度评分，1为最低，10为最高。"
    )
    reason: str = Field(
        description="简要说明给出此信号的原因，必须基于提供的技术指标和价格数据。"
    )


class GeminiStrategy(BaseStrategy): # 继承 BaseStrategy
    """
    基于 Gemini AI 的交易策略。
    
    特点：
    1. 使用 Gemini API 分析技术指标和价格走势
    2. 支持缓存以减少 API 调用和成本
    3. 可自定义系统提示词（trading persona）
    4. 灵活的参数配置
    """
    
    # 默认系统提示词
    DEFAULT_SYSTEM_PROMPT = """你是一位经验丰富的量化交易员，专注于短期交易和技术分析。

你的任务是分析提供的股票技术指标数据（包括价格、布林带、RSI等），并给出明确的交易建议。

分析要点：
1. **趋势判断**：观察价格相对于移动平均线(SMA)的位置
2. **布林带分析**：
   - 价格触及下轨可能是买入机会（超卖）
   - 价格触及上轨可能是卖出机会（超买）
3. **RSI指标**：
   - RSI < 30 表示超卖，可能反弹
   - RSI > 70 表示超买，可能回调
   - RSI 在 30-70 之间为中性区域
4. **成交量**：异常放量可能预示趋势变化

交易原则：
- 保守谨慎，不确定时选择 HOLD
- 信号强度（confidence_score）要如实反映分析的确定性
- 必须基于提供的数据，不要臆测

请严格按照 JSON 格式输出，包含 signal, confidence_score, reason 三个字段。"""

    def __init__(self, 
                 data_fetcher: AlpacaDataFetcher,
                 cache: Optional[TradingCache] = None,
                 use_cache: bool = True,
                 system_prompt: Optional[str] = None,
                 model: str = GEMINI_MODEL,
                 temperature: float = 0.2,
                 delay_seconds: int = 2):
        """
        初始化 Gemini 交易策略。
        
        Args:
            data_fetcher: AlpacaDataFetcher 实例
            cache: TradingCache 实例（可选）
            use_cache: 是否使用缓存
            system_prompt: 自定义系统提示词
            model: Gemini 模型名称
            temperature: 生成温度（0-1，越低越确定）
            delay_seconds: API 调用间隔（避免速率限制）
        """
        super().__init__(data_fetcher) # 调用基类构造函数
        self.cache = cache
        self.use_cache = use_cache and cache is not None
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.model = model
        self.temperature = temperature
        self.delay_seconds = delay_seconds
        
        # 初始化 Gemini 客户端
        try:
            self.client = genai.Client()
            print(f"✅ GeminiStrategy 初始化完成。")
            print(f"   模型: {model}, 温度: {temperature}, 缓存: {'启用' if self.use_cache else '禁用'}")
        except Exception as e:
            print(f"❌ 初始化 Gemini 客户端失败：{e}")
            print("   请检查 .env 文件中的 GEMINI_API_KEY 是否设置正确。")
            self.client = None
    
    def _format_data_for_llm(self, df: pd.DataFrame, ticker: str) -> str:
        """
        将技术指标数据格式化为 LLM 友好的文本。
        
        Args:
            df: 包含技术指标的 DataFrame
            ticker: 股票代码
            
        Returns:
            str: 格式化的 Markdown 表格文本
        """
        if df.empty:
            return "没有可用的市场数据。"
        
        # 选择最近 10 个数据点
        df_display = df.tail(10).copy()
        
        # 格式化时间索引
        if hasattr(df_display.index, 'strftime'):
            df_display.index = df_display.index.strftime('%H:%M')
        
        # 选择需要显示的列
        cols_to_show = []
        for col in ['close', 'volume', 'SMA', 'BB_UPPER', 'BB_LOWER', 'RSI']:
            if col in df_display.columns:
                cols_to_show.append(col)
        
        df_display = df_display[cols_to_show]
        
        # 重命名列为更友好的名称
        col_mapping = {
            'close': 'Close',
            'volume': 'Volume',
            'SMA': 'SMA_20',
            'BB_UPPER': 'BB_Upper',
            'BB_LOWER': 'BB_Lower',
            'RSI': 'RSI_14'
        }
        df_display.rename(columns=col_mapping, inplace=True)
        
        # 格式化数值
        for col in df_display.columns:
            if col != 'Volume':
                df_display[col] = df_display[col].round(2)
        
        # 转换为 Markdown
        markdown_table = df_display.to_markdown()
        
        return f"### {ticker} 技术指标数据（最近10个时间点）\n\n{markdown_table}"
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标（布林带和 RSI）。
        
        Args:
            df: 原始 OHLCV DataFrame
            
        Returns:
            pd.DataFrame: 添加了技术指标的 DataFrame
        """
        df = df.copy()
        
        # 布林带 (20 period, 2 std dev)
        df['SMA'] = df['close'].rolling(window=20).mean()
        df['STD'] = df['close'].rolling(window=20).std()
        df['BB_UPPER'] = df['SMA'] + (df['STD'] * 2)
        df['BB_LOWER'] = df['SMA'] - (df['STD'] * 2)
        
        # RSI (14 period)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        RS = gain / loss
        df['RSI'] = 100 - (100 / (1 + RS))
        
        # 删除计算产生的中间列
        df.drop(['STD'], axis=1, inplace=True, errors='ignore')
        
        return df
    
    def _generate_cache_key(self, ticker: str, timestamp: datetime, formatted_data: str) -> str:
        """
        生成缓存键。
        
        Args:
            ticker: 股票代码
            timestamp: 时间戳
            formatted_data: 格式化后的数据文本
            
        Returns:
            str: SHA256 哈希值作为缓存键
        """
        # 组合所有输入来生成唯一键
        key_input = f"{ticker}|{timestamp.isoformat()}|{formatted_data}"
        return hashlib.sha256(key_input.encode('utf-8')).hexdigest()
    
    def _call_gemini_api(self, user_prompt: str) -> Dict:
        """
        调用 Gemini API 获取交易信号。
        
        Args:
            user_prompt: 用户提示词（包含格式化的市场数据）
            
        Returns:
            Dict: 包含 signal, confidence_score, reason 的字典
        """
        if not self.client:
            return {
                "signal": "HOLD",
                "confidence_score": 0,
                "reason": "Gemini client not initialized"
            }
        
        print(f"🤖 正在调用 Gemini API ({self.model})...")
        
        # 等待以避免速率限制
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    {"role": "user", "parts": [{"text": self.system_prompt}]},
                    {"role": "user", "parts": [{"text": user_prompt}]}
                ],
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TradingSignal,
                    temperature=self.temperature
                )
            )
            
            if not response.text:
                raise Exception("Gemini API 返回了空响应。")
            
            result = json.loads(response.text)
            print(f"✅ Gemini 分析完成。信号: {result['signal']}, 置信度: {result['confidence_score']}/10")
            
            return result
            
        except Exception as e:
            print(f"❌ 调用 Gemini API 失败: {e}")
            return {
                "signal": "HOLD",
                "confidence_score": 0,
                "reason": f"API Error: {str(e)}"
            }
    
    def get_signal(self,
                   ticker: str,
                   end_dt: Optional[datetime] = None,
                   lookback_minutes: int = 120,
                   timeframe: TimeFrame = TimeFrame(5, TimeFrameUnit.Minute)) -> Tuple[Dict, float]:
        """
        获取指定时间点的 AI 交易信号。
        
        Args:
            ticker: 股票代码
            end_dt: 结束时间（默认为当前时间）
            lookback_minutes: 回溯时间长度（分钟）
            timeframe: K线时间框架
            
        Returns:
            Tuple[signal_dict, current_price]:
                - signal_dict: 包含 signal, confidence_score, reason 的字典
                - current_price: 当前价格
        """
        # 1. 获取原始数据
        df = self.data_fetcher.get_latest_bars(
            ticker=ticker,
            lookback_minutes=lookback_minutes,
            timeframe=timeframe,
            end_dt=end_dt
        )
        
        if df.empty:
            print(f"❌ 无法获取 {ticker} 的数据。")
            return {"signal": "HOLD", "confidence_score": 0, "reason": "No data"}, 0.0
        
        # 2. 计算技术指标
        df = self._calculate_technical_indicators(df)
        df = df.dropna()
        
        if df.empty:
            print(f"❌ 计算技术指标后数据不足。")
            return {"signal": "HOLD", "confidence_score": 0, "reason": "Insufficient data"}, 0.0
        
        # 3. 获取当前价格
        current_price = df['close'].iloc[-1]
        
        # 4. 格式化数据给 LLM
        formatted_data = self._format_data_for_llm(df, ticker)
        
        # 5. 检查缓存
        timestamp_for_display = end_dt if end_dt else datetime.now(timezone.utc)
        
        if self.use_cache:
            cache_key = self._generate_cache_key(ticker, timestamp_for_display, formatted_data)
            cached_result = self.cache.get(cache_key)
            
            if cached_result:
                print(f"✅ 缓存命中！返回缓存的 Gemini 分析结果。")
                return cached_result, current_price
        
        # 6. 构造用户提示词
        user_prompt = f"""请分析以下 {ticker} 的市场数据并给出交易建议。

当前时间: {timestamp_for_display.strftime('%Y-%m-%d %H:%M UTC')}

{formatted_data}

请基于以上技术指标，给出你的交易建议。"""
        
        # 7. 调用 Gemini API
        signal_result = self._call_gemini_api(user_prompt)
        
        # 8. 保存到缓存
        if self.use_cache and (signal_result.get('signal') != 'HOLD' or signal_result.get('confidence_score', 0) > 0):
            # 只有当 signal_result 包含有效信号时才保存
            self.cache.add(cache_key, signal_result)
        
        # 9. 打印信号信息
        print(f"\n🎯 [{timestamp_for_display.strftime('%Y-%m-%d %H:%M UTC')}] {ticker} Gemini 分析:")
        print(f"   价格: ${current_price:.2f}")
        print(f"   信号: {signal_result.get('signal', 'N/A')} (置信度: {signal_result.get('confidence_score', 0)}/10)")
        print(f"   原因: {signal_result.get('reason', 'N/A')}")
        
        return signal_result, current_price


# 测试用例
if __name__ == '__main__':
    from datetime import datetime, timezone
    
    # 需要假设 AlpacaDataFetcher 和 TradingCache 存在
    class MockDataFetcher:
        def get_latest_bars(self, ticker, lookback_minutes, timeframe, end_dt):
            print(f"Mocking data fetch for {ticker}...")
            # 构造模拟数据，确保有足够的行进行指标计算
            data = {
                'open': [100, 101, 99, 98, 97, 96, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                'high': [101, 102, 100, 99, 98, 97, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                'low': [99, 100, 98, 97, 96, 95, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                'close': [100.5, 101.5, 99.5, 98.5, 97.5, 96.5, 95.5, 96.5, 97.5, 98.5, 99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5],
                'volume': [1000] * 22
            }
            # 创建一个时间索引
            index = pd.to_datetime(pd.date_range(end=datetime.now(timezone.utc), periods=len(data['close']), freq='5min'), utc=True)
            return pd.DataFrame(data, index=index)
        
    class MockTradingCache:
        def __init__(self, filename):
            self.data = {}
            self.filename = filename
        def get(self, key):
            return self.data.get(key)
        def add(self, key, value):
            self.data[key] = value
        def save(self):
            print(f"Saving mock cache to {self.filename}")
            
    fetcher = MockDataFetcher()
    cache = MockTradingCache('gemini_test_cache.json')
    
    # 模拟 genai.Client 以避免真正的 API 调用
    class MockGenaiClient:
        def __init__(self):
            class MockModels:
                def generate_content(self, model, contents, config):
                    class MockResponse:
                        def __init__(self, text):
                            self.text = text
                    
                    # 检查 prompt 决定返回 BUY 或 SELL
                    if "跌破布林带" in contents[1]['parts'][0]['text']:
                        signal_text = '{"signal": "BUY", "confidence_score": 8, "reason": "价格已触及布林带下轨，且RSI处于超卖区域，预计短期内将反弹。"}'
                    elif "突破布林带" in contents[1]['parts'][0]['text']:
                        signal_text = '{"signal": "SELL", "confidence_score": 7, "reason": "价格已突破布林带上轨，出现超买信号，建议获利了结。"}'
                    else:
                        signal_text = '{"signal": "HOLD", "confidence_score": 5, "reason": "价格在均线附近盘整，缺乏明确方向。"}'
                        
                    return MockResponse(signal_text)
            self.models = MockModels()

    # 替换实际的 genai.Client
    strategy = GeminiStrategy(
        data_fetcher=fetcher,
        cache=cache,
        use_cache=True,
        temperature=0.2,
        delay_seconds=0 # 移除延迟
    )
    strategy.client = MockGenaiClient() # 使用 Mock Client
    
    # 测试获取信号
    print("\n" + "="*60)
    print("测试 GeminiStrategy - AI 驱动的交易决策 (使用 Mock)")
    print("="*60)
    
    signal_dict, price = strategy.get_signal(
        ticker="TSLA",
        lookback_minutes=120,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute)
    )
    
    print(f"\n最终输出:")
    print(f"  信号: {signal_dict.get('signal')}")
    print(f"  置信度: {signal_dict.get('confidence_score')}/10")
    print(f"  原因: {signal_dict.get('reason')}")
    print(f"  当前价格: ${price:.2f}")
    
    # 保存缓存
    if len(cache.data) > 0:
        cache.save()