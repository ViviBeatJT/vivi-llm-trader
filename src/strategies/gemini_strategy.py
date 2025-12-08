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

# 导入缓存和基类
from src.cache.trading_cache import TradingCache
from src.strategies.base_strategy import BaseStrategy

# 加载环境变量
load_dotenv()

# Gemini 模型配置
GEMINI_MODEL = "gemini-2.0-flash-exp"


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


class GeminiStrategy(BaseStrategy):
    """
    基于 Gemini AI 的交易策略。
    
    特点：
    1. 不依赖 data_fetcher，数据通过参数传入
    2. 维护历史数据，合并后计算技术指标再交给 AI 分析
    3. 支持缓存以减少 API 调用和成本
    4. 可自定义系统提示词（trading persona）
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

    # 技术指标参数
    DEFAULT_BB_PERIOD = 20
    DEFAULT_RSI_WINDOW = 14
    DEFAULT_MAX_HISTORY_BARS = 500

    def __init__(self, 
                 cache: Optional[TradingCache] = None,
                 use_cache: bool = True,
                 system_prompt: Optional[str] = None,
                 model: str = GEMINI_MODEL,
                 temperature: float = 0.2,
                 delay_seconds: int = 2,
                 bb_period: int = DEFAULT_BB_PERIOD,
                 rsi_window: int = DEFAULT_RSI_WINDOW,
                 max_history_bars: int = DEFAULT_MAX_HISTORY_BARS):
        """
        初始化 Gemini 交易策略。
        
        Args:
            cache: TradingCache 实例（可选）
            use_cache: 是否使用缓存
            system_prompt: 自定义系统提示词
            model: Gemini 模型名称
            temperature: 生成温度（0-1，越低越确定）
            delay_seconds: API 调用间隔（避免速率限制）
            bb_period: 布林带计算周期
            rsi_window: RSI 计算窗口
            max_history_bars: 最大保留的历史K线数量
        """
        self.cache = cache
        self.use_cache = use_cache and cache is not None
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.model = model
        self.temperature = temperature
        self.delay_seconds = delay_seconds
        self.bb_period = bb_period
        self.rsi_window = rsi_window
        self.max_history_bars = max_history_bars
        
        # 历史数据存储：按 ticker 分别存储
        self._history_data: Dict[str, pd.DataFrame] = {}
        
        # 初始化 Gemini 客户端
        try:
            self.client = genai.Client()
            print(f"✅ GeminiStrategy 初始化完成")
            print(f"   模型: {model}, 温度: {temperature}, 缓存: {'启用' if self.use_cache else '禁用'}")
        except Exception as e:
            print(f"❌ 初始化 Gemini 客户端失败：{e}")
            print("   请检查 .env 文件中的 GEMINI_API_KEY 是否设置正确。")
            self.client = None
    
    # ==================== 历史数据管理 ====================
    
    def _merge_data(self, ticker: str, new_df: pd.DataFrame) -> pd.DataFrame:
        """将新数据与历史数据合并。"""
        if new_df.empty:
            return self._history_data.get(ticker, pd.DataFrame())
        
        if ticker not in self._history_data or self._history_data[ticker].empty:
            merged_df = new_df.copy()
        else:
            history_df = self._history_data[ticker]
            merged_df = pd.concat([history_df, new_df])
            merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
            merged_df = merged_df.sort_index()
        
        if len(merged_df) > self.max_history_bars:
            merged_df = merged_df.iloc[-self.max_history_bars:]
        
        self._history_data[ticker] = merged_df
        return merged_df
    
    def get_history_data(self, ticker: str) -> pd.DataFrame:
        """获取指定 ticker 的历史数据副本。"""
        if ticker in self._history_data:
            return self._history_data[ticker].copy()
        return pd.DataFrame()
    
    def clear_history(self, ticker: Optional[str] = None):
        """清除历史数据。如果 ticker 为 None，清除所有。"""
        if ticker is None:
            self._history_data.clear()
            print("🗑️ 已清除所有历史数据。")
        elif ticker in self._history_data:
            del self._history_data[ticker]
            print(f"🗑️ 已清除 {ticker} 的历史数据。")
    
    def get_history_size(self, ticker: str) -> int:
        """获取指定 ticker 的历史数据条数。"""
        return len(self._history_data.get(ticker, []))
    
    # ==================== 技术指标计算 ====================
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标（布林带和 RSI）。"""
        df = df.copy()
        
        # 布林带
        df['SMA'] = df['close'].rolling(window=self.bb_period).mean()
        df['STD'] = df['close'].rolling(window=self.bb_period).std()
        df['BB_UPPER'] = df['SMA'] + (df['STD'] * 2)
        df['BB_LOWER'] = df['SMA'] - (df['STD'] * 2)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        RS = gain / loss
        df['RSI'] = 100 - (100 / (1 + RS))
        
        # 删除中间列
        df.drop(['STD'], axis=1, inplace=True, errors='ignore')
        
        return df
    
    # ==================== LLM 交互 ====================
    
    def _format_data_for_llm(self, df: pd.DataFrame, ticker: str) -> str:
        """将技术指标数据格式化为 LLM 友好的文本。"""
        if df.empty:
            return "没有可用的市场数据。"
        
        df_display = df.tail(10).copy()
        
        if hasattr(df_display.index, 'strftime'):
            df_display.index = df_display.index.strftime('%H:%M')
        
        cols_to_show = []
        for col in ['close', 'volume', 'SMA', 'BB_UPPER', 'BB_LOWER', 'RSI']:
            if col in df_display.columns:
                cols_to_show.append(col)
        
        df_display = df_display[cols_to_show]
        
        col_mapping = {
            'close': 'Close',
            'volume': 'Volume',
            'SMA': 'SMA_20',
            'BB_UPPER': 'BB_Upper',
            'BB_LOWER': 'BB_Lower',
            'RSI': 'RSI_14'
        }
        df_display.rename(columns=col_mapping, inplace=True)
        
        for col in df_display.columns:
            if col != 'Volume':
                df_display[col] = df_display[col].round(2)
        
        markdown_table = df_display.to_markdown()
        
        return f"### {ticker} 技术指标数据（最近10个时间点）\n\n{markdown_table}"
    
    def _generate_cache_key(self, ticker: str, timestamp: datetime, formatted_data: str) -> str:
        """生成缓存键。"""
        key_input = f"{ticker}|{timestamp.isoformat()}|{formatted_data}"
        return hashlib.sha256(key_input.encode('utf-8')).hexdigest()
    
    def _call_gemini_api(self, user_prompt: str) -> Dict:
        """调用 Gemini API 获取交易信号。"""
        if not self.client:
            return {
                "signal": "HOLD",
                "confidence_score": 0,
                "reason": "Gemini client not initialized"
            }
        
        print(f"🤖 正在调用 Gemini API ({self.model})...")
        
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
    
    # ==================== 主接口 ====================
    
    def get_signal(self,
                   ticker: str,
                   new_data: pd.DataFrame,
                   verbose: bool = True) -> Tuple[Dict, float]:
        """
        分析数据并获取 AI 交易信号。
        
        数据会与历史数据合并后再计算指标，然后交给 Gemini 分析。
        
        Args:
            ticker: 股票代码
            new_data: 新的 OHLCV DataFrame，索引为时间戳
            verbose: 是否打印详细信息
            
        Returns:
            Tuple[signal_dict, current_price]:
                - signal_dict: {'signal': str, 'confidence_score': int, 'reason': str}
                - current_price: 最新价格
        """
        # 1. 合并历史数据
        df = self._merge_data(ticker, new_data)
        
        if verbose:
            print(f"📊 {ticker} 数据: {len(df)} 条K线 (新增: {len(new_data)})")
        
        if df.empty:
            return {"signal": "HOLD", "confidence_score": 0, "reason": "No data"}, 0.0
        
        # 2. 计算技术指标
        df = self._calculate_technical_indicators(df)
        df_valid = df.dropna()
        
        min_required = max(self.bb_period, self.rsi_window)
        if df_valid.empty:
            if verbose:
                print(f"❌ 数据不足，需要至少 {min_required} 条有效数据")
            return {"signal": "HOLD", "confidence_score": 0, 
                    "reason": f"Insufficient data (need {min_required})"}, 0.0
        
        # 3. 获取当前价格
        current_price = df_valid['close'].iloc[-1]
        
        # 4. 格式化数据给 LLM
        formatted_data = self._format_data_for_llm(df_valid, ticker)
        
        # 5. 获取时间戳用于缓存和显示
        if hasattr(df_valid.index[-1], 'strftime'):
            timestamp_for_cache = df_valid.index[-1]
            timestamp_str = timestamp_for_cache.strftime('%Y-%m-%d %H:%M UTC')
        else:
            timestamp_for_cache = datetime.now(timezone.utc)
            timestamp_str = str(df_valid.index[-1])
        
        # 6. 检查缓存
        if self.use_cache:
            cache_key = self._generate_cache_key(ticker, timestamp_for_cache, formatted_data)
            cached_result = self.cache.get(cache_key)
            
            if cached_result:
                if verbose:
                    print(f"✅ 缓存命中！返回缓存的 Gemini 分析结果。")
                return cached_result, current_price
        
        # 7. 构造用户提示词
        user_prompt = f"""请分析以下 {ticker} 的市场数据并给出交易建议。

当前时间: {timestamp_str}

{formatted_data}

请基于以上技术指标，给出你的交易建议。"""
        
        # 8. 调用 Gemini API
        signal_result = self._call_gemini_api(user_prompt)
        
        # 9. 保存到缓存
        if self.use_cache and signal_result.get('confidence_score', 0) > 0:
            self.cache.add(cache_key, signal_result)
        
        # 10. 打印信号信息
        if verbose:
            print(f"\n🎯 [{timestamp_str}] {ticker} Gemini 分析:")
            print(f"   价格: ${current_price:.2f}")
            print(f"   信号: {signal_result.get('signal', 'N/A')} (置信度: {signal_result.get('confidence_score', 0)}/10)")
            print(f"   原因: {signal_result.get('reason', 'N/A')}")
        
        return signal_result, current_price
    
    def __str__(self):
        return f"GeminiStrategy(model={self.model}, cache={'on' if self.use_cache else 'off'})"


# ==================== 测试用例 ====================
if __name__ == '__main__':
    import numpy as np
    from datetime import timedelta
    
    def create_test_data(num_bars: int, base_price: float, start_time: datetime) -> pd.DataFrame:
        """创建测试用 OHLCV 数据"""
        np.random.seed(42)
        prices = base_price + np.cumsum(np.random.randn(num_bars) * 0.5)
        index = pd.DatetimeIndex([start_time + timedelta(minutes=i*5) for i in range(num_bars)])
        return pd.DataFrame({
            'open': prices - 0.1,
            'high': prices + 0.3,
            'low': prices - 0.3,
            'close': prices,
            'volume': np.random.randint(1000, 5000, num_bars)
        }, index=index)
    
    # Mock cache
    class MockCache:
        def __init__(self):
            self.data = {}
        def get(self, key):
            return self.data.get(key)
        def add(self, key, value):
            self.data[key] = value
    
    # Mock Gemini client
    class MockGenaiClient:
        class MockModels:
            def generate_content(self, model, contents, config):
                class MockResponse:
                    text = '{"signal": "HOLD", "confidence_score": 6, "reason": "价格在均线附近，缺乏明确方向。"}'
                return MockResponse()
        models = MockModels()
    
    print("="*60)
    print("测试 GeminiStrategy (无 data_fetcher 依赖)")
    print("="*60)
    
    # 初始化
    cache = MockCache()
    strategy = GeminiStrategy(
        cache=cache,
        use_cache=True,
        temperature=0.2,
        delay_seconds=0
    )
    strategy.client = MockGenaiClient()  # 使用 Mock
    
    # 测试
    base_time = datetime(2025, 12, 5, 9, 0, 0, tzinfo=timezone.utc)
    
    print("\n--- 第1批数据 (15条，不足) ---")
    data_1 = create_test_data(15, 100.0, base_time)
    signal, price = strategy.get_signal("TSLA", data_1)
    print(f"历史累积: {strategy.get_history_size('TSLA')} 条")
    
    print("\n--- 第2批数据 (15条，累积后足够) ---")
    data_2 = create_test_data(15, 102.0, base_time + timedelta(minutes=75))
    signal, price = strategy.get_signal("TSLA", data_2)
    print(f"历史累积: {strategy.get_history_size('TSLA')} 条")
    
    print(f"\n最终信号: {signal['signal']}, 置信度: {signal['confidence_score']}/10")
    print(f"当前价格: ${price:.2f}")