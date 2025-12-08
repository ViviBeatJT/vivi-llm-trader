# src/strategies/base_strategy.py

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
from datetime import datetime
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

class BaseStrategy(ABC):
    """
    所有交易策略的抽象基类。
    
    定义了策略必须实现的核心接口：get_signal。
    这确保了所有策略（无论是技术分析、AI驱动还是情绪分析）
    都可以被统一的方式调用。
    """

    def __init__(self, data_fetcher):
        """
        初始化策略。每个策略都需要一个数据获取器。
        
        Args:
            data_fetcher: 数据获取器实例（例如 AlpacaDataFetcher）。
        """
        self.data_fetcher = data_fetcher
        print(f"📊 {self.__class__.__name__} initialized.")

    @abstractmethod
    def get_signal(self,
                   ticker: str,
                   end_dt: Optional[datetime] = None,
                   lookback_minutes: int = 60,
                   timeframe: TimeFrame = TimeFrame(5, TimeFrameUnit.Minute)) -> Tuple[Dict, float]:
        """
        获取指定时间点的交易信号。
        
        所有子类必须实现此方法。
        
        Args:
            ticker: 股票代码。
            end_dt: 结束时间（默认为当前时间）。
            lookback_minutes: K线数据回溯时间长度（分钟）。
            timeframe: K线时间框架。
            
        Returns:
            Tuple[signal_dict, current_price]:
                - signal_dict: 包含 signal, confidence_score, reason 的字典。
                - current_price: 当前价格。
        """
        pass

    def __str__(self):
        """返回策略名称。"""
        return self.__class__.__name__

# 定义统一的信号输出结构（可以在需要的地方导入）
SIGNAL_OUTPUT_EXAMPLE = {
    "signal": "BUY" or "SELL" or "HOLD",
    "confidence_score": 10, # 1-10
    "reason": "..."
}