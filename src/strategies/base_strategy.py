# src/strategies/base_strategy.py

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import pandas as pd


class BaseStrategy(ABC):
    """
    所有交易策略的抽象基类。
    
    设计原则：
    - 策略只负责分析数据并生成信号
    - 数据获取由外部（如 BacktestEngine）负责
    - 策略是一个纯粹的"数据 → 信号"转换器
    
    信号说明：
    - BUY: 买入开多仓（做多）
    - SELL: 卖出平多仓
    - SHORT: 卖空开空仓（做空）
    - COVER: 买入平空仓
    - HOLD: 持仓不动
    """
    
    # 有效的交易信号
    VALID_SIGNALS = {'BUY', 'SELL', 'SHORT', 'COVER', 'HOLD'}

    @abstractmethod
    def get_signal(self,
                   ticker: str,
                   new_data: pd.DataFrame,
                   verbose: bool = True) -> Tuple[Dict, float]:
        """
        分析数据并生成交易信号。
        
        Args:
            ticker: 股票代码
            new_data: OHLCV DataFrame，索引为时间戳，
                      包含 'open', 'high', 'low', 'close', 'volume' 列
            verbose: 是否打印详细信息
            
        Returns:
            Tuple[signal_dict, current_price]:
                - signal_dict: {
                    'signal': 'BUY' | 'SELL' | 'SHORT' | 'COVER' | 'HOLD',
                    'confidence_score': int (1-10),
                    'reason': str
                  }
                - current_price: float
        """
        pass
    
    def _validate_signal(self, signal: str) -> str:
        """验证并规范化信号。"""
        signal = signal.upper().strip()
        if signal not in self.VALID_SIGNALS:
            return 'HOLD'
        return signal

    def __str__(self):
        return self.__class__.__name__


# 信号输出结构示例
SIGNAL_OUTPUT_EXAMPLE = {
    "signal": "BUY",  # or "SELL", "SHORT", "COVER", "HOLD"
    "confidence_score": 8,  # 1-10
    "reason": "价格跌破布林带下轨，RSI 超卖"
}