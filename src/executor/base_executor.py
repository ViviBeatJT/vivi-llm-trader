# src/core/base_executor.py

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal, Dict, Any, List
import pandas as pd

class BaseExecutor(ABC):
    """
    交易执行器基类 (Interface)。
    所有交易执行器（模拟或实盘）都必须实现这些方法。
    """

    @abstractmethod
    def execute_trade(self,
                      timestamp: datetime,
                      signal: Literal["BUY", "SELL"],
                      current_price: float) -> bool:
        """
        根据信号执行交易（模拟或实盘下单）。
        
        Args:
            timestamp: 当前时间点。
            signal: 交易信号 ("BUY" 或 "SELL")。
            current_price: 策略执行时的最新价格。
            
        Returns:
            bool: 交易是否成功执行。
        """
        pass

    @abstractmethod
    def get_account_status(self, current_price: float) -> Dict[str, float]:
        """
        获取当前的账户状态（现金、总资产、持仓）。
        
        Args:
            current_price: 用于计算当前持仓市值的最新价格。
            
        Returns:
            Dict: 包含 'cash', 'position', 'avg_cost', 'equity' 的字典。
        """
        pass

    @abstractmethod
    def get_trade_log(self) -> pd.DataFrame:
        """返回交易日志 DataFrame。"""
        pass