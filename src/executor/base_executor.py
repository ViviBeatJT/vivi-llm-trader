# src/executor/base_executor.py

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal, Dict, Any, List, Optional
import pandas as pd

class BaseExecutor(ABC):
    """
    交易执行器基类 (Interface)。
    职责：根据信号和当前价格，计算和执行交易（模拟计算或提交实盘订单）。
    它**不负责**资金、仓位和 P&L 的管理。
    """

    @abstractmethod
    def execute_trade(self,
                      timestamp: datetime,
                      signal: Literal["BUY", "SELL"],
                      current_price: float,
                      current_position: float,  # 新增：传入当前持仓
                      current_cash: float,      # 新增：传入当前现金
                      avg_cost: float) -> Dict[str, Any]:
        """
        根据信号执行交易（模拟或实盘下单）。
        
        Args:
            timestamp: 当前时间点。
            signal: 交易信号 ("BUY" 或 "SELL")。
            current_price: 策略执行时的最新价格。
            current_position: PositionManager 传入的当前持仓数量。
            current_cash: PositionManager 传入的当前可用现金。
            avg_cost: PositionManager 传入的当前平均成本。
            
        Returns:
            Dict: 包含交易结果的字典。
            --- 必须包含的字段 ---
            'executed': bool (是否成功执行交易)
            'trade_type': Literal["BUY", "SELL"]
            'executed_qty': float (实际成交数量)
            'executed_price': float (实际成交价格，含滑点)
            'fee': float (总费用，包含手续费和印花税)
            'log_message': str (日志信息)
        """
        pass