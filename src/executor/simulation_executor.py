# src/executor/simulation_executor.py

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Literal, Dict, Any, List
from src.executor.base_executor import BaseExecutor # 导入基类

class SimulationExecutor(BaseExecutor):
    """
    模拟执行器：用于回测环境，**仅负责模拟交易执行和计算费用**。
    它不管理资金和仓位。
    """

    def __init__(self, finance_params: Dict[str, float]):
        """
        初始化模拟执行器，加载交易所需的财务参数。
        """
        # 交易执行所需参数
        self.COMMISSION_RATE = finance_params.get('COMMISSION_RATE', 0.0003)
        self.SLIPPAGE_RATE = finance_params.get('SLIPPAGE_RATE', 0.0001)
        self.MIN_LOT_SIZE = finance_params.get('MIN_LOT_SIZE', 100)
        self.MAX_ALLOCATION = finance_params.get('MAX_ALLOCATION', 0.2)
        self.STAMP_DUTY_RATE = finance_params.get('STAMP_DUTY_RATE', 0.001)

        print("💡 SimulationExecutor 初始化成功。")


    def execute_trade(self,
                      timestamp: datetime,
                      signal: Literal["BUY", "SELL"],
                      current_price: float,
                      current_position: float,
                      current_cash: float,
                      avg_cost: float) -> Dict[str, Any]:
        """实现 BaseExecutor 接口：模拟交易执行并返回结果。"""
        
        if current_price <= 0:
             return self._fail_result("价格无效。")

        # 始终使用传入的最新资金和仓位状态
        current_equity = current_cash + (current_position * current_price)

        if signal == 'BUY':
            return self._execute_buy(current_price, current_cash, current_equity)
        
        elif signal == 'SELL' and current_position > 0:
            return self._execute_sell(current_price, current_position)
            
        return self._fail_result(f"无执行信号或无仓位可卖 ({signal}).")
        
    def _fail_result(self, reason: str) -> Dict[str, Any]:
        """返回失败的交易结果模板。"""
        return {
            'executed': False,
            'trade_type': 'N/A',
            'executed_qty': 0.0,
            'executed_price': 0.0,
            'fee': 0.0,
            'log_message': f"模拟交易失败: {reason}"
        }

    def _execute_buy(self, current_price: float, current_cash: float, current_equity: float) -> Dict[str, Any]:
        """模拟买入逻辑。"""
        
        # 1. 计算最大可用资金 (基于总资产的MAX_ALLOCATION)
        max_capital_for_trade = current_equity * self.MAX_ALLOCATION
        available_cash_to_use = min(current_cash, max_capital_for_trade)
        
        # 2. 计算可买入数量 (四舍五入到最小交易单位 MIN_LOT_SIZE)
        qty_to_buy_float = available_cash_to_use / current_price
        qty_to_buy = np.floor(qty_to_buy_float / self.MIN_LOT_SIZE) * self.MIN_LOT_SIZE
        
        if qty_to_buy < self.MIN_LOT_SIZE:
            return self._fail_result("计算数量低于最小交易单位。")

        # 3. 计算实际成交细节
        execution_price = current_price * (1 + self.SLIPPAGE_RATE) # 考虑滑点
        fee = qty_to_buy * execution_price * self.COMMISSION_RATE  # 手续费
        
        total_cost = qty_to_buy * execution_price + fee
        
        if total_cost <= current_cash:
            # 交易成功
            return {
                'executed': True,
                'trade_type': 'BUY',
                'executed_qty': qty_to_buy,
                'executed_price': execution_price,
                'fee': fee, # 仅手续费
                'log_message': f"模拟买入 {qty_to_buy:,.0f} 股 @ ${execution_price:.2f}"
            }
        else:
            return self._fail_result("现金不足以支付交易成本。")

    def _execute_sell(self, current_price: float, current_position: float) -> Dict[str, Any]:
        """模拟卖出逻辑。"""
        
        qty_to_sell = current_position # 默认卖出全部仓位
        
        # 1. 计算实际成交细节
        execution_price = current_price * (1 - self.SLIPPAGE_RATE) # 考虑滑点
        income_before_fee = qty_to_sell * execution_price
        
        # 2. 计算费用 (手续费 + 印花税)
        commission = income_before_fee * self.COMMISSION_RATE
        stamp_duty = income_before_fee * self.STAMP_DUTY_RATE 
        total_fee = commission + stamp_duty
        
        # 交易成功
        return {
            'executed': True,
            'trade_type': 'SELL',
            'executed_qty': qty_to_sell,
            'executed_price': execution_price,
            'fee': total_fee, # 总费用 (手续费 + 印花税)
            'log_message': f"模拟卖出 {qty_to_sell:,.0f} 股 @ ${execution_price:.2f}"
        }