# src/manager/position_manager.py

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Literal, Dict, Any, List, Optional
from src.executor.base_executor import BaseExecutor # 导入 BaseExecutor

class PositionManager:
    """
    仓位和资金管理器。
    职责：
    1. 统一管理账户状态（现金、持仓、平均成本）。
    2. 执行交易逻辑（通过 BaseExecutor 成员）。
    3. 根据执行结果，统一更新资金、仓位并计算净盈亏 (P&L)。
    """

    def __init__(self, 
                 executor: BaseExecutor, 
                 finance_params: Dict[str, float]):
        
        # 交易执行器实例（可为 SimulationExecutor 或 AlpacaExecutor）
        self.executor = executor
        
        # 财务参数 (用于初始化和 P&L 计算)
        self.INITIAL_CAPITAL = finance_params.get('INITIAL_CAPITAL', 100000.0)
        
        # 核心跟踪变量
        self.cash = self.INITIAL_CAPITAL  # 当前可用现金
        self.position = 0.0              # 当前持仓数量 (股)
        self.avg_cost = 0.0              # 当前持仓平均成本
        self.trade_log: List[Dict[str, Any]] = []  # 记录所有交易详情
        
        print(f"💰 PositionManager 初始化成功。初始资金: ${self.cash:,.2f}。使用执行器: {self.executor.__class__.__name__}")

    def get_account_status(self, current_price: float) -> Dict[str, float]:
        """
        获取当前的账户状态（现金、总资产、持仓数量、平均成本）。
        """
        market_value = self.position * current_price
        equity = self.cash + market_value
        return {
            'cash': self.cash,
            'position': self.position,
            'avg_cost': self.avg_cost,
            'equity': equity,
            'market_value': market_value
        }

    def execute_and_update(self,
                           timestamp: datetime,
                           signal: Literal["BUY", "SELL"],
                           current_price: float) -> bool:
        """
        步骤 1: 调用 Executor 计算交易结果或提交实盘订单。
        步骤 2: 根据 Executor 返回的结果，更新 Position Manager 的内部状态。
        """
        # 1. 调用 Executor 执行交易，获取结果
        execution_result = self.executor.execute_trade(
            timestamp=timestamp, 
            signal=signal, 
            current_price=current_price,
            current_position=self.position,
            current_cash=self.cash,
            avg_cost=self.avg_cost # 传入平均成本供Executor使用（如果需要）
        )
        
        if not execution_result.get('executed', False):
            print(f"  ⚠️ Executor 未执行交易: {execution_result.get('log_message', '未知原因')}")
            return False

        # 2. 从结果中提取关键数据
        trade_type = execution_result['trade_type']
        executed_qty = execution_result['executed_qty']
        executed_price = execution_result['executed_price'] # 实际成交价格 (含滑点)
        fee = execution_result['fee'] # 总费用 (手续费 + 印花税)
        
        # 3. 统一的资金和仓位更新逻辑 (这是 PositionManager 的核心价值)
        net_pnl = 0.0
        
        if trade_type == 'BUY':
            total_cost = executed_qty * executed_price + fee
            
            # **更新仓位和平均成本**
            new_position = self.position + executed_qty
            # 避免除以零
            if new_position > 0:
                self.avg_cost = (self.position * self.avg_cost + executed_qty * executed_price) / new_position
            else:
                self.avg_cost = 0.0
                
            self.position = new_position
            
            # **更新现金**
            self.cash -= total_cost
            log_message = f"  ⭐ 统一更新: 买入 {executed_qty:,.0f} 股 @ ${executed_price:.2f} | 费用: ${fee:.2f} | 剩余现金: ${self.cash:,.2f}"
            
        elif trade_type == 'SELL' and executed_qty > 0:
            
            # **计算本次交易的 净收益 (P&L)**
            capital_cost = executed_qty * self.avg_cost
            income_before_fee = executed_qty * executed_price
            net_pnl = income_before_fee - fee - capital_cost
            
            # **更新现金**
            self.cash += (income_before_fee - fee) 
            
            # **更新仓位**
            self.position -= executed_qty
            # 如果仓位完全清零，则平均成本归零
            if self.position == 0.0:
                self.avg_cost = 0.0 
                
            log_message = f"  🌟 统一更新: 卖出 {executed_qty:,.0f} 股 @ ${executed_price:.2f} | 净P&L: ${net_pnl:,.2f}"

        else:
            log_message = "  ❌ 统一更新失败: 执行器返回结果无效。"
            
        print(log_message)
        
        # 4. 记录交易日志
        self.trade_log.append({
            'time': timestamp, 
            'type': trade_type, 
            'qty': executed_qty,
            'price': executed_price, 
            'fee': fee, 
            'net_pnl': net_pnl, 
            'current_pos': self.position, 
            'avg_cost': self.avg_cost
        })
        
        return True

    def get_trade_log(self) -> pd.DataFrame:
        """返回交易日志 DataFrame。"""
        return pd.DataFrame(self.trade_log)