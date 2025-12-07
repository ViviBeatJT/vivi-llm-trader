# not useful right now.

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Literal, Dict, Any, List

class PositionManager:
    """
    负责回测中的所有金融计算、仓位管理和交易记录。
    包括：资金管理、交易数量计算、成本核算、手续费/滑点扣除。
    """

    def __init__(self, finance_params: Dict[str, float]):
        """
        初始化仓位管理器，加载财务参数。
        """
        self.INITIAL_CAPITAL = finance_params.get('INITIAL_CAPITAL', 100000.0)
        self.COMMISSION_RATE = finance_params.get('COMMISSION_RATE', 0.0003)
        self.SLIPPAGE_RATE = finance_params.get('SLIPPAGE_RATE', 0.0001)
        self.MIN_LOT_SIZE = finance_params.get('MIN_LOT_SIZE', 100)
        self.MAX_ALLOCATION = finance_params.get('MAX_ALLOCATION', 0.2)
        self.STAMP_DUTY_RATE = finance_params.get('STAMP_DUTY_RATE', 0.001)

        # 核心跟踪变量
        self.cash = self.INITIAL_CAPITAL  # 当前可用现金
        self.position = 0.0              # 当前持仓数量 (股)
        self.avg_cost = 0.0              # 当前持仓平均成本
        self.trade_log: List[Dict[str, Any]] = []  # 记录所有交易详情

    @property
    def current_equity(self, current_price: float) -> float:
        """计算实时总资产 (Total Equity)。"""
        return self.cash + self.position * current_price

    def execute_trade(self, 
                      timestamp: datetime, 
                      signal: Literal["BUY", "SELL"], 
                      current_price: float) -> bool:
        """
        根据信号执行交易，更新仓位和现金。
        
        Args:
            timestamp: 当前时间点。
            signal: 交易信号 ("BUY" 或 "SELL")。
            current_price: 策略执行时的最新价格。
            
        Returns:
            bool: 交易是否成功执行。
        """
        if current_price <= 0:
            print(f"  ⚠️ 交易失败：价格无效 ({current_price:.2f})。")
            return False
            
        current_equity = self.current_equity(current_price)

        if signal == 'BUY':
            return self._execute_buy(timestamp, current_price, current_equity)
        
        elif signal == 'SELL' and self.position > 0:
            return self._execute_sell(timestamp, current_price)
            
        return False # HOLD 或 SELL 但无持仓

    def _execute_buy(self, timestamp: datetime, current_price: float, current_equity: float) -> bool:
        """执行买入逻辑和计算。"""
        
        # 1. 计算最大可用资金 (基于总资产的MAX_ALLOCATION)
        max_capital_for_trade = current_equity * self.MAX_ALLOCATION
        available_cash_to_use = min(self.cash, max_capital_for_trade)
        
        # 2. 计算可买入数量 (四舍五入到最小交易单位 MIN_LOT_SIZE)
        qty_to_buy_float = available_cash_to_use / current_price
        qty_to_buy = np.floor(qty_to_buy_float / self.MIN_LOT_SIZE) * self.MIN_LOT_SIZE
        
        if qty_to_buy == 0:
            return False

        # 考虑滑点和手续费后的实际执行价格
        execution_price = current_price * (1 + self.SLIPPAGE_RATE)
        
        # 计算总成本 (股本金 + 手续费)
        fee = qty_to_buy * execution_price * self.COMMISSION_RATE
        total_cost = qty_to_buy * execution_price + fee
        
        if total_cost <= self.cash:
            # **更新仓位和平均成本**
            new_position = self.position + qty_to_buy
            # 重新计算平均成本 (加权平均)
            self.avg_cost = (self.position * self.avg_cost + qty_to_buy * execution_price) / new_position
            self.position = new_position
            
            # **更新现金**
            self.cash -= total_cost
            
            # 记录交易
            self.trade_log.append({
                'time': timestamp, 'type': 'BUY', 'qty': qty_to_buy,
                'price': execution_price, 'fee': fee, 'net_pnl': 0.0, 
                'current_pos': self.position, 'avg_cost': self.avg_cost
            })
            print(f"  ⭐ 交易执行: 买入 {qty_to_buy:,.0f} 股 @ ${execution_price:.2f} | 费用: ${fee:.2f} | 剩余现金: ${self.cash:,.2f}")
            return True
        else:
            # print("  ❌ 交易失败：资金不足，无法覆盖交易成本。")
            return False

    def _execute_sell(self, timestamp: datetime, current_price: float) -> bool:
        """执行卖出逻辑和计算 (全部平仓)。"""
        
        qty_to_sell = self.position 
        
        # 考虑滑点后的实际执行价格
        execution_price = current_price * (1 - self.SLIPPAGE_RATE)
        
        # 计算收入 (Income)
        income_before_fee = qty_to_sell * execution_price
        
        # 计算总费用 (手续费 + 印花税)
        commission = income_before_fee * self.COMMISSION_RATE
        stamp_duty = income_before_fee * self.STAMP_DUTY_RATE 
        total_fee = commission + stamp_duty
        
        # **计算本次交易的 净收益 (P&L)**
        capital_cost = qty_to_sell * self.avg_cost # 平仓部分的成本
        net_pnl = income_before_fee - total_fee - capital_cost
        
        # **更新现金**
        self.cash += (income_before_fee - total_fee) # 实际入账金额
        
        # **更新仓位**
        self.position = 0.0 # 仓位归零
        self.avg_cost = 0.0 # 平均成本归零
        
        # 记录交易
        self.trade_log.append({
            'time': timestamp, 'type': 'SELL', 'qty': qty_to_sell,
            'price': execution_price, 'fee': total_fee, 'net_pnl': net_pnl, 
            'current_pos': self.position, 'avg_cost': self.avg_cost
        })
        print(f"  🌟 交易执行: 卖出 {qty_to_sell:,.0f} 股 @ ${execution_price:.2f} | 净P&L: ${net_pnl:,.2f}")
        return True

    def get_trade_log(self) -> pd.DataFrame:
        """返回交易日志 DataFrame。"""
        return pd.DataFrame(self.trade_log)