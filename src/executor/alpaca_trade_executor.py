# src/executor/alpaca_trade_executor.py

import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest, ClosePositionRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from alpaca.trading.models import Position
from alpaca.common.exceptions import APIError
from src.executor.base_executor import BaseExecutor
from datetime import datetime
from typing import Literal, Dict, Any, Optional
import pandas as pd
import numpy as np

# --- 配置 ---
load_dotenv()

# --- 交易参数 ---
MAX_ALLOCATION_RATE = 0.2
MIN_LOT_SIZE = 1 # Alpaca 允许 fractional share，但我们这里简化为 1 股最小单位。

class AlpacaExecutor(BaseExecutor):
    """
    Alpaca 交易执行器：用于实盘或模拟交易环境，对接 Alpaca API。
    它实现了 BaseExecutor 接口。
    
    职责：仅负责将交易信号转换为 Alpaca 订单并提交。
    """
    def __init__(self, paper: bool = True, max_allocation_rate: float = MAX_ALLOCATION_RATE):
        self.paper = paper
        self.MAX_ALLOCATION_RATE = max_allocation_rate
        # 在实盘模式下，P&L和持仓由 Alpaca 账户管理，PositionManager 负责跟踪本地日志。
        
        # 初始化 Alpaca 客户端
        self.trading_client = TradingClient(
            os.getenv('ALPACA_API_KEY_ID'), 
            os.getenv('ALPACA_SECRET_KEY'), 
            paper=self.paper
        )
        mode = "模拟 (Paper)" if self.paper else "实盘 (Live)"
        print(f"🚀 AlpacaExecutor 初始化成功。工作模式: {mode}")

    def _get_current_position(self, ticker: str) -> Optional[Position]:
        """获取指定股票的当前持仓。"""
        try:
            position_data = self.trading_client.get_open_position_by_symbol(ticker)
            return position_data
        except APIError as e:
            if "position not found" in str(e):
                return None
            # 实盘模式下，如果 API 失败，必须抛出错误
            raise

    def execute_trade(self,
                      timestamp: datetime,
                      signal: Literal["BUY", "SELL"],
                      current_price: float,
                      current_position: float,
                      current_cash: float,
                      avg_cost: float) -> Dict[str, Any]:
        """
        实现 BaseExecutor 接口：提交订单到 Alpaca 并返回结果。
        注意：实盘模式下，成交价格、数量和费用需要等待订单成交后才能确定。
        为简化，我们假设市价单立即成交，并返回预期结果。PositionManager 会记录这些预期的交易。
        """

        ticker = "TSLA" # 假设我们只交易 TSLA，实际应用中应该传递 Ticker

        if current_price <= 0:
            return self._fail_result("价格无效。")

        if signal == 'BUY':
            return self._execute_alpaca_buy(ticker, current_price, current_cash)
        
        elif signal == 'SELL' and current_position > 0:
            return self._execute_alpaca_sell(ticker, current_position)
            
        return self._fail_result(f"无执行信号或无仓位可卖 ({signal}).")

    def _fail_result(self, reason: str) -> Dict[str, Any]:
        """返回失败的交易结果模板。"""
        return {
            'executed': False,
            'trade_type': 'N/A',
            'executed_qty': 0.0,
            'executed_price': 0.0,
            'fee': 0.0,
            'log_message': f"Alpaca 交易失败: {reason}"
        }

    def _execute_alpaca_buy(self, ticker: str, current_price: float, current_cash: float) -> Dict[str, Any]:
        """提交 Alpaca 买入订单，并返回预期结果。"""
        try:
            # 1. 获取当前账户总资产 (需要 API 调用)
            account = self.trading_client.get_account()
            equity = float(account.equity)
            
            # 2. 计算可用于交易的金额
            capital_to_use = min(current_cash, equity * self.MAX_ALLOCATION_RATE)
            
            if capital_to_use <= 0:
                return self._fail_result("资金不足。")

            # 3. 计算购买数量
            qty_float = capital_to_use / current_price
            qty = np.floor(qty_float / MIN_LOT_SIZE) * MIN_LOT_SIZE
            
            if qty < MIN_LOT_SIZE:
                return self._fail_result("计算数量低于最小交易单位。")

            # 4. 提交市价买入订单 (Market Order)
            order_request = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            order = self.trading_client.submit_order(order_request)
            
            # **注意: 实盘中需要等待订单填充才能获取真实的 executed_price 和 fee。
            # 为了让 PositionManager 能够继续工作，我们返回一个预期结果。**
            
            # 假设 Alpaca 默认手续费为 0 (Commission-free)
            # 假设成交价格就是 current_price
            
            return {
                'executed': True,
                'trade_type': 'BUY',
                'executed_qty': qty,
                'executed_price': current_price, 
                'fee': 0.0, 
                'log_message': f"Alpaca 订单 {order.id} 已提交 (买入 {qty:,.0f} 股，状态: {order.status.value})"
            }

        except APIError as e:
            return self._fail_result(f"Alpaca API 错误: {e}")
        except Exception as e:
            return self._fail_result(f"交易执行失败: {e}")

    def _execute_alpaca_sell(self, ticker: str, current_position: float) -> Dict[str, Any]:
        """提交 Alpaca 卖出订单 (平仓) 并返回预期结果。"""
        try:
            # 1. 提交平仓请求
            close_request = ClosePositionRequest(symbol=ticker)
            order = self.trading_client.close_position(close_request)
            
            # **注意: 实际成交数量/价格/费用需要等待订单填充。**
            # 为简化，我们假设卖出全部持仓，费用为 0。

            return {
                'executed': True,
                'trade_type': 'SELL',
                'executed_qty': current_position, # 预期卖出全部
                'executed_price': 0.0, # 预期价格 (P/L由PositionManager计算，这里给0.0)
                'fee': 0.0, 
                'log_message': f"Alpaca 订单 {order.id} 已提交 (平仓 {current_position:,.0f} 股，状态: {order.status.value})"
            }

        except APIError as e:
            return self._fail_result(f"Alpaca API 错误: {e}")
        except Exception as e:
            return self._fail_result(f"交易执行失败: {e}")