# src/executor/alpaca_executor.py

import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest, ClosePositionRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from alpaca.trading.models import Position
from alpaca.common.exceptions import APIError
from src.executor.base_executor import BaseExecutor
from datetime import datetime
from typing import Literal, Dict, Any, List
import pandas as pd
import numpy as np
from typing import Literal, Dict, Any, Optional # 导入 Optional 修复 Python 3.9 兼容性

# --- 配置 ---
load_dotenv()

# --- 交易参数 ---
# 每次交易动用总资产的比例（用于计算购买数量）
MAX_ALLOCATION_RATE = 0.2
MIN_LOT_SIZE = 1 # Alpaca 允许 fractional share，但我们这里简化为 1 股最小单位。

class AlpacaExecutor(BaseExecutor):
    """
    Alpaca 交易执行器：用于实盘或模拟交易环境，对接 Alpaca API。
    它实现了 BaseExecutor 接口。
    """
    def __init__(self, paper: bool = True, max_allocation_rate: float = MAX_ALLOCATION_RATE):
        self.paper = paper
        self.MAX_ALLOCATION_RATE = max_allocation_rate
        self.trade_log: List[Dict[str, Any]] = [] # 在实盘模式下，仍然记录本地交易尝试
        
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
            raise

    def get_account_status(self, current_price: float = 0.0) -> Dict[str, float]:
        """实现 BaseExecutor 接口：获取 Alpaca 账户的实时状态。"""
        try:
            account = self.trading_client.get_account()
            
            # 获取现金 (Cash)
            cash = float(account.cash) 
            
            # 获取总资产 (Equity)
            equity = float(account.equity)
            
            # 查找持仓 (Position)
            # 注意：Alpaca 返回的是 Account 级别数据，Position 需要额外 API 调用
            # 考虑到回测/实时运行需要指定 Ticker，这里 Position/Avg_cost 的值设为 0
            # 因为 Account API 并没有返回某个 Ticker 的 Position 信息
            # 在 execute_trade 中会单独查询 Position
            
            return {
                'cash': cash,
                'position': 0.0, 
                'avg_cost': 0.0,
                'equity': equity,
                'market_value': equity - cash # 这是一个近似值
            }
        except Exception as e:
            print(f"❌ 无法连接或获取 Alpaca 账户状态: {e}")
            return {'cash': 0.0, 'position': 0.0, 'avg_cost': 0.0, 'equity': 0.0, 'market_value': 0.0}

    def execute_trade(self,
                      timestamp: datetime, # 在实盘中 timestamp 仅用于 log
                      signal: Literal["BUY", "SELL"],
                      current_price: float) -> bool:
        """实现 BaseExecutor 接口：提交订单到 Alpaca。"""

        ticker = "TSLA" # 假设我们只交易 TSLA，实际应用中应该传递 Ticker

        if signal == 'BUY':
            return self._execute_alpaca_buy(timestamp, ticker, current_price)
        
        elif signal == 'SELL':
            return self._execute_alpaca_sell(timestamp, ticker)
            
        return False

    def _execute_alpaca_buy(self, timestamp: datetime, ticker: str, current_price: float) -> bool:
        """执行 Alpaca 买入逻辑。"""
        try:
            # 1. 获取当前账户总资产
            account = self.trading_client.get_account()
            equity = float(account.equity)
            cash = float(account.cash)
            
            # 2. 计算可用于交易的金额
            capital_to_use = min(cash, equity * self.MAX_ALLOCATION_RATE)
            
            if capital_to_use <= 0 or current_price <= 0:
                print("  ❌ Alpaca BUY 失败：资金不足或价格无效。")
                return False

            # 3. 计算购买数量 (四舍五入到最小单位，并向下取整)
            qty_float = capital_to_use / current_price
            qty = np.floor(qty_float / MIN_LOT_SIZE) * MIN_LOT_SIZE
            
            if qty < MIN_LOT_SIZE:
                print(f"  ❌ Alpaca BUY 失败：计算数量 {qty} 低于最小交易单位 {MIN_LOT_SIZE}。")
                return False

            # 4. 提交市价买入订单 (Market Order)
            order_request = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY, # 当日有效
            )
            order = self.trading_client.submit_order(order_request)
            
            self.trade_log.append({
                'time': datetime.now(), 'type': 'BUY', 'qty': qty,
                'price': current_price, 'fee': 0.0, 'net_pnl': 0.0, 
                'current_pos': qty, 'order_id': order.id, 'status': order.status.value
            })

            print(f"  ⭐ Alpaca 订单提交成功: 买入 {qty:,.0f} 股 {ticker}。订单状态: {order.status.value}")
            return True

        except APIError as e:
            print(f"  ❌ Alpaca API 错误 (BUY): {e}")
            return False
        except Exception as e:
            print(f"  ❌ 交易执行失败 (BUY): {e}")
            return False

    def _execute_alpaca_sell(self, timestamp: datetime, ticker: str) -> bool:
        """执行 Alpaca 卖出逻辑 (平仓)。"""
        try:
            # 1. 获取当前持仓
            current_position = self._get_current_position(ticker)
            
            if not current_position or float(current_position.qty) <= 0:
                print(f"  ⚠️ Alpaca SELL 失败：{ticker} 无持仓可平。")
                return False

            # 2. 提交平仓请求 (ClosePositionRequest 将卖出全部持仓)
            close_request = ClosePositionRequest(
                symbol=ticker
            )
            # close_position API 会返回一个 Order 对象
            order = self.trading_client.close_position(close_request)
            
            qty_to_sell = float(current_position.qty) # 记录平仓数量
            
            self.trade_log.append({
                'time': datetime.now(), 'type': 'SELL', 'qty': qty_to_sell,
                'price': float(current_position.current_price), 'fee': 0.0, 'net_pnl': 0.0, 
                'current_pos': 0.0, 'order_id': order.id, 'status': order.status.value
            })

            print(f"  🌟 Alpaca 订单提交成功: 平仓 {qty_to_sell:,.0f} 股 {ticker}。订单状态: {order.status.value}")
            return True

        except APIError as e:
            print(f"  ❌ Alpaca API 错误 (SELL): {e}")
            return False
        except Exception as e:
            print(f"  ❌ 交易执行失败 (SELL): {e}")
            return False
            
    def get_trade_log(self) -> pd.DataFrame:
        """返回交易日志 DataFrame。在实盘模式中，这只记录尝试提交的订单。"""
        return pd.DataFrame(self.trade_log)