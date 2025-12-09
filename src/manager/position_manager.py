# src/manager/position_manager.py

from datetime import datetime, timezone
from typing import Dict, Any, Optional, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from src.data_fetcher.alpaca_data_fetcher import AlpacaDataFetcher


class PositionManager:
    """
    仓位管理器 - 管理交易仓位、现金和交易记录。
    
    支持两种模式：
    1. 本地模拟模式：使用 SimulationExecutor，完全本地计算
    2. API 模式：使用 AlpacaExecutor，可从 API 同步仓位状态
    """

    def __init__(self, executor, finance_params: Dict[str, Any], data_fetcher: Optional['AlpacaDataFetcher'] = None):
        """
        初始化仓位管理器。
        
        Args:
            executor: 交易执行器（SimulationExecutor 或 AlpacaExecutor）
            finance_params: 财务参数字典，包含：
                - INITIAL_CAPITAL: 初始资金
                - COMMISSION_RATE: 佣金率
                - SLIPPAGE_RATE: 滑点率
                - MIN_LOT_SIZE: 最小交易手数
                - MAX_ALLOCATION: 最大仓位比例
            data_fetcher: 数据获取器（可选，用于从 API 同步仓位）
        """
        self.executor = executor
        self.finance_params = finance_params
        self.data_fetcher = data_fetcher
        
        # 本地状态
        self._cash = finance_params.get('INITIAL_CAPITAL', 100000.0)
        self._position = 0.0  # 持仓数量
        self._avg_cost = 0.0  # 平均成本
        
        # 交易记录
        self._trade_log = []
        
        # 同步标志
        self._synced = False
    
    def sync_from_api(self, ticker: str) -> bool:
        """
        从 API 同步仓位状态。
        
        Args:
            ticker: 股票代码
            
        Returns:
            bool: 是否同步成功
        """
        if not self.data_fetcher:
            print("⚠️ 未配置 data_fetcher，无法从 API 同步")
            return False
        
        try:
            status = self.data_fetcher.sync_position_status(ticker)
            
            if not status:
                print("❌ 从 API 同步仓位失败")
                return False
            
            # 更新本地状态
            self._cash = status.get('cash', self._cash)
            self._position = status.get('position', 0.0)
            self._avg_cost = status.get('avg_cost', 0.0)
            self._synced = True
            
            print(f"✅ 仓位同步成功:")
            print(f"   现金: ${self._cash:,.2f}")
            print(f"   持仓: {self._position:.0f} 股")
            if self._position > 0:
                print(f"   均价: ${self._avg_cost:.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ 同步仓位时出错: {e}")
            return False
    
    def get_account_status(self, current_price: float = 0.0) -> Dict[str, Any]:
        """
        获取当前账户状态。
        
        Args:
            current_price: 当前价格（用于计算市值和权益）
            
        Returns:
            dict: 账户状态，包含：
                - cash: 现金
                - position: 持仓数量
                - avg_cost: 平均成本
                - market_value: 持仓市值
                - equity: 总权益
                - unrealized_pnl: 未实现盈亏
        """
        market_value = self._position * current_price
        equity = self._cash + market_value
        
        unrealized_pnl = 0.0
        if self._position > 0 and self._avg_cost > 0:
            unrealized_pnl = (current_price - self._avg_cost) * self._position
        
        return {
            'cash': self._cash,
            'position': self._position,
            'avg_cost': self._avg_cost,
            'market_value': market_value,
            'equity': equity,
            'unrealized_pnl': unrealized_pnl,
            'synced': self._synced
        }
    
    def execute_and_update(self, 
                          timestamp: datetime, 
                          signal: str, 
                          current_price: float,
                          ticker: str = "UNKNOWN") -> bool:
        """
        执行交易并更新仓位。
        
        Args:
            timestamp: 交易时间
            signal: 交易信号 ('BUY' 或 'SELL')
            current_price: 当前价格
            ticker: 股票代码
            
        Returns:
            bool: 是否执行成功
        """
        if signal not in ["BUY", "SELL"]:
            return False
        
        # 计算交易数量
        qty = self._calculate_trade_qty(signal, current_price)
        
        if qty <= 0:
            print(f"⚠️ 计算交易数量为 0，跳过交易")
            return False
        
        # 执行交易
        try:
            result = self.executor.execute(
                signal=signal,
                qty=qty,
                price=current_price,
                ticker=ticker
            )
            
            if not result.get('success', False):
                print(f"❌ 交易执行失败: {result.get('error', 'Unknown error')}")
                return False
            
            # 更新本地状态
            executed_qty = result.get('qty', qty)
            executed_price = result.get('price', current_price)
            fee = result.get('fee', 0.0)
            
            self._update_position(signal, executed_qty, executed_price, fee)
            
            # 记录交易
            self._record_trade(
                timestamp=timestamp,
                signal=signal,
                qty=executed_qty,
                price=executed_price,
                fee=fee,
                ticker=ticker
            )
            
            return True
            
        except Exception as e:
            print(f"❌ 交易执行异常: {e}")
            return False
    
    def _calculate_trade_qty(self, signal: str, current_price: float) -> int:
        """
        计算交易数量。
        
        Args:
            signal: 交易信号
            current_price: 当前价格
            
        Returns:
            int: 交易数量
        """
        min_lot_size = self.finance_params.get('MIN_LOT_SIZE', 10)
        max_allocation = self.finance_params.get('MAX_ALLOCATION', 0.2)
        
        if signal == "BUY":
            # 计算可用资金
            available_cash = self._cash * max_allocation
            
            # 考虑佣金和滑点
            commission_rate = self.finance_params.get('COMMISSION_RATE', 0.0003)
            slippage_rate = self.finance_params.get('SLIPPAGE_RATE', 0.0001)
            effective_price = current_price * (1 + commission_rate + slippage_rate)
            
            # 计算可买数量
            max_qty = int(available_cash / effective_price)
            
            # 取整到最小手数
            qty = (max_qty // min_lot_size) * min_lot_size
            
            return max(qty, 0)
            
        elif signal == "SELL":
            # 卖出全部持仓
            qty = int(self._position)
            
            # 取整到最小手数
            qty = (qty // min_lot_size) * min_lot_size
            
            return max(qty, 0)
        
        return 0
    
    def _update_position(self, signal: str, qty: int, price: float, fee: float):
        """
        更新仓位状态。
        
        Args:
            signal: 交易信号
            qty: 交易数量
            price: 成交价格
            fee: 交易费用
        """
        if signal == "BUY":
            # 买入：增加持仓，减少现金
            total_cost = qty * price + fee
            
            # 更新平均成本
            if self._position > 0:
                total_value = self._position * self._avg_cost + qty * price
                self._avg_cost = total_value / (self._position + qty)
            else:
                self._avg_cost = price
            
            self._position += qty
            self._cash -= total_cost
            
        elif signal == "SELL":
            # 卖出：减少持仓，增加现金
            proceeds = qty * price - fee
            
            self._position -= qty
            self._cash += proceeds
            
            # 如果清仓，重置平均成本
            if self._position <= 0:
                self._position = 0
                self._avg_cost = 0.0
    
    def _record_trade(self, 
                     timestamp: datetime, 
                     signal: str, 
                     qty: int, 
                     price: float, 
                     fee: float,
                     ticker: str):
        """记录交易。"""
        # 计算本次交易盈亏（仅对卖出有意义）
        net_pnl = 0.0
        if signal == "SELL" and self._avg_cost > 0:
            net_pnl = (price - self._avg_cost) * qty - fee
        
        trade_record = {
            'time': timestamp,
            'ticker': ticker,
            'type': signal,
            'qty': qty,
            'price': price,
            'fee': fee,
            'net_pnl': net_pnl,
            'cash_after': self._cash,
            'position_after': self._position
        }
        
        self._trade_log.append(trade_record)
    
    def get_trade_log(self) -> pd.DataFrame:
        """
        获取交易记录。
        
        Returns:
            pd.DataFrame: 交易记录表
        """
        if not self._trade_log:
            return pd.DataFrame()
        
        return pd.DataFrame(self._trade_log)
    
    def reset(self):
        """重置仓位管理器状态。"""
        self._cash = self.finance_params.get('INITIAL_CAPITAL', 100000.0)
        self._position = 0.0
        self._avg_cost = 0.0
        self._trade_log = []
        self._synced = False
        print("🔄 仓位管理器已重置")
    
    def set_data_fetcher(self, data_fetcher: 'AlpacaDataFetcher'):
        """
        设置数据获取器（用于 API 同步）。
        
        Args:
            data_fetcher: 数据获取器实例
        """
        self.data_fetcher = data_fetcher
        print("✅ 已设置数据获取器，可使用 sync_from_api() 同步仓位")