# src/manager/position_manager.py

from datetime import datetime, timezone
from typing import Dict, Any, Optional, TYPE_CHECKING, Literal
import pandas as pd

if TYPE_CHECKING:
    from src.data_fetcher.alpaca_data_fetcher import AlpacaDataFetcher


class PositionManager:
    """
    仓位管理器 - 管理交易仓位、现金和交易记录。
    
    支持两种模式：
    1. 本地模拟模式：使用 SimulationExecutor，完全本地计算
    2. API 模式：使用 AlpacaExecutor，可从 API 同步仓位状态
    
    仓位类型：
    - 多仓 (long): position > 0
    - 空仓 (short): position < 0
    - 无仓位 (flat): position == 0
    
    信号处理逻辑：
    - BUY: 开多仓或平空仓
    - SELL: 平多仓
    - SHORT: 开空仓或平多仓
    - COVER: 平空仓
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
        self._position = 0.0  # 持仓数量（正数=多仓，负数=空仓）
        self._avg_cost = 0.0  # 平均成本
        
        # 交易记录
        self._trade_log = []
        
        # 同步标志
        self._synced = False
    
    @property
    def position_side(self) -> Literal['long', 'short', 'flat']:
        """获取当前仓位方向。"""
        if self._position > 0:
            return 'long'
        elif self._position < 0:
            return 'short'
        else:
            return 'flat'
    
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
            
            side_str = {"long": "多仓", "short": "空仓", "flat": "空仓位"}[self.position_side]
            print(f"✅ 仓位同步成功:")
            print(f"   现金: ${self._cash:,.2f}")
            print(f"   持仓: {abs(self._position):.0f} 股 ({side_str})")
            if self._position != 0:
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
            dict: 账户状态
        """
        # 计算市值（空仓时为负）
        market_value = self._position * current_price
        equity = self._cash + market_value
        
        unrealized_pnl = 0.0
        if self._position != 0 and self._avg_cost > 0:
            if self._position > 0:  # 多仓
                unrealized_pnl = (current_price - self._avg_cost) * self._position
            else:  # 空仓
                unrealized_pnl = (self._avg_cost - current_price) * abs(self._position)
        
        return {
            'cash': self._cash,
            'position': self._position,
            'position_side': self.position_side,
            'avg_cost': self._avg_cost,
            'market_value': market_value,
            'equity': equity,
            'unrealized_pnl': unrealized_pnl,
            'synced': self._synced
        }
    
    def _translate_signal(self, signal: str) -> Optional[str]:
        """
        根据当前仓位状态，将策略信号转换为实际执行动作。
        
        Args:
            signal: 原始信号 (BUY, SELL, SHORT, COVER, HOLD)
            
        Returns:
            str or None: 实际执行动作 (BUY, SELL, SHORT, COVER) 或 None（无需操作）
        """
        side = self.position_side
        
        if signal == 'HOLD':
            return None
        
        elif signal == 'BUY':
            if side == 'flat':
                return 'BUY'  # 开多
            elif side == 'short':
                return 'COVER'  # 平空
            else:  # long
                return None  # 已有多仓，不加仓
        
        elif signal == 'SELL':
            if side == 'long':
                return 'SELL'  # 平多
            else:
                return None  # 无多仓可平
        
        elif signal == 'SHORT':
            if side == 'flat':
                return 'SHORT'  # 开空
            elif side == 'long':
                return 'SELL'  # 先平多（可选择是否同时开空）
            else:  # short
                return None  # 已有空仓，不加仓
        
        elif signal == 'COVER':
            if side == 'short':
                return 'COVER'  # 平空
            else:
                return None  # 无空仓可平
        
        return None
    
    def execute_and_update(self, 
                          timestamp: datetime, 
                          signal: str, 
                          current_price: float,
                          ticker: str = "UNKNOWN") -> bool:
        """
        执行交易并更新仓位。
        
        Args:
            timestamp: 交易时间
            signal: 交易信号 ('BUY', 'SELL', 'SHORT', 'COVER')
            current_price: 当前价格
            ticker: 股票代码
            
        Returns:
            bool: 是否执行成功
        """
        # 翻译信号
        action = self._translate_signal(signal)
        
        if action is None:
            print(f"⚪ 信号 {signal} 在当前仓位状态下无需操作 (仓位: {self.position_side})")
            return False
        
        # 计算交易数量
        qty = self._calculate_trade_qty(action, current_price)
        
        if qty <= 0:
            print(f"⚠️ 计算交易数量为 0，跳过交易")
            return False
        
        # 执行交易
        try:
            result = self.executor.execute(
                signal=action,
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
            
            self._update_position(action, executed_qty, executed_price, fee)
            
            # 记录交易
            self._record_trade(
                timestamp=timestamp,
                signal=action,
                qty=executed_qty,
                price=executed_price,
                fee=fee,
                ticker=ticker
            )
            
            return True
            
        except Exception as e:
            print(f"❌ 交易执行异常: {e}")
            return False
    
    def _calculate_trade_qty(self, action: str, current_price: float) -> int:
        """
        计算交易数量。
        
        Args:
            action: 交易动作 (BUY, SELL, SHORT, COVER)
            current_price: 当前价格
            
        Returns:
            int: 交易数量
        """
        min_lot_size = self.finance_params.get('MIN_LOT_SIZE', 10)
        max_allocation = self.finance_params.get('MAX_ALLOCATION', 0.2)
        
        if action == "BUY":
            # 计算可用资金开多
            available_cash = self._cash * max_allocation
            
            # 考虑佣金和滑点
            commission_rate = self.finance_params.get('COMMISSION_RATE', 0.0003)
            slippage_rate = self.finance_params.get('SLIPPAGE_RATE', 0.0001)
            effective_price = current_price * (1 + commission_rate + slippage_rate)
            
            max_qty = int(available_cash / effective_price)
            qty = (max_qty // min_lot_size) * min_lot_size
            
            return max(qty, 0)
            
        elif action == "SELL":
            # 平掉所有多仓
            qty = int(self._position)
            qty = (qty // min_lot_size) * min_lot_size
            return max(qty, 0)
        
        elif action == "SHORT":
            # 计算可用资金开空（需要保证金）
            available_cash = self._cash * max_allocation
            
            commission_rate = self.finance_params.get('COMMISSION_RATE', 0.0003)
            slippage_rate = self.finance_params.get('SLIPPAGE_RATE', 0.0001)
            # 做空需要保证金，假设 50% 保证金要求
            margin_requirement = 0.5
            effective_price = current_price * margin_requirement * (1 + commission_rate + slippage_rate)
            
            max_qty = int(available_cash / effective_price)
            qty = (max_qty // min_lot_size) * min_lot_size
            
            return max(qty, 0)
        
        elif action == "COVER":
            # 平掉所有空仓
            qty = int(abs(self._position))
            qty = (qty // min_lot_size) * min_lot_size
            return max(qty, 0)
        
        return 0
    
    def _update_position(self, action: str, qty: int, price: float, fee: float):
        """
        更新仓位状态。
        
        Args:
            action: 交易动作 (BUY, SELL, SHORT, COVER)
            qty: 交易数量
            price: 成交价格
            fee: 交易费用
        """
        if action == "BUY":
            # 买入开多：增加持仓，减少现金
            total_cost = qty * price + fee
            
            if self._position > 0:
                # 已有多仓，计算加权平均成本
                total_value = self._position * self._avg_cost + qty * price
                self._avg_cost = total_value / (self._position + qty)
            else:
                self._avg_cost = price
            
            self._position += qty
            self._cash -= total_cost
            
        elif action == "SELL":
            # 卖出平多：减少持仓，增加现金
            proceeds = qty * price - fee
            
            self._position -= qty
            self._cash += proceeds
            
            if self._position <= 0:
                self._position = 0
                self._avg_cost = 0.0
        
        elif action == "SHORT":
            # 卖空开空：持仓变负，收到卖出资金（但需要保证金）
            proceeds = qty * price - fee
            
            if self._position < 0:
                # 已有空仓，计算加权平均成本
                total_value = abs(self._position) * self._avg_cost + qty * price
                self._avg_cost = total_value / (abs(self._position) + qty)
            else:
                self._avg_cost = price
            
            self._position -= qty  # 变为负数
            self._cash += proceeds  # 收到卖出资金
            
        elif action == "COVER":
            # 买入平空：持仓归零，支付买入成本
            total_cost = qty * price + fee
            
            self._position += qty  # 从负数向0移动
            self._cash -= total_cost
            
            if self._position >= 0:
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
        # 计算本次交易盈亏
        net_pnl = 0.0
        if signal == "SELL" and self._avg_cost > 0:
            # 平多盈亏
            net_pnl = (price - self._avg_cost) * qty - fee
        elif signal == "COVER" and self._avg_cost > 0:
            # 平空盈亏
            net_pnl = (self._avg_cost - price) * qty - fee
        
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
        """获取交易记录。"""
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
        """设置数据获取器。"""
        self.data_fetcher = data_fetcher
        print("✅ 已设置数据获取器，可使用 sync_from_api() 同步仓位")