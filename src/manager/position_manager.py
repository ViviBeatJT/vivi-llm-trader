# src/manager/position_manager.py

"""
仓位管理器 - Position Manager

管理交易仓位、现金和交易记录。
邮件通知功能已移至 TradingEngine。

支持两种模式：
1. 本地模拟模式：使用 SimulationExecutor，完全本地计算
2. API 模式：使用 AlpacaExecutor，可从 API 同步仓位状态
"""

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
    """

    def __init__(self,
                 executor,
                 finance_params: Dict[str, Any],
                 data_fetcher: Optional['AlpacaDataFetcher'] = None):
        """
        初始化仓位管理器。

        Args:
            executor: 交易执行器（SimulationExecutor 或 AlpacaExecutor）
            finance_params: 财务参数字典
            data_fetcher: 数据获取器（可选）
        """
        self.executor = executor
        self.finance_params = finance_params
        self.data_fetcher = data_fetcher

        # 本地状态
        self._cash = finance_params.get('INITIAL_CAPITAL', 100000.0)
        self._position = 0.0
        self._avg_cost = 0.0

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
        """从 API 同步仓位状态。"""
        if not self.data_fetcher:
            print("⚠️ 未配置 data_fetcher，无法从 API 同步")
            return False

        try:
            status = self.data_fetcher.sync_position_status(ticker)

            if not status:
                print("❌ 从 API 同步仓位失败")
                return False

            self._cash = status.get('cash', self._cash)
            self._position = status.get('position', 0.0)
            self._avg_cost = status.get('avg_cost', 0.0)
            self._synced = True

            side_str = {"long": "多仓", "short": "空仓",
                        "flat": "空仓位"}[self.position_side]
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
        """获取当前账户状态。"""
        market_value = self._position * current_price
        equity = self._cash + market_value

        unrealized_pnl = 0.0
        if self._position != 0 and self._avg_cost > 0:
            if self._position > 0:
                unrealized_pnl = (
                    current_price - self._avg_cost) * self._position
            else:
                unrealized_pnl = (
                    self._avg_cost - current_price) * abs(self._position)

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
        """根据当前仓位状态，将策略信号转换为实际执行动作。"""
        side = self.position_side

        if signal == 'HOLD':
            return None

        elif signal == 'BUY':
            if side == 'flat':
                return 'BUY'
            elif side == 'short':
                return 'COVER'
            else:
                return None

        elif signal == 'SELL':
            if side == 'long':
                return 'SELL'
            else:
                return None

        elif signal == 'SHORT':
            if side == 'flat':
                return 'SHORT'
            elif side == 'long':
                return 'SELL'
            else:
                return None

        elif signal == 'COVER':
            if side == 'short':
                return 'COVER'
            else:
                return None

        return None

    def execute_and_update(self,
                           timestamp: datetime,
                           signal: str,
                           current_price: float,
                           ticker: str = "UNKNOWN") -> Dict[str, Any]:
        """
        执行交易并更新仓位。

        Returns:
            Dict with trade result:
            - success: bool
            - action: str (actual action taken)
            - qty: int
            - price: float
            - fee: float
            - pnl: float (for closing trades)
            - pnl_pct: float
            - error: str (if failed)
        """
        action = self._translate_signal(signal)

        if action is None:
            return {
                'success': False,
                'error': f"信号 {signal} 在当前仓位状态下无需操作 (仓位: {self.position_side})"
            }

        qty = self._calculate_trade_qty(action, current_price)

        if qty <= 0:
            return {
                'success': False,
                'error': "计算交易数量为 0"
            }

        # 记录交易前的状态（用于计算盈亏）
        pre_trade_avg_cost = self._avg_cost
        pre_trade_position = self._position

        try:
            result = self.executor.execute(
                signal=action,
                qty=qty,
                price=current_price,
                ticker=ticker
            )

            if not result.get('success', False):
                return {
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                }

            executed_qty = result.get('qty', qty)
            executed_price = result.get('price', current_price)
            fee = result.get('fee', 0.0)

            self._update_position(action, executed_qty, executed_price, fee)

            # 计算盈亏
            pnl = 0.0
            pnl_pct = 0.0
            if action in ['SELL', 'COVER'] and pre_trade_avg_cost > 0:
                if action == 'SELL':
                    pnl = (executed_price - pre_trade_avg_cost) * \
                        executed_qty - fee
                else:  # COVER
                    pnl = (pre_trade_avg_cost - executed_price) * \
                        executed_qty - fee
                pnl_pct = pnl / (pre_trade_avg_cost * executed_qty) * \
                    100 if pre_trade_avg_cost > 0 else 0

            # 记录交易
            self._record_trade(
                timestamp=timestamp,
                signal=action,
                qty=executed_qty,
                price=executed_price,
                fee=fee,
                ticker=ticker,
                pnl=pnl
            )

            return {
                'success': True,
                'action': action,
                'qty': executed_qty,
                'price': executed_price,
                'fee': fee,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'pre_avg_cost': pre_trade_avg_cost,
                'pre_position': pre_trade_position
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

    def _calculate_trade_qty(self, action: str, current_price: float) -> int:
        """计算交易数量。"""
        min_lot_size = self.finance_params.get('MIN_LOT_SIZE', 10)
        max_allocation = self.finance_params.get('MAX_ALLOCATION', 0.2)

        if action == "BUY":
            available_cash = self._cash * max_allocation
            commission_rate = self.finance_params.get(
                'COMMISSION_RATE', 0.0003)
            slippage_rate = self.finance_params.get('SLIPPAGE_RATE', 0.0001)
            effective_price = current_price * \
                (1 + commission_rate + slippage_rate)
            max_qty = int(available_cash / effective_price)
            qty = (max_qty // min_lot_size) * min_lot_size
            return max(qty, 0)

        elif action == "SELL":
            qty = int(self._position)
            qty = (qty // min_lot_size) * min_lot_size
            return max(qty, 0)

        elif action == "SHORT":
            available_cash = self._cash * max_allocation
            commission_rate = self.finance_params.get(
                'COMMISSION_RATE', 0.0003)
            slippage_rate = self.finance_params.get('SLIPPAGE_RATE', 0.0001)
            margin_requirement = 0.5
            effective_price = current_price * margin_requirement * \
                (1 + commission_rate + slippage_rate)
            max_qty = int(available_cash / effective_price)
            qty = (max_qty // min_lot_size) * min_lot_size
            return max(qty, 0)

        elif action == "COVER":
            qty = int(abs(self._position))
            qty = (qty // min_lot_size) * min_lot_size
            return max(qty, 0)

        return 0

    def _update_position(self, action: str, qty: int, price: float, fee: float):
        """更新仓位状态。"""
        if action == "BUY":
            total_cost = qty * price + fee
            if self._position > 0:
                total_value = self._position * self._avg_cost + qty * price
                self._avg_cost = total_value / (self._position + qty)
            else:
                self._avg_cost = price
            self._position += qty
            self._cash -= total_cost

        elif action == "SELL":
            proceeds = qty * price - fee
            self._position -= qty
            self._cash += proceeds
            if self._position <= 0:
                self._position = 0
                self._avg_cost = 0.0

        elif action == "SHORT":
            proceeds = qty * price - fee
            if self._position < 0:
                total_value = abs(self._position) * \
                    self._avg_cost + qty * price
                self._avg_cost = total_value / (abs(self._position) + qty)
            else:
                self._avg_cost = price
            self._position -= qty
            self._cash += proceeds

        elif action == "COVER":
            total_cost = qty * price + fee
            self._position += qty
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
                      ticker: str,
                      pnl: float = 0.0):
        """记录交易。"""
        trade_record = {
            'time': timestamp,
            'ticker': ticker,
            'type': signal,
            'qty': qty,
            'price': price,
            'fee': fee,
            'net_pnl': pnl,
            'cash_after': self._cash,
            'position_after': self._position
        }

        self._trade_log.append(trade_record)

    def get_trade_log(self) -> pd.DataFrame:
        """获取交易记录。"""
        if not self._trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self._trade_log)

    def get_last_trade(self) -> Optional[Dict[str, Any]]:
        """获取最后一笔交易记录。"""
        if not self._trade_log:
            return None
        return self._trade_log[-1]

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
