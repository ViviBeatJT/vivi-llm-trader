# src/executor/alpaca_trade_executor.py

import os
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

load_dotenv()


class AlpacaExecutor:
    """
    Alpaca 交易执行器 - 连接 Alpaca API 执行真实/模拟交易。
    
    支持的交易动作：
    - BUY: 买入开多
    - SELL: 卖出平多
    - SHORT: 卖空开空（需要 margin 账户）
    - COVER: 买入平空
    """

    def __init__(self, paper: bool = True, max_allocation_rate: float = 0.2):
        """
        初始化 Alpaca 执行器。
        
        Args:
            paper: 是否使用模拟盘（默认 True）
            max_allocation_rate: 最大仓位比例
        """
        api_key = os.getenv('ALPACA_API_KEY_ID')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key:
            raise ValueError("Alpaca API 密钥未设置")
        
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        self.paper = paper
        self.max_allocation_rate = max_allocation_rate
        
        mode_str = "模拟盘" if paper else "实盘"
        print(f"🔗 AlpacaExecutor 初始化: {mode_str}")
    
    def execute(self, 
               signal: str, 
               qty: int, 
               price: float, 
               ticker: str = "UNKNOWN") -> Dict[str, Any]:
        """
        执行交易。
        
        Args:
            signal: 交易信号 (BUY, SELL, SHORT, COVER)
            qty: 交易数量
            price: 参考价格（市价单不使用，仅作记录）
            ticker: 股票代码
            
        Returns:
            dict: 执行结果
        """
        if signal not in ['BUY', 'SELL', 'SHORT', 'COVER']:
            return {
                'success': False,
                'error': f'Invalid signal: {signal}'
            }
        
        if qty <= 0:
            return {
                'success': False,
                'error': f'Invalid quantity: {qty}'
            }
        
        # 确定订单方向
        if signal in ['BUY', 'COVER']:
            order_side = OrderSide.BUY
        else:  # SELL, SHORT
            order_side = OrderSide.SELL
        
        # 创建市价单
        order_request = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY
        )
        
        timestamp_str = datetime.now(timezone.utc).strftime('%H:%M:%S')
        action_emoji = {
            'BUY': '🟢 买入开多',
            'SELL': '🔴 卖出平多',
            'SHORT': '🔻 卖空开空',
            'COVER': '🔺 买入平空'
        }.get(signal, signal)
        
        try:
            # 提交订单
            order = self.trading_client.submit_order(order_request)
            
            print(f"   💱 [{timestamp_str}] {action_emoji} {ticker}: {qty} 股")
            print(f"      订单ID: {order.id}")
            print(f"      状态: {order.status}")
            
            # 获取成交价格（市价单可能需要等待成交）
            filled_price = float(order.filled_avg_price) if order.filled_avg_price else price
            filled_qty = int(order.filled_qty) if order.filled_qty else qty
            
            # 估算费用（Alpaca 免佣金，但可能有其他费用）
            fee = 0.0
            
            return {
                'success': True,
                'signal': signal,
                'ticker': ticker,
                'qty': filled_qty,
                'price': filled_price,
                'fee': fee,
                'order_id': str(order.id),
                'order_status': str(order.status),
                'timestamp': datetime.now(timezone.utc)
            }
            
        except Exception as e:
            print(f"   ❌ [{timestamp_str}] 订单执行失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_account(self) -> Optional[Dict[str, Any]]:
        """获取账户信息。"""
        try:
            account = self.trading_client.get_account()
            return {
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'portfolio_value': float(account.portfolio_value),
                'shorting_enabled': account.shorting_enabled,
            }
        except Exception as e:
            print(f"❌ 获取账户信息失败: {e}")
            return None
    
    def cancel_all_orders(self, ticker: Optional[str] = None) -> bool:
        """
        取消所有挂单。
        
        Args:
            ticker: 股票代码（可选，如果指定则只取消该股票的订单）
            
        Returns:
            bool: 是否成功
        """
        try:
            if ticker:
                # 获取指定股票的订单并取消
                orders = self.trading_client.get_orders(
                    filter={'symbol': ticker, 'status': 'open'}
                )
                for order in orders:
                    self.trading_client.cancel_order_by_id(order.id)
                print(f"✅ 已取消 {ticker} 的所有挂单")
            else:
                self.trading_client.cancel_orders()
                print("✅ 已取消所有挂单")
            return True
        except Exception as e:
            print(f"❌ 取消订单失败: {e}")
            return False
    
    def close_position(self, ticker: str) -> bool:
        """
        平仓指定股票的所有持仓。
        
        Args:
            ticker: 股票代码
            
        Returns:
            bool: 是否成功
        """
        try:
            self.trading_client.close_position(ticker)
            print(f"✅ 已平仓 {ticker}")
            return True
        except Exception as e:
            if "position does not exist" in str(e).lower():
                print(f"⚠️ {ticker} 无持仓")
                return True
            print(f"❌ 平仓失败: {e}")
            return False
    
    def close_all_positions(self) -> bool:
        """
        平仓所有持仓。
        
        Returns:
            bool: 是否成功
        """
        try:
            self.trading_client.close_all_positions(cancel_orders=True)
            print("✅ 已平仓所有持仓")
            return True
        except Exception as e:
            print(f"❌ 平仓所有持仓失败: {e}")
            return False