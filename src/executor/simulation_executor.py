# src/executor/simulation_executor.py

from typing import Dict, Any
from datetime import datetime, timezone


class SimulationExecutor:
    """
    模拟交易执行器 - 用于本地回测，不连接真实 API。
    
    支持的交易动作：
    - BUY: 买入开多
    - SELL: 卖出平多
    - SHORT: 卖空开空
    - COVER: 买入平空
    """

    def __init__(self, finance_params: Dict[str, Any]):
        """
        初始化模拟执行器。
        
        Args:
            finance_params: 财务参数字典，包含：
                - COMMISSION_RATE: 佣金率
                - SLIPPAGE_RATE: 滑点率
        """
        self.commission_rate = finance_params.get('COMMISSION_RATE', 0.0003)
        self.slippage_rate = finance_params.get('SLIPPAGE_RATE', 0.0001)
        
        print(f"🔧 SimulationExecutor 初始化: 佣金={self.commission_rate*100:.2f}%, 滑点={self.slippage_rate*100:.2f}%")
    
    def execute(self, 
               signal: str, 
               qty: int, 
               price: float, 
               ticker: str = "UNKNOWN") -> Dict[str, Any]:
        """
        执行模拟交易。
        
        Args:
            signal: 交易信号 (BUY, SELL, SHORT, COVER)
            qty: 交易数量
            price: 请求价格
            ticker: 股票代码
            
        Returns:
            dict: 执行结果，包含：
                - success: 是否成功
                - signal: 执行的信号
                - qty: 实际成交数量
                - price: 实际成交价格（考虑滑点）
                - fee: 交易费用
                - error: 错误信息（如果有）
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
        
        if price <= 0:
            return {
                'success': False,
                'error': f'Invalid price: {price}'
            }
        
        # 计算滑点
        if signal in ['BUY', 'COVER']:
            # 买入时价格上滑
            executed_price = price * (1 + self.slippage_rate)
        else:  # SELL, SHORT
            # 卖出时价格下滑
            executed_price = price * (1 - self.slippage_rate)
        
        # 计算佣金
        fee = qty * executed_price * self.commission_rate
        
        timestamp_str = datetime.now(timezone.utc).strftime('%H:%M:%S')
        action_emoji = {
            'BUY': '🟢 买入开多',
            'SELL': '🔴 卖出平多',
            'SHORT': '🔻 卖空开空',
            'COVER': '🔺 买入平空'
        }.get(signal, signal)
        
        print(f"   💱 [{timestamp_str}] {action_emoji} {ticker}: {qty} 股 @ ${executed_price:.2f} (费用: ${fee:.2f})")
        
        return {
            'success': True,
            'signal': signal,
            'ticker': ticker,
            'qty': qty,
            'price': executed_price,
            'fee': fee,
            'timestamp': datetime.now(timezone.utc)
        }