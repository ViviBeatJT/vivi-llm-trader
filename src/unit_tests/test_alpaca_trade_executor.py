import unittest
from unittest.mock import MagicMock, patch, ANY
from datetime import datetime
import os
import sys

# 设置路径以确保能找到模块，如果你的项目结构需要
# 假设源文件位于 'src/executor/'，测试文件位于 'src/unit_tests/'。
# 修正路径：从 'src/unit_tests' 向上退一级 (..) 到 'src'，然后进入 'executor' 目录。
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'executor')))

# 导入要测试的类 - 使用模块名直接导入
from alpaca_trade_executor import AlpacaExecutor
# 导入 Alpaca 相关的类进行模拟
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError


# Mock Alpaca 客户端和相关对象
class MockOrder:
    """模拟 Alpaca Order 对象，只需要 id 和 status 属性。"""
    def __init__(self, order_id="test_order_id", status="accepted"):
        self.id = order_id
        self.status = MagicMock(value=status)

class MockAccount:
    """模拟 Alpaca Account 对象，只需要 equity 属性。"""
    def __init__(self, equity=100000.0):
        self.equity = str(equity) # equity 在 Alpaca 模型中通常是字符串

# 使用 patch 装饰器模拟外部依赖
# 注意：patch 的路径也需要对应更新回模块名
@patch('alpaca_trade_executor.load_dotenv', MagicMock())
@patch('alpaca_trade_executor.os.getenv', MagicMock(return_value='dummy_key'))
@patch('alpaca_trade_executor.TradingClient')
class TestAlpacaExecutor(unittest.TestCase):
    
    TICKER = "TSLA"
    
    def setUp(self):
        """初始化通用的测试数据。"""
        # 初始化 AlpacaExecutor 实例。TradingClient 已经被 patch 掉了。
        self.executor = AlpacaExecutor(paper=True, max_allocation_rate=0.2)
        self.test_timestamp = datetime(2023, 1, 1, 9, 30)
        # 获取 mocked client 实例，供测试方法使用
        self.mock_client = self.executor.trading_client

    def test_initialization(self, MockTradingClient):
        """测试 AlpacaExecutor 是否正确初始化 TradingClient。"""
        # MockTradingClient 是通过装饰器传入的
        MockTradingClient.assert_called_once()
        # 验证是否以 paper=True 初始化
        self.assertTrue(MockTradingClient.call_args[1]['paper'])
        self.assertEqual(self.executor.MAX_ALLOCATION_RATE, 0.2)
        
    # --- BUY 交易测试 ---

    def test_buy_successful_execution(self, MockTradingClient):
        """测试成功的买入交易，验证数量计算和 API 调用。"""
        
        # 直接使用 setUp 中设置的 mocked client 实例
        mock_client = self.mock_client
        
        # 1. 模拟 get_account 响应
        current_equity = 100000.0
        current_cash = 50000.0
        
        # 使用 patch.object 确保 get_account 和 submit_order 被正确 Mock
        with patch.object(mock_client, 'get_account', return_value=MockAccount(equity=current_equity)) as mock_get_account:
            # FIX: 使用 patch.object 明确 Mock submit_order
            with patch.object(mock_client, 'submit_order', return_value=MockOrder(order_id="TEST_BUY_ORDER")) as mock_submit_order:
                
                current_price = 100.0
                
                # 3. 预期计算
                # MAX_ALLOCATION_RATE = 0.2
                # Capital to use: min(50000, 100000 * 0.2) = 20000.0
                # Qty: floor(20000.0 / 100.0) = 200.0
                expected_qty = 200.0

                result = self.executor.execute_trade(
                    timestamp=self.test_timestamp,
                    signal='BUY',
                    current_price=current_price,
                    current_position=0.0,
                    current_cash=current_cash,
                    avg_cost=0.0
                )
                
                # 4. 验证 API 调用
                mock_get_account.assert_called_once() # 使用 patch 对象的断言
                mock_submit_order.assert_called_once()
                
                # 验证 submit_order 是否使用了正确的参数
                call_args = mock_submit_order.call_args[0][0]
                self.assertEqual(call_args.symbol, self.TICKER)
                self.assertEqual(call_args.qty, expected_qty)
                self.assertEqual(call_args.side, OrderSide.BUY)
                
                # 5. 验证返回结果
                self.assertTrue(result['executed'])
                self.assertEqual(result['trade_type'], 'BUY')
                self.assertEqual(result['executed_qty'], expected_qty)
                # AlpacaExecutor 返回的成交价是 current_price (简化处理)
                self.assertEqual(result['executed_price'], current_price) 
                self.assertEqual(result['fee'], 0.0)

    def test_buy_failure_low_capital(self, MockTradingClient):
        """测试因可用资金不足导致的买入失败。"""
        mock_client = self.mock_client
        
        # 模拟账户资产很低
        current_equity = 500.0
        current_cash = 10.0
        
        # 使用 patch.object 确保 get_account 被正确 Mock
        with patch.object(mock_client, 'get_account', return_value=MockAccount(equity=current_equity)) as mock_get_account:
        
            current_price = 100.0
            
            # 预期计算：
            # Capital to use: min(10.0, 500.0 * 0.2) = 10.0
            # Qty: floor(10.0 / 100.0) = 0.0
            
            result = self.executor.execute_trade(
                timestamp=self.test_timestamp,
                signal='BUY',
                current_price=current_price,
                current_position=0.0,
                current_cash=current_cash,
                avg_cost=0.0
            )
            
            # 验证结果
            self.assertFalse(result['executed'])
            self.assertIn("资金不足", result['log_message'])
            mock_get_account.assert_called_once()
            mock_client.submit_order.assert_not_called()
        
    def test_buy_failure_alpaca_api_error(self, MockTradingClient):
        """测试 Alpaca API 在买入时抛出异常。"""
        mock_client = self.mock_client
        
        # 1. 模拟 get_account 响应
        # 使用 patch.object 确保 get_account 被正确 Mock
        with patch.object(mock_client, 'get_account', return_value=MockAccount(equity=100000.0)) as mock_get_account:
            
            # FIX: 使用 patch.object 明确 Mock submit_order 的 side_effect
            with patch.object(mock_client, 'submit_order', side_effect=APIError("Rate limit exceeded")) as mock_submit_order:
            
                result = self.executor.execute_trade(
                    timestamp=self.test_timestamp,
                    signal='BUY',
                    current_price=100.0,
                    current_position=0.0,
                    current_cash=50000.0,
                    avg_cost=0.0
                )
            
            # 验证结果
            self.assertFalse(result['executed'])
            self.assertIn("API 错误", result['log_message'])
            # 验证 get_account 被调用 (用于获取账户权益)
            mock_get_account.assert_called_once()
            # 验证 submit_order 被调用 (并抛出异常)
            mock_submit_order.assert_called_once()

        
    # --- SELL 交易测试 ---

    def test_sell_successful_execution(self, MockTradingClient):
        """测试成功的卖出（平仓）交易。"""
        mock_client = self.mock_client
        
        # 使用 patch.object 确保 close_position 被正确 Mock
        with patch.object(mock_client, 'close_position', return_value=MockOrder(order_id="TEST_SELL_ORDER")) as mock_close_position:
            
            current_position = 300.0 # 持仓 300 股
            
            result = self.executor.execute_trade(
                timestamp=self.test_timestamp,
                signal='SELL',
                current_price=100.0,
                current_position=current_position,
                current_cash=50000.0,
                avg_cost=95.0
            )
            
            # 验证 API 调用
            mock_close_position.assert_called_once()
            # 验证 close_position 是否使用了正确的 ticker
            call_args = mock_close_position.call_args[0][0]
            self.assertEqual(call_args.symbol, self.TICKER)
            
            # 验证返回结果
            self.assertTrue(result['executed'])
            self.assertEqual(result['trade_type'], 'SELL')
            # 预期卖出全部
            self.assertEqual(result['executed_qty'], current_position) 
            self.assertEqual(result['executed_price'], 100.0 * (1 - 0.0)) # 假设滑点为 0.0，价格简化处理
            self.assertEqual(result['fee'], 0.0) # AlpacaExecutor 简化，fee为0
        
    def test_sell_failure_no_position(self, MockTradingClient):
        """测试在没有持仓时，卖出信号应该被拒绝。"""
        mock_client = self.mock_client
        
        # FIX: 使用 patch.object 确保 close_position 是一个可断言的 Mock 对象
        with patch.object(mock_client, 'close_position') as mock_close_position:
        
            result = self.executor.execute_trade(
                timestamp=self.test_timestamp,
                signal='SELL',
                current_price=100.0,
                current_position=0.0,
                current_cash=50000.0,
                avg_cost=0.0
            )
            
            self.assertFalse(result['executed'])
            self.assertIn("无仓位可卖", result['log_message'])
            mock_close_position.assert_not_called()

    def test_sell_failure_alpaca_api_error(self, MockTradingClient):
        """测试 Alpaca API 在卖出时抛出异常。"""
        mock_client = self.mock_client
        
        # 使用 patch.object 确保 close_position 被正确 Mock
        with patch.object(mock_client, 'close_position', side_effect=APIError("Invalid position to close")) as mock_close_position:
        
            result = self.executor.execute_trade(
                timestamp=self.test_timestamp,
                signal='SELL',
                current_price=100.0,
                current_position=100.0,
                current_cash=50000.0,
                avg_cost=95.0
            )
            
            # 验证结果
            self.assertFalse(result['executed'])
            self.assertIn("API 错误", result['log_message'])
            mock_close_position.assert_called_once()


if __name__ == '__main__':
    unittest.main()