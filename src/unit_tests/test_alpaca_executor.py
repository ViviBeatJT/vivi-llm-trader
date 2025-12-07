# tests/test_alpaca_executor.py

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
import os

# 导入 AlpacaExecutor 及其依赖
# 注意：确保 AlpacaExecutor 所在的 src.executor 路径能被识别
from src.executor.alpaca_trade_executor import (
    AlpacaExecutor, 
    TradingClient, 
    APIError, 
    OrderSide, 
    TimeInForce,
    MarketOrderRequest,
    ClosePositionRequest,
    Position
)

# ----------------------------------------------------
# 共享配置
# ----------------------------------------------------
TEST_TIME = datetime(2025, 12, 10, 10, 0, 0, tzinfo=timezone.utc)
TEST_PRICE = 100.0
TEST_SYMBOL = 'TSLA' # 假设我们交易 TSLA
MAX_ALLOCATION_RATE = 0.2

class TestAlpacaExecutor(unittest.TestCase):
    """
    测试 AlpacaExecutor 的订单提交逻辑 (使用 Mocking 隔离外部 API)。
    通过 patch.start()/stop() 确保 setUp 正确配置 Mock 对象。
    """
    
    # 定义 patchers 作为类属性
    ENV_PATCH = patch('src.executor.alpaca_trade_executor.os.getenv', return_value='DUMMY_KEY')
    CLIENT_PATCH = patch('src.executor.alpaca_trade_executor.TradingClient')

    def setUp(self):
        """设置 AlpacaExecutor 实例并配置 Mock 对象。"""
        
        # 1. 启动 Patchers 并获取 Mock 对象
        self.mock_getenv = self.ENV_PATCH.start()
        # MockTradingClientClass 是被替换的 TradingClient 构造函数
        MockTradingClientClass = self.CLIENT_PATCH.start() 
        
        # 2. 实例化 AlpacaExecutor，它将使用 Mock 的 TradingClient
        self.executor = AlpacaExecutor(paper=True, max_allocation_rate=MAX_ALLOCATION_RATE)
        # 获取 Mock 的 TradingClient 实例
        # self.mock_client 现在是一个 Mock 实例，其方法都是可配置的 Mock 对象
        self.mock_client = self.executor.trading_client 
        
        # 3. 模拟 get_account 返回值 (用于确定可用资金)
        self.mock_account = MagicMock()
        self.mock_account.equity = '100000.0' # 模拟总资产
        self.mock_account.cash = '90000.0'    # 模拟可用现金
        
        # 核心修复：现在 self.mock_client.get_account 保证是一个 Mock 方法，有 return_value 属性
        self.mock_client.get_account.return_value = self.mock_account
        
        # 4. 模拟 AlpacaExecutor 中的私有方法 (获取当前持仓)
        self.executor._get_current_position = MagicMock()
        
        # 5. 初始化 PositionManager 需要的参数
        self.current_cash = 90000.0
        self.current_position = 0.0
        self.avg_cost = 0.0

    def tearDown(self):
        """停止所有 Patchers。"""
        self.CLIENT_PATCH.stop()
        self.ENV_PATCH.stop()

    def test_alpaca_buy_order_calculation(self):
        """测试购买数量的计算逻辑和订单提交参数。"""
        
        # 预设订单提交成功的返回对象
        mock_order = MagicMock()
        mock_order.id = 'test-order-buy-id'
        mock_order.status.value = 'accepted'
        self.mock_client.submit_order.return_value = mock_order
        
        # 预期购买金额：min(90000.0, 100000.0 * 0.2) = 20000.0
        # 预期购买数量：20000.0 / 100.0 = 200.0 股 (四舍五入到 1 股最小单位)
        expected_qty = 200.0

        result = self.executor.execute_trade(
            TEST_TIME, 'BUY', TEST_PRICE, self.current_position, self.current_cash, self.avg_cost
        )
        
        self.assertTrue(result['executed'])
        self.assertAlmostEqual(result['executed_qty'], expected_qty)
        self.assertEqual(result['executed_price'], 0.0) # AlpacaExecutor 默认返回 0，因为成交价未知
        self.assertIn("订单提交成功", result['log_message'])
        
        # 验证 submit_order 是否被正确调用
        self.mock_client.submit_order.assert_called_once()
        # 检查传入的参数是否正确
        args, kwargs = self.mock_client.submit_order.call_args
        order_request = args[0]
        
        self.assertIsInstance(order_request, MarketOrderRequest)
        self.assertEqual(order_request.symbol, TEST_SYMBOL)
        self.assertAlmostEqual(float(order_request.qty), expected_qty)
        self.assertEqual(order_request.side, OrderSide.BUY)
        self.assertEqual(order_request.time_in_force, TimeInForce.DAY)

    def test_alpaca_buy_not_enough_cash(self):
        """测试购买数量不足最小单位时的失败情况。"""
        
        # 模拟账户总资产和现金
        self.mock_account.equity = '1000.0'
        self.mock_account.cash = '50.0'
        
        # 预期购买金额：min(50.0, 1000.0 * 0.2) = 50.0 USD
        # 预期购买数量：50.0 / 100.0 = 0.5 股。四舍五入到 1 股最小单位 (1) -> 数量为 0。
        
        result = self.executor.execute_trade(
            TEST_TIME, 'BUY', TEST_PRICE, self.current_position, self.current_cash, self.avg_cost
        )
        
        self.assertFalse(result['executed'])
        self.assertIn("小于最小交易单位", result['log_message'])
        self.mock_client.submit_order.assert_not_called()
        

    def test_alpaca_sell_success_clearing_position(self):
        """测试成功的 Alpaca 卖出订单提交（平仓）。"""
        
        # 模拟当前持仓
        mock_position = MagicMock(spec=Position)
        mock_position.qty = '50.0'
        mock_position.current_price = '100.50'
        self.executor._get_current_position.return_value = mock_position
        
        # 预设平仓请求成功的返回对象
        mock_order = MagicMock()
        mock_order.id = 'test-order-sell-id'
        mock_order.status.value = 'accepted'
        # 修复：确保 close_position 在 mock_client 上正确设置
        self.mock_client.close_position.return_value = mock_order
        
        # 模拟 PositionManager 中的参数，但 AlpacaExecutor 不会使用它们
        current_position = 50.0
        
        result = self.executor.execute_trade(
            TEST_TIME, 'SELL', TEST_PRICE, current_position, self.current_cash, self.avg_cost
        )
        
        self.assertTrue(result['executed'])
        self.assertAlmostEqual(result['executed_qty'], 50.0) # 预期卖出全部
        self.assertEqual(result['trade_type'], 'SELL')
        self.assertIn("订单提交成功: 平仓", result['log_message'])
        
        # 验证 close_position 是否被正确调用
        self.mock_client.close_position.assert_called_once()
        
        # 检查传入的参数是否正确
        args, kwargs = self.mock_client.close_position.call_args
        close_request = args[0]
        
        self.assertIsInstance(close_request, ClosePositionRequest)
        self.assertEqual(close_request.symbol, TEST_SYMBOL)

    def test_alpaca_sell_no_position(self):
        """测试 Alpaca API 返回无持仓时的失败情况。"""
        
        self.executor._get_current_position.return_value = None # 模拟无持仓
        
        # 模拟 PositionManager 中的参数，但 AlpacaExecutor 不会使用它们
        current_position = 0.0 
        
        result = self.executor.execute_trade(
            TEST_TIME, 'SELL', TEST_PRICE, current_position, self.current_cash, self.avg_cost
        )
        
        self.assertFalse(result['executed'])
        self.assertIn("无执行信号或无仓位可卖", result['log_message'])
        # 修复：如果 self.mock_client.close_position 在 setUp 中未被定义，则需要确保它是一个 Mock
        if hasattr(self.mock_client.close_position, 'assert_not_called'):
            self.mock_client.close_position.assert_not_called()


    def test_alpaca_api_error_handling(self):
        """测试 Alpaca API 错误时的处理。"""
        
        # 模拟 APIError
        # 确保在 mock_client.submit_order 上设置 side_effect
        self.mock_client.submit_order.side_effect = APIError('Mock API Error: Insufficient funds.')
        
        # 确保账户状态已配置 (在 setUp 中完成)

        # 恢复 cash/equity 的 mock，因为 error handling 仍然会调用 get_account
        self.mock_client.get_account.return_value = self.mock_account
        
        result = self.executor.execute_trade(
            TEST_TIME, 'BUY', TEST_PRICE, self.current_position, self.current_cash, self.avg_cost
        )
        
        self.assertFalse(result['executed'])
        self.assertIn("API 错误", result['log_message'])
        # 验证 submit_order 是否被调用（然后抛出异常）
        self.mock_client.submit_order.assert_called_once()


    def test_get_account_status(self):
        """测试 get_account_status 是否正确调用 Alpaca API 并返回结构化数据。"""
        
        # 模拟 Alpaca get_account 的返回值 (覆盖 setUp 中的默认值)
        self.mock_account.equity = '105000.0'
        self.mock_account.cash = '50000.0'
        
        # 模拟 Alpaca list_positions 返回值
        mock_position = MagicMock(spec=Position)
        mock_position.symbol = TEST_SYMBOL
        mock_position.qty = '500.0'
        mock_position.avg_entry_price = '90.0'
        mock_position.market_value = '55000.0' # 500 * 110.0
        
        self.mock_client.get_all_positions.return_value = [mock_position]

        status = self.executor.get_account_status(current_price=110.0) # 当前价格用于计算市值，但 Alpaca 直接提供
        
        # 验证 get_account 被调用
        self.mock_client.get_account.assert_called_once()
        
        # 验证返回结果
        self.assertAlmostEqual(status['equity'], 105000.0)
        self.assertAlmostEqual(status['cash'], 50000.0)
        self.assertAlmostEqual(status['position'], 500.0)
        self.assertAlmostEqual(status['avg_cost'], 90.0)
        self.assertAlmostEqual(status['market_value'], 55000.0)


if __name__ == '__main__':
    unittest.main()