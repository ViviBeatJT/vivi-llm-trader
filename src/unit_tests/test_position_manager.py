# tests/test_position_manager.py

import unittest
from unittest.mock import MagicMock
from datetime import datetime, timezone
import pandas as pd

# 导入 PositionManager。注意：PositionManager 依赖于 BaseExecutor 的子类。
# 为了测试 PositionManager，我们需要一个实现了 execute_trade 方法的 Mock 对象。
from src.manager.position_manager import PositionManager

# ----------------------------------------------------
# 共享配置
# ----------------------------------------------------
FINANCE_PARAMS = {
    'INITIAL_CAPITAL': 100000.0,
    'COMMISSION_RATE': 0.0003,
    'SLIPPAGE_RATE': 0.0001,
    'MIN_LOT_SIZE': 100,
    'MAX_ALLOCATION': 0.2,
    'STAMP_DUTY_RATE': 0.001,
}
TEST_TIME = datetime(2025, 12, 10, 10, 0, 0, tzinfo=timezone.utc)
TEST_PRICE = 100.0

class TestPositionManager(unittest.TestCase):
    """
    测试 PositionManager 的核心逻辑：
    1. 初始化状态。
    2. 账户状态计算 (get_account_status)。
    3. 交易执行后的状态更新 (execute_and_update) - 假设执行器总是成功。
    """

    def setUp(self):
        """为每个测试设置一个模拟执行器和 PositionManager 实例。"""
        
        # 创建一个模拟执行器 (Mock Executor)
        self.mock_executor = MagicMock()
        # 模拟 execute_trade 方法，让它返回一个标准的成功交易结果
        self.mock_executor.execute_trade.return_value = {
            'executed': True,
            'trade_type': 'BUY', # 默认返回 BUY，但在 sell 测试中会覆盖
            'executed_qty': 100.0,
            'executed_price': 101.0,
            'fee': 3.0,
            'log_message': 'Mock trade successful'
        }
        
        # 实例化 PositionManager
        self.pm = PositionManager(executor=self.mock_executor, finance_params=FINANCE_PARAMS)

    def test_initialization(self):
        """测试 PositionManager 初始化状态是否正确。"""
        self.assertEqual(self.pm.cash, FINANCE_PARAMS['INITIAL_CAPITAL'])
        self.assertEqual(self.pm.position, 0.0)
        self.assertEqual(self.pm.avg_cost, 0.0)
        self.assertEqual(len(self.pm.trade_log), 0)

    def test_get_account_status(self):
        """测试账户状态计算是否正确。"""
        self.pm.cash = 50000.0
        self.pm.position = 500.0
        self.pm.avg_cost = 90.0
        current_price = 110.0
        
        status = self.pm.get_account_status(current_price)
        
        expected_market_value = 500.0 * 110.0  # 55000.0
        expected_equity = 50000.0 + 55000.0    # 105000.0
        
        self.assertAlmostEqual(status['market_value'], expected_market_value)
        self.assertAlmostEqual(status['equity'], expected_equity)
        self.assertEqual(status['position'], 500.0)
        self.assertEqual(status['avg_cost'], 90.0)

    def test_successful_buy_update(self):
        """测试买入交易后的资金、仓位和平均成本更新。"""
        
        # 模拟执行器返回结果 (买入 100 股 @ 101.0, 费用 3.0)
        executed_qty_1 = 100.0
        executed_price_1 = 101.0
        fee_1 = 3.0
        self.mock_executor.execute_trade.return_value = {
            'executed': True, 'trade_type': 'BUY',
            'executed_qty': executed_qty_1, 'executed_price': executed_price_1,
            'fee': fee_1, 'log_message': 'Buy 1 successful'
        }
        
        # 第一次买入
        self.pm.execute_and_update(TEST_TIME, 'BUY', TEST_PRICE)
        
        # 第一次买入后的期望值
        initial_cash = FINANCE_PARAMS['INITIAL_CAPITAL']
        total_cost_1 = executed_qty_1 * executed_price_1 + fee_1  # 10103.0
        expected_cash_1 = initial_cash - total_cost_1
        
        self.assertAlmostEqual(self.pm.cash, expected_cash_1)
        self.assertAlmostEqual(self.pm.position, executed_qty_1)
        self.assertAlmostEqual(self.pm.avg_cost, executed_price_1)
        self.assertEqual(len(self.pm.trade_log), 1)

        # 第二次买入：50 股 @ 102.0, 费用 2.0
        executed_qty_2 = 50.0
        executed_price_2 = 102.0
        fee_2 = 2.0
        self.mock_executor.execute_trade.return_value = {
            'executed': True, 'trade_type': 'BUY',
            'executed_qty': executed_qty_2, 'executed_price': executed_price_2,
            'fee': fee_2, 'log_message': 'Buy 2 successful'
        }
        
        self.pm.execute_and_update(TEST_TIME, 'BUY', TEST_PRICE)
        
        # 第二次买入后的期望值
        cost_2 = executed_qty_2 * executed_price_2 + fee_2 # 5102.0
        expected_cash_2 = expected_cash_1 - cost_2
        expected_position_2 = executed_qty_1 + executed_qty_2 # 150.0
        
        # 平均成本计算: (100 * 101.0 + 50 * 102.0) / 150 = 15200 / 150
        expected_avg_cost_2 = (executed_qty_1 * executed_price_1 + executed_qty_2 * executed_price_2) / expected_position_2
        
        self.assertAlmostEqual(self.pm.cash, expected_cash_2)
        self.assertAlmostEqual(self.pm.position, expected_position_2)
        self.assertAlmostEqual(self.pm.avg_cost, expected_avg_cost_2)
        self.assertEqual(len(self.pm.trade_log), 2)


    def test_successful_sell_update_partial(self):
        """测试部分卖出后的 P&L、资金和仓位更新。"""
        
        # 预设初始状态
        self.pm.cash = 90000.0
        self.pm.position = 200.0
        self.pm.avg_cost = 100.0 # 成本 200 * 100 = 20000.0
        
        # 模拟执行器返回结果 (卖出 150 股 @ 110.0, 费用 5.0)
        executed_qty = 150.0
        executed_price = 110.0
        fee = 5.0
        self.mock_executor.execute_trade.return_value = {
            'executed': True, 'trade_type': 'SELL',
            'executed_qty': executed_qty, 'executed_price': executed_price,
            'fee': fee, 'log_message': 'Sell successful'
        }
        
        self.pm.execute_and_update(TEST_TIME, 'SELL', TEST_PRICE)
        
        # 期望值计算
        capital_cost = executed_qty * self.pm.avg_cost       # 150 * 100.0 = 15000.0
        income_before_fee = executed_qty * executed_price    # 150 * 110.0 = 16500.0
        expected_net_pnl = income_before_fee - fee - capital_cost # 16500.0 - 5.0 - 15000.0 = 1495.0
        
        # 现金更新: cash += (收入 - 费用)
        expected_cash = 90000.0 + (16500.0 - 5.0) # 106495.0
        
        # 仓位更新
        expected_position = 200.0 - 150.0 # 50.0
        
        self.assertAlmostEqual(self.pm.cash, expected_cash)
        self.assertAlmostEqual(self.pm.position, expected_position)
        self.assertAlmostEqual(self.pm.avg_cost, 100.0) # 剩余仓位的成本不变
        self.assertAlmostEqual(self.pm.trade_log[0]['net_pnl'], expected_net_pnl)
        self.assertEqual(len(self.pm.trade_log), 1)

    def test_successful_sell_to_zero(self):
        """测试完全清仓后，平均成本是否归零。"""
        
        # 预设初始状态
        self.pm.cash = 90000.0
        self.pm.position = 100.0
        self.pm.avg_cost = 100.0
        
        # 模拟执行器返回结果 (卖出全部 100 股)
        executed_qty = 100.0
        executed_price = 105.0
        fee = 2.0
        self.mock_executor.execute_trade.return_value = {
            'executed': True, 'trade_type': 'SELL',
            'executed_qty': executed_qty, 'executed_price': executed_price,
            'fee': fee, 'log_message': 'Sell all successful'
        }
        
        self.pm.execute_and_update(TEST_TIME, 'SELL', TEST_PRICE)
        
        # P&L = (100 * 105.0) - 2.0 - (100 * 100.0) = 498.0
        expected_net_pnl = (executed_qty * executed_price) - fee - (executed_qty * 100.0)
        
        self.assertAlmostEqual(self.pm.position, 0.0)
        self.assertAlmostEqual(self.pm.avg_cost, 0.0) # 核心检查点
        self.assertAlmostEqual(self.pm.trade_log[0]['net_pnl'], expected_net_pnl)
        
    def test_failed_execution_no_update(self):
        """测试执行器返回失败时，PositionManager 状态不变。"""
        
        # 预设初始状态
        self.pm.cash = 90000.0
        self.pm.position = 100.0
        self.pm.avg_cost = 100.0
        
        initial_status = self.pm.get_account_status(TEST_PRICE)
        
        # 模拟执行器返回失败
        self.mock_executor.execute_trade.return_value = {
            'executed': False,
            'trade_type': 'N/A',
            'executed_qty': 0.0,
            'executed_price': 0.0,
            'fee': 0.0,
            'log_message': 'Cash insufficient'
        }
        
        success = self.pm.execute_and_update(TEST_TIME, 'BUY', TEST_PRICE)
        
        final_status = self.pm.get_account_status(TEST_PRICE)
        
        self.assertFalse(success)
        self.assertEqual(initial_status['cash'], final_status['cash'])
        self.assertEqual(initial_status['position'], final_status['position'])
        self.assertEqual(initial_status['avg_cost'], final_status['avg_cost'])
        self.assertEqual(len(self.pm.trade_log), 0)

    def test_get_trade_log(self):
        """测试 get_trade_log 返回 DataFrame。"""
        self.pm.trade_log.append({'time': TEST_TIME, 'type': 'BUY', 'qty': 100.0})
        self.pm.trade_log.append({'time': TEST_TIME, 'type': 'SELL', 'qty': 50.0})
        
        df = self.pm.get_trade_log()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertTrue('type' in df.columns)

if __name__ == '__main__':
    unittest.main()