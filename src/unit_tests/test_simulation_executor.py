# tests/test_simulation_executor.py

import unittest
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# 导入 SimulationExecutor
from src.executor.simulation_executor import SimulationExecutor

# ----------------------------------------------------
# 共享配置
# ----------------------------------------------------
FINANCE_PARAMS = {
    'INITIAL_CAPITAL': 100000.0,
    'COMMISSION_RATE': 0.0003,
    'SLIPPAGE_RATE': 0.0001,
    'MIN_LOT_SIZE': 100, # 最小交易单位
    'MAX_ALLOCATION': 0.2, # 最大动用资产比例
    'STAMP_DUTY_RATE': 0.001,
}
TEST_TIME = datetime(2025, 12, 10, 10, 0, 0, tzinfo=timezone.utc)

class TestSimulationExecutor(unittest.TestCase):
    """
    测试 SimulationExecutor 的核心交易模拟逻辑和财务计算。
    """

    def setUp(self):
        """设置 SimulationExecutor 实例。"""
        self.executor = SimulationExecutor(FINANCE_PARAMS)

    def test_initialization(self):
        """测试初始化时核心变量是否正确设置。"""
        self.assertEqual(self.executor.cash, FINANCE_PARAMS['INITIAL_CAPITAL'])
        self.assertEqual(self.executor.position, 0.0)
        self.assertEqual(self.executor.avg_cost, 0.0)
        self.assertEqual(self.executor.MIN_LOT_SIZE, 100)

    def test_buy_calculation_within_limits(self):
        """测试购买数量计算：数量受 MAX_ALLOCATION 和 MIN_LOT_SIZE 限制。"""
        current_price = 100.0
        self.executor.cash = 100000.0 # 初始现金
        
        # MAX_ALLOCATION: 100000 * 0.2 = 20000.0 USD
        # 理论可购买数量: 20000.0 / 100.0 = 200 股
        # 四舍五入到 MIN_LOT_SIZE (100) 的倍数: 200 股
        
        # 模拟执行，但我们直接调用私有方法来测试计算逻辑
        result = self.executor._execute_buy(TEST_TIME, current_price) 
        
        expected_qty = 200.0
        
        self.assertTrue(result) # 交易成功
        self.assertEqual(self.executor.position, expected_qty)
        
        # 验证交易日志中的数量
        self.assertAlmostEqual(self.executor.trade_log[0]['qty'], expected_qty)
        
        
    def test_buy_calculation_rounding_down(self):
        """测试购买数量计算：向下取整到 MIN_LOT_SIZE。"""
        current_price = 105.0
        self.executor.cash = 30000.0 
        
        # MAX_ALLOCATION: 30000 * 0.2 = 6000.0 USD
        # 理论可购买数量: 6000.0 / 105.0 = 57.14 股
        # 四舍五入到 MIN_LOT_SIZE (100) 的倍数: 0 股 (不足 100)
        
        result = self.executor._execute_buy(TEST_TIME, current_price)
        
        self.assertFalse(result) # 交易失败，因为数量为 0
        self.assertEqual(self.executor.position, 0.0)
        self.assertEqual(len(self.executor.trade_log), 0)
        
    def test_buy_calculation_rounding_up(self):
        """测试购买数量计算：数量向下取整到 MIN_LOT_SIZE 的最大倍数。"""
        current_price = 100.0
        self.executor.cash = 35000.0 
        
        # MAX_ALLOCATION: 35000 * 0.2 = 7000.0 USD
        # 理论可购买数量: 7000.0 / 100.0 = 70 股
        # 四舍五入到 MIN_LOT_SIZE (100) 的倍数: 0 股 (不足 100)
        
        result = self.executor._execute_buy(TEST_TIME, current_price)
        
        self.assertFalse(result) # 交易失败
        self.assertEqual(self.executor.position, 0.0)

    def test_buy_execution_details(self):
        """测试成功的买入交易的费用和价格计算。"""
        
        current_price = 100.0
        self.executor.cash = 100000.0
        
        # 预期购买数量 (基于 MAX_ALLOCATION): 200 股
        executed_qty = 200.0
        
        # 预期成交价格: 100.0 * (1 + 0.0001) = 100.01
        executed_price = current_price * (1 + FINANCE_PARAMS['SLIPPAGE_RATE'])
        
        # 预期费用: 200 * 100.01 * 0.0003 = 6.0006
        fee = executed_qty * executed_price * FINANCE_PARAMS['COMMISSION_RATE']
        
        # 调用公共接口进行测试
        result = self.executor.execute_trade(
            TEST_TIME, 'BUY', current_price, self.executor.position, self.executor.cash, self.executor.avg_cost
        )
        
        self.assertTrue(result['executed'])
        self.assertAlmostEqual(result['executed_qty'], executed_qty)
        self.assertAlmostEqual(result['executed_price'], executed_price)
        self.assertAlmostEqual(result['fee'], fee)
        
        # 验证平均成本和现金更新
        total_cost = executed_qty * executed_price + fee
        self.assertAlmostEqual(self.executor.cash, 100000.0 - total_cost)
        self.assertAlmostEqual(self.executor.avg_cost, executed_price)

    def test_sell_execution_details(self):
        """测试成功的卖出交易的费用和 P&L 计算。"""
        
        current_price = 110.0
        self.executor.cash = 90000.0
        self.executor.position = 300.0 # 持仓 300 股
        self.executor.avg_cost = 95.0  # 平均成本
        
        qty_to_sell = 300.0
        
        # 预期成交价格: 110.0 * (1 - 0.0001) = 109.989
        executed_price = current_price * (1 - FINANCE_PARAMS['SLIPPAGE_RATE']) 
        
        # 收入 (滑点后): 300 * 109.989 = 32996.7
        income = qty_to_sell * executed_price 
        
        # 预期费用 (手续费 + 印花税)
        commission = income * FINANCE_PARAMS['COMMISSION_RATE'] # 32996.7 * 0.0003 = 9.89901
        stamp_duty = income * FINANCE_PARAMS['STAMP_DUTY_RATE'] # 32996.7 * 0.001 = 32.9967
        total_fee = commission + stamp_duty                     # 42.89571
        
        # 预期 P&L: 收入 - 费用 - 成本 = 32996.7 - 42.89571 - (300 * 95.0) 
        # 32996.7 - 42.89571 - 28500.0 = 4453.80429
        capital_cost = qty_to_sell * self.executor.avg_cost
        expected_net_pnl = income - total_fee - capital_cost
        
        # 调用公共接口进行测试
        result = self.executor.execute_trade(
            TEST_TIME, 'SELL', current_price, self.executor.position, self.executor.cash, self.executor.avg_cost
        )
        
        self.assertTrue(result['executed'])
        self.assertAlmostEqual(result['executed_qty'], qty_to_sell)
        self.assertAlmostEqual(result['executed_price'], executed_price)
        self.assertAlmostEqual(result['fee'], total_fee)
        
        # 验证 P&L 是否被正确记录 (PositionManager 负责 P&L 核算，这里检查返回结果)
        # SimulationExecutor 不直接返回 P&L，所以需要检查 trade_log 是否正确更新。
        self.assertTrue(self.executor.trade_log[-1]['net_pnl'] > 0)
        self.assertAlmostEqual(self.executor.trade_log[-1]['net_pnl'], expected_net_pnl)
        
        # 验证最终状态
        self.assertAlmostEqual(self.executor.position, 0.0)
        self.assertAlmostEqual(self.executor.avg_cost, 0.0)
        self.assertAlmostEqual(self.executor.cash, 90000.0 + (income - total_fee))

    def test_sell_no_position(self):
        """测试没有仓位时尝试卖出的失败情况。"""
        current_price = 100.0
        self.executor.position = 0.0

        result = self.executor.execute_trade(
            TEST_TIME, 'SELL', current_price, self.executor.position, self.executor.cash, self.executor.avg_cost
        )
        
        self.assertFalse(result['executed'])
        self.assertIn("无执行信号或无仓位可卖", result['log_message'])


if __name__ == '__main__':
    unittest.main()