# src/unit_tests/test_simulation_executor.py

import unittest
from datetime import datetime, timezone

from src.executor.simulation_executor import SimulationExecutor


class TestSimulationExecutorInitialization(unittest.TestCase):
    """测试 SimulationExecutor 初始化"""
    
    def test_default_initialization(self):
        """测试默认参数初始化"""
        executor = SimulationExecutor({})
        self.assertEqual(executor.commission_rate, 0.0003)
        self.assertEqual(executor.slippage_rate, 0.0001)
    
    def test_custom_initialization(self):
        """测试自定义参数初始化"""
        executor = SimulationExecutor({
            'COMMISSION_RATE': 0.001,
            'SLIPPAGE_RATE': 0.0005
        })
        self.assertEqual(executor.commission_rate, 0.001)
        self.assertEqual(executor.slippage_rate, 0.0005)


class TestSimulationExecutorBuy(unittest.TestCase):
    """测试买入执行"""
    
    def setUp(self):
        self.executor = SimulationExecutor({
            'COMMISSION_RATE': 0.001,
            'SLIPPAGE_RATE': 0.001
        })
    
    def test_buy_success(self):
        """测试成功买入"""
        result = self.executor.execute('BUY', 100, 50.0, 'TSLA')
        
        self.assertTrue(result['success'])
        self.assertEqual(result['signal'], 'BUY')
        self.assertEqual(result['qty'], 100)
        # 价格上滑：50 * 1.001 = 50.05
        self.assertAlmostEqual(result['price'], 50.05, places=2)
        # 费用：100 * 50.05 * 0.001 = 5.005
        self.assertAlmostEqual(result['fee'], 5.005, places=2)
    
    def test_buy_invalid_qty(self):
        """测试无效数量"""
        result = self.executor.execute('BUY', 0, 50.0, 'TSLA')
        self.assertFalse(result['success'])
        self.assertIn('Invalid quantity', result['error'])
    
    def test_buy_invalid_price(self):
        """测试无效价格"""
        result = self.executor.execute('BUY', 100, -10.0, 'TSLA')
        self.assertFalse(result['success'])
        self.assertIn('Invalid price', result['error'])


class TestSimulationExecutorSell(unittest.TestCase):
    """测试卖出执行"""
    
    def setUp(self):
        self.executor = SimulationExecutor({
            'COMMISSION_RATE': 0.001,
            'SLIPPAGE_RATE': 0.001
        })
    
    def test_sell_success(self):
        """测试成功卖出"""
        result = self.executor.execute('SELL', 100, 50.0, 'TSLA')
        
        self.assertTrue(result['success'])
        self.assertEqual(result['signal'], 'SELL')
        self.assertEqual(result['qty'], 100)
        # 价格下滑：50 * 0.999 = 49.95
        self.assertAlmostEqual(result['price'], 49.95, places=2)


class TestSimulationExecutorShort(unittest.TestCase):
    """测试做空执行"""
    
    def setUp(self):
        self.executor = SimulationExecutor({
            'COMMISSION_RATE': 0.001,
            'SLIPPAGE_RATE': 0.001
        })
    
    def test_short_success(self):
        """测试成功做空"""
        result = self.executor.execute('SHORT', 100, 50.0, 'TSLA')
        
        self.assertTrue(result['success'])
        self.assertEqual(result['signal'], 'SHORT')
        self.assertEqual(result['qty'], 100)
        # 卖出时价格下滑：50 * 0.999 = 49.95
        self.assertAlmostEqual(result['price'], 49.95, places=2)


class TestSimulationExecutorCover(unittest.TestCase):
    """测试平空执行"""
    
    def setUp(self):
        self.executor = SimulationExecutor({
            'COMMISSION_RATE': 0.001,
            'SLIPPAGE_RATE': 0.001
        })
    
    def test_cover_success(self):
        """测试成功平空"""
        result = self.executor.execute('COVER', 100, 50.0, 'TSLA')
        
        self.assertTrue(result['success'])
        self.assertEqual(result['signal'], 'COVER')
        self.assertEqual(result['qty'], 100)
        # 买入时价格上滑：50 * 1.001 = 50.05
        self.assertAlmostEqual(result['price'], 50.05, places=2)


class TestSimulationExecutorInvalidSignal(unittest.TestCase):
    """测试无效信号"""
    
    def setUp(self):
        self.executor = SimulationExecutor({})
    
    def test_invalid_signal(self):
        """测试无效信号"""
        result = self.executor.execute('INVALID', 100, 50.0, 'TSLA')
        self.assertFalse(result['success'])
        self.assertIn('Invalid signal', result['error'])
    
    def test_hold_signal(self):
        """测试 HOLD 信号"""
        result = self.executor.execute('HOLD', 100, 50.0, 'TSLA')
        self.assertFalse(result['success'])


class TestSimulationExecutorSlippage(unittest.TestCase):
    """测试滑点计算"""
    
    def test_buy_slippage_up(self):
        """测试买入时价格上滑"""
        executor = SimulationExecutor({'SLIPPAGE_RATE': 0.01})
        result = executor.execute('BUY', 100, 100.0, 'TSLA')
        
        # 买入上滑 1%：100 * 1.01 = 101
        self.assertAlmostEqual(result['price'], 101.0, places=2)
    
    def test_sell_slippage_down(self):
        """测试卖出时价格下滑"""
        executor = SimulationExecutor({'SLIPPAGE_RATE': 0.01})
        result = executor.execute('SELL', 100, 100.0, 'TSLA')
        
        # 卖出下滑 1%：100 * 0.99 = 99
        self.assertAlmostEqual(result['price'], 99.0, places=2)
    
    def test_short_slippage_down(self):
        """测试做空时价格下滑"""
        executor = SimulationExecutor({'SLIPPAGE_RATE': 0.01})
        result = executor.execute('SHORT', 100, 100.0, 'TSLA')
        
        # 做空（卖出）下滑 1%
        self.assertAlmostEqual(result['price'], 99.0, places=2)
    
    def test_cover_slippage_up(self):
        """测试平空时价格上滑"""
        executor = SimulationExecutor({'SLIPPAGE_RATE': 0.01})
        result = executor.execute('COVER', 100, 100.0, 'TSLA')
        
        # 平空（买入）上滑 1%
        self.assertAlmostEqual(result['price'], 101.0, places=2)


if __name__ == '__main__':
    unittest.main()