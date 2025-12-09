# src/unit_tests/test_position_manager.py

import unittest
from unittest.mock import MagicMock
from datetime import datetime, timezone
import pandas as pd

from src.manager.position_manager import PositionManager


class TestPositionManagerInitialization(unittest.TestCase):
    """测试 PositionManager 初始化"""
    
    def setUp(self):
        self.mock_executor = MagicMock()
        self.finance_params = {
            'INITIAL_CAPITAL': 100000.0,
            'COMMISSION_RATE': 0.0003,
            'SLIPPAGE_RATE': 0.0001,
            'MIN_LOT_SIZE': 10,
            'MAX_ALLOCATION': 0.2,
        }
    
    def test_initialization(self):
        """测试初始化"""
        pm = PositionManager(self.mock_executor, self.finance_params)
        
        self.assertEqual(pm._cash, 100000.0)
        self.assertEqual(pm._position, 0.0)
        self.assertEqual(pm._avg_cost, 0.0)
        self.assertEqual(pm.position_side, 'flat')
        self.assertFalse(pm._synced)


class TestPositionManagerPositionSide(unittest.TestCase):
    """测试仓位方向判断"""
    
    def setUp(self):
        self.mock_executor = MagicMock()
        self.pm = PositionManager(self.mock_executor, {'INITIAL_CAPITAL': 100000.0})
    
    def test_position_side_flat(self):
        self.pm._position = 0
        self.assertEqual(self.pm.position_side, 'flat')
    
    def test_position_side_long(self):
        self.pm._position = 100
        self.assertEqual(self.pm.position_side, 'long')
    
    def test_position_side_short(self):
        self.pm._position = -100
        self.assertEqual(self.pm.position_side, 'short')


class TestPositionManagerSignalTranslation(unittest.TestCase):
    """测试信号转换"""
    
    def setUp(self):
        self.mock_executor = MagicMock()
        self.pm = PositionManager(self.mock_executor, {'INITIAL_CAPITAL': 100000.0})
    
    # Flat 状态
    def test_translate_buy_when_flat(self):
        self.pm._position = 0
        self.assertEqual(self.pm._translate_signal('BUY'), 'BUY')
    
    def test_translate_short_when_flat(self):
        self.pm._position = 0
        self.assertEqual(self.pm._translate_signal('SHORT'), 'SHORT')
    
    def test_translate_sell_when_flat(self):
        self.pm._position = 0
        self.assertIsNone(self.pm._translate_signal('SELL'))
    
    def test_translate_cover_when_flat(self):
        self.pm._position = 0
        self.assertIsNone(self.pm._translate_signal('COVER'))
    
    # Long 状态
    def test_translate_buy_when_long(self):
        self.pm._position = 100
        self.assertIsNone(self.pm._translate_signal('BUY'))
    
    def test_translate_sell_when_long(self):
        self.pm._position = 100
        self.assertEqual(self.pm._translate_signal('SELL'), 'SELL')
    
    def test_translate_short_when_long(self):
        self.pm._position = 100
        self.assertEqual(self.pm._translate_signal('SHORT'), 'SELL')
    
    # Short 状态
    def test_translate_buy_when_short(self):
        self.pm._position = -100
        self.assertEqual(self.pm._translate_signal('BUY'), 'COVER')
    
    def test_translate_short_when_short(self):
        self.pm._position = -100
        self.assertIsNone(self.pm._translate_signal('SHORT'))
    
    def test_translate_cover_when_short(self):
        self.pm._position = -100
        self.assertEqual(self.pm._translate_signal('COVER'), 'COVER')
    
    def test_translate_hold(self):
        self.assertIsNone(self.pm._translate_signal('HOLD'))


class TestPositionManagerPositionUpdate(unittest.TestCase):
    """测试仓位更新"""
    
    def setUp(self):
        self.mock_executor = MagicMock()
        self.pm = PositionManager(self.mock_executor, {'INITIAL_CAPITAL': 100000.0})
    
    def test_update_position_buy(self):
        self.pm._update_position('BUY', 100, 50.0, 5.0)
        self.assertEqual(self.pm._position, 100)
        self.assertEqual(self.pm._avg_cost, 50.0)
        self.assertEqual(self.pm._cash, 94995.0)
    
    def test_update_position_sell(self):
        self.pm._position = 100
        self.pm._avg_cost = 50.0
        self.pm._cash = 95000.0
        self.pm._update_position('SELL', 100, 60.0, 5.0)
        self.assertEqual(self.pm._position, 0)
        self.assertEqual(self.pm._avg_cost, 0.0)
        self.assertEqual(self.pm._cash, 100995.0)
    
    def test_update_position_short(self):
        self.pm._update_position('SHORT', 100, 50.0, 5.0)
        self.assertEqual(self.pm._position, -100)
        self.assertEqual(self.pm._avg_cost, 50.0)
        self.assertEqual(self.pm._cash, 104995.0)
    
    def test_update_position_cover(self):
        self.pm._position = -100
        self.pm._avg_cost = 50.0
        self.pm._cash = 105000.0
        self.pm._update_position('COVER', 100, 45.0, 5.0)
        self.assertEqual(self.pm._position, 0)
        self.assertEqual(self.pm._avg_cost, 0.0)
        self.assertEqual(self.pm._cash, 100495.0)


class TestPositionManagerAccountStatus(unittest.TestCase):
    """测试账户状态"""
    
    def setUp(self):
        self.mock_executor = MagicMock()
        self.pm = PositionManager(self.mock_executor, {'INITIAL_CAPITAL': 100000.0})
    
    def test_account_status_flat(self):
        status = self.pm.get_account_status(current_price=100.0)
        self.assertEqual(status['position_side'], 'flat')
        self.assertEqual(status['equity'], 100000.0)
    
    def test_account_status_long_profit(self):
        self.pm._position = 100
        self.pm._avg_cost = 50.0
        self.pm._cash = 95000.0
        status = self.pm.get_account_status(current_price=60.0)
        self.assertEqual(status['position_side'], 'long')
        self.assertEqual(status['unrealized_pnl'], 1000.0)
    
    def test_account_status_short_profit(self):
        self.pm._position = -100
        self.pm._avg_cost = 50.0
        self.pm._cash = 105000.0
        status = self.pm.get_account_status(current_price=40.0)
        self.assertEqual(status['position_side'], 'short')
        self.assertEqual(status['unrealized_pnl'], 1000.0)


class TestPositionManagerReset(unittest.TestCase):
    """测试重置"""
    
    def setUp(self):
        self.mock_executor = MagicMock()
        self.pm = PositionManager(self.mock_executor, {'INITIAL_CAPITAL': 100000.0})
    
    def test_reset(self):
        self.pm._cash = 50000.0
        self.pm._position = 100
        self.pm._avg_cost = 50.0
        self.pm._trade_log = [{'test': 'data'}]
        
        self.pm.reset()
        
        self.assertEqual(self.pm._cash, 100000.0)
        self.assertEqual(self.pm._position, 0.0)
        self.assertEqual(self.pm._avg_cost, 0.0)
        self.assertEqual(self.pm._trade_log, [])


if __name__ == '__main__':
    unittest.main()