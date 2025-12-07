import unittest
from unittest.mock import MagicMock, patch, ANY
from datetime import datetime
import os
import sys

# Set path to ensure modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the class to test
from executor.alpaca_trade_executor import AlpacaExecutor
# Import Alpaca related classes for mocking
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError


# Mock Alpaca client and related objects
class MockOrder:
    """Mock Alpaca Order object with id and status attributes."""
    def __init__(self, order_id="test_order_id", status="accepted"):
        self.id = order_id
        self.status = MagicMock(value=status)

class MockAccount:
    """Mock Alpaca Account object with equity attribute."""
    def __init__(self, equity=100000.0):
        self.equity = str(equity)  # equity in Alpaca model is usually a string


class TestAlpacaExecutor(unittest.TestCase):
    
    TICKER = "TSLA"
    
    def setUp(self):
        """Initialize common test data before each test method."""
        # Patch environment variables and TradingClient at the module level
        self.env_patcher = patch.dict('os.environ', {
            'ALPACA_API_KEY_ID': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret'
        })
        self.env_patcher.start()
        
        # Patch TradingClient at the module level where it's imported
        self.client_patcher = patch('executor.alpaca_trade_executor.TradingClient')
        self.MockTradingClient = self.client_patcher.start()
        
        # Create a mock client instance that will be returned by TradingClient()
        self.mock_client_instance = MagicMock()
        self.MockTradingClient.return_value = self.mock_client_instance
        
        # Now instantiate AlpacaExecutor (it will use the mocked TradingClient)
        self.executor = AlpacaExecutor(paper=True, max_allocation_rate=0.2)
        
        self.test_timestamp = datetime(2023, 1, 1, 9, 30)
    
    def tearDown(self):
        """Clean up patches after each test."""
        self.client_patcher.stop()
        self.env_patcher.stop()

    def test_initialization(self):
        """Test that AlpacaExecutor correctly initializes TradingClient."""
        # Verify TradingClient was called during __init__
        self.MockTradingClient.assert_called_once()
        # Verify it was initialized with paper=True
        call_kwargs = self.MockTradingClient.call_args[1]
        self.assertTrue(call_kwargs['paper'])
        self.assertEqual(self.executor.MAX_ALLOCATION_RATE, 0.2)
        
    # --- BUY trade tests ---

    def test_buy_successful_execution(self):
        """Test successful buy trade, verify quantity calculation and API calls."""
        current_equity = 100000.0
        current_cash = 50000.0
        
        # Mock get_account response
        self.mock_client_instance.get_account.return_value = MockAccount(equity=current_equity)
        # Mock submit_order response
        self.mock_client_instance.submit_order.return_value = MockOrder(order_id="TEST_BUY_ORDER")
        
        current_price = 100.0
        
        # Expected calculation:
        # MAX_ALLOCATION_RATE = 0.2
        # Capital to use: min(50000, 100000 * 0.2) = 20000.0
        # Qty: floor(20000.0 / 100.0 / 1) * 1 = 200.0
        expected_qty = 200.0

        result = self.executor.execute_trade(
            timestamp=self.test_timestamp,
            signal='BUY',
            current_price=current_price,
            current_position=0.0,
            current_cash=current_cash,
            avg_cost=0.0
        )
        
        # Verify API calls
        self.mock_client_instance.get_account.assert_called_once()
        self.mock_client_instance.submit_order.assert_called_once()
        
        # Verify submit_order was called with correct parameters
        call_args = self.mock_client_instance.submit_order.call_args[0][0]
        self.assertEqual(call_args.symbol, self.TICKER)
        self.assertEqual(call_args.qty, expected_qty)
        self.assertEqual(call_args.side, OrderSide.BUY)
        
        # Verify return result
        self.assertTrue(result['executed'])
        self.assertEqual(result['trade_type'], 'BUY')
        self.assertEqual(result['executed_qty'], expected_qty)
        self.assertEqual(result['executed_price'], current_price)
        self.assertEqual(result['fee'], 0.0)

    def test_buy_failure_low_capital(self):
        """Test buy failure due to insufficient available capital."""
        # Mock account with very low equity
        current_equity = 500.0
        current_cash = 10.0
        
        self.mock_client_instance.get_account.return_value = MockAccount(equity=current_equity)
        
        current_price = 100.0
        
        # Expected calculation:
        # Capital to use: min(10.0, 500.0 * 0.2) = 10.0
        # Qty: floor(10.0 / 100.0 / 1) * 1 = 0.0
        # This should trigger "计算数量低于最小交易单位" error
        
        result = self.executor.execute_trade(
            timestamp=self.test_timestamp,
            signal='BUY',
            current_price=current_price,
            current_position=0.0,
            current_cash=current_cash,
            avg_cost=0.0
        )
        
        # Verify result
        self.assertFalse(result['executed'])
        # The executor returns "计算数量低于最小交易单位" which is technically correct
        self.assertIn("计算数量低于最小交易单位", result['log_message'])
        self.mock_client_instance.get_account.assert_called_once()
        self.mock_client_instance.submit_order.assert_not_called()
    
    def test_buy_failure_zero_capital(self):
        """Test buy failure when capital_to_use is zero or negative."""
        # Test the specific "资金不足" error path
        current_equity = 100000.0
        current_cash = 0.0  # No cash available
        
        self.mock_client_instance.get_account.return_value = MockAccount(equity=current_equity)
        
        result = self.executor.execute_trade(
            timestamp=self.test_timestamp,
            signal='BUY',
            current_price=100.0,
            current_position=0.0,
            current_cash=current_cash,
            avg_cost=0.0
        )
        
        # This should trigger the "资金不足" error
        self.assertFalse(result['executed'])
        self.assertIn("资金不足", result['log_message'])
        
    def test_buy_failure_alpaca_api_error(self):
        """Test Alpaca API throws exception during buy."""
        self.mock_client_instance.get_account.return_value = MockAccount(equity=100000.0)
        self.mock_client_instance.submit_order.side_effect = APIError("Rate limit exceeded")
        
        result = self.executor.execute_trade(
            timestamp=self.test_timestamp,
            signal='BUY',
            current_price=100.0,
            current_position=0.0,
            current_cash=50000.0,
            avg_cost=0.0
        )
        
        # Verify result
        self.assertFalse(result['executed'])
        self.assertIn("Alpaca API 错误", result['log_message'])
        self.mock_client_instance.get_account.assert_called_once()
        self.mock_client_instance.submit_order.assert_called_once()

        
    # --- SELL trade tests ---

    def test_sell_successful_execution(self):
        """Test successful sell (close position) trade."""
        current_position = 300.0  # Holding 300 shares
        
        self.mock_client_instance.close_position.return_value = MockOrder(order_id="TEST_SELL_ORDER")
        
        result = self.executor.execute_trade(
            timestamp=self.test_timestamp,
            signal='SELL',
            current_price=100.0,
            current_position=current_position,
            current_cash=50000.0,
            avg_cost=95.0
        )
        
        # Verify API call
        self.mock_client_instance.close_position.assert_called_once()
        
        # Verify return result
        self.assertTrue(result['executed'])
        self.assertEqual(result['trade_type'], 'SELL')
        self.assertEqual(result['executed_qty'], current_position)
        self.assertEqual(result['fee'], 0.0)
        
    def test_sell_failure_no_position(self):
        """Test sell signal should be rejected when there's no position."""
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
        self.mock_client_instance.close_position.assert_not_called()

    def test_sell_failure_alpaca_api_error(self):
        """Test Alpaca API throws exception during sell."""
        self.mock_client_instance.close_position.side_effect = APIError("Invalid position to close")
        
        result = self.executor.execute_trade(
            timestamp=self.test_timestamp,
            signal='SELL',
            current_price=100.0,
            current_position=100.0,
            current_cash=50000.0,
            avg_cost=95.0
        )
        
        # Verify result
        self.assertFalse(result['executed'])
        self.assertIn("Alpaca API 错误", result['log_message'])
        self.mock_client_instance.close_position.assert_called_once()


if __name__ == '__main__':
    unittest.main()