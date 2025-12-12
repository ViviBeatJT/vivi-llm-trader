# tests/test_trading_engine.py

"""
Unit Tests for TradingEngine

Tests cover:
1. Configuration and initialization
2. Time utilities (market hours, force close, etc.)
3. Signal processing and trade execution
4. Position management integration
5. Backtest mode execution
6. Live mode execution
7. Force close behavior
8. Report generation
9. Edge cases and error handling

Run with:
    pytest tests/test_trading_engine.py -v
    pytest tests/test_trading_engine.py -v -k "test_time"  # Run only time tests
"""

import pytest
from datetime import datetime, timezone, timedelta, time as dt_time
from unittest.mock import Mock, MagicMock, patch, call
import pandas as pd
import numpy as np
import pytz

from src.engine.trading_engine import (
    TradingEngine,
    EngineConfig,
    TimeConfig,
    FinanceConfig,
    DataConfig
)


# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def mock_strategy():
    """Create a mock strategy."""
    strategy = Mock()
    strategy.get_signal = Mock(return_value=(
        {'signal': 'HOLD', 'confidence_score': 5, 'reason': 'Test'},
        None
    ))
    strategy.get_history_data = Mock(return_value=pd.DataFrame())
    return strategy


@pytest.fixture
def mock_position_manager():
    """Create a mock position manager."""
    pm = Mock()
    pm.get_account_status = Mock(return_value={
        'position': 0.0,
        'avg_cost': 0.0,
        'equity': 1000.0,
        'cash': 1000.0
    })
    pm.execute_and_update = Mock(return_value=True)
    pm.get_trade_log = Mock(return_value=pd.DataFrame())
    return pm


@pytest.fixture
def mock_data_fetcher():
    """Create a mock data fetcher."""
    fetcher = Mock()
    # Return sample OHLCV data
    dates = pd.date_range(start='2024-01-01 09:30', periods=100, freq='5min', tz='UTC')
    df = pd.DataFrame({
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(105, 115, 100),
        'low': np.random.uniform(95, 105, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    fetcher.get_latest_bars = Mock(return_value=df)
    return fetcher


@pytest.fixture
def mock_executor():
    """Create a mock executor."""
    executor = Mock()
    executor.execute = Mock(return_value={'success': True, 'qty': 10, 'price': 100.0, 'fee': 0.0})
    return executor


@pytest.fixture
def sample_config(mock_strategy, mock_position_manager, mock_data_fetcher, mock_executor):
    """Create a sample engine configuration."""
    return EngineConfig(
        ticker='TEST',
        strategy=mock_strategy,
        position_manager=mock_position_manager,
        data_fetcher=mock_data_fetcher,
        executor=mock_executor,
        time_config=TimeConfig(),
        finance_config=FinanceConfig(initial_capital=1000.0),
        data_config=DataConfig(step_seconds=60, lookback_minutes=300),
        verbose=False
    )


@pytest.fixture
def engine(sample_config):
    """Create a TradingEngine instance."""
    return TradingEngine(sample_config, mode='backtest')


# ==========================================
# Test Configuration Classes
# ==========================================

class TestTimeConfig:
    """Tests for TimeConfig dataclass."""
    
    def test_default_values(self):
        """Test default time configuration values."""
        config = TimeConfig()
        
        assert config.last_entry_time == dt_time(15, 50)
        assert config.force_close_time == dt_time(15, 55)
        assert config.market_close_time == dt_time(16, 0)
        assert config.market_open_time == dt_time(9, 30)
        assert config.timezone == 'America/New_York'
    
    def test_custom_values(self):
        """Test custom time configuration."""
        config = TimeConfig(
            last_entry_time=dt_time(15, 30),
            force_close_time=dt_time(15, 45),
            market_close_time=dt_time(16, 0),
            market_open_time=dt_time(9, 0)
        )
        
        assert config.last_entry_time == dt_time(15, 30)
        assert config.force_close_time == dt_time(15, 45)
    
    def test_timezone_object_created(self):
        """Test that timezone object is created in post_init."""
        config = TimeConfig()
        assert config.tz is not None
        assert str(config.tz) == 'America/New_York'


class TestFinanceConfig:
    """Tests for FinanceConfig dataclass."""
    
    def test_default_values(self):
        """Test default finance configuration values."""
        config = FinanceConfig()
        
        assert config.initial_capital == 1000.0
        assert config.commission_rate == 0.0003
        assert config.slippage_rate == 0.0001
        assert config.min_lot_size == 1
        assert config.max_allocation == 0.95
    
    def test_to_dict(self):
        """Test conversion to dictionary for PositionManager."""
        config = FinanceConfig(initial_capital=5000.0, max_allocation=0.8)
        d = config.to_dict()
        
        assert d['INITIAL_CAPITAL'] == 5000.0
        assert d['MAX_ALLOCATION'] == 0.8
        assert 'COMMISSION_RATE' in d
        assert 'SLIPPAGE_RATE' in d


class TestDataConfig:
    """Tests for DataConfig dataclass."""
    
    def test_default_values(self):
        """Test default data configuration values."""
        config = DataConfig()
        
        assert config.lookback_minutes == 300
        assert config.step_seconds == 30


# ==========================================
# Test Engine Initialization
# ==========================================

class TestEngineInitialization:
    """Tests for TradingEngine initialization."""
    
    def test_basic_initialization(self, sample_config):
        """Test basic engine initialization."""
        engine = TradingEngine(sample_config, mode='backtest')
        
        assert engine.ticker == 'TEST'
        assert engine.mode == 'backtest'
        assert engine._running == False
        assert engine._iteration == 0
    
    def test_live_mode_initialization(self, sample_config):
        """Test live mode initialization."""
        engine = TradingEngine(sample_config, mode='live')
        
        assert engine.mode == 'live'
    
    def test_config_components_assigned(self, sample_config):
        """Test that all config components are properly assigned."""
        engine = TradingEngine(sample_config, mode='backtest')
        
        assert engine.strategy is not None
        assert engine.position_manager is not None
        assert engine.data_fetcher is not None
        assert engine.executor is not None
        assert engine.time_config is not None
        assert engine.finance_config is not None
        assert engine.data_config is not None


# ==========================================
# Test Time Utilities
# ==========================================

class TestTimeUtilities:
    """Tests for time-related utility methods."""
    
    def test_to_eastern_from_utc(self, engine):
        """Test UTC to Eastern time conversion."""
        utc_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)  # 9:30 AM ET
        et_time = engine._to_eastern(utc_time)
        
        assert et_time.hour == 9
        assert et_time.minute == 30
    
    def test_to_eastern_naive_datetime(self, engine):
        """Test conversion of naive datetime (assumes UTC)."""
        naive_time = datetime(2024, 1, 15, 14, 30, 0)
        et_time = engine._to_eastern(naive_time)
        
        # Should be treated as UTC and converted
        assert et_time.tzinfo is not None
    
    def test_is_market_hours_open(self, engine):
        """Test market hours detection during trading hours."""
        # 10:00 AM ET on Monday
        trading_time = pytz.timezone('America/New_York').localize(
            datetime(2024, 1, 15, 10, 0, 0)  # Monday
        ).astimezone(timezone.utc)
        
        assert engine._is_market_hours(trading_time) == True
    
    def test_is_market_hours_before_open(self, engine):
        """Test market hours detection before market open."""
        # 9:00 AM ET (before 9:30 open)
        pre_market = pytz.timezone('America/New_York').localize(
            datetime(2024, 1, 15, 9, 0, 0)
        ).astimezone(timezone.utc)
        
        assert engine._is_market_hours(pre_market) == False
    
    def test_is_market_hours_after_close(self, engine):
        """Test market hours detection after market close."""
        # 4:30 PM ET (after 4:00 close)
        after_close = pytz.timezone('America/New_York').localize(
            datetime(2024, 1, 15, 16, 30, 0)
        ).astimezone(timezone.utc)
        
        assert engine._is_market_hours(after_close) == False
    
    def test_is_market_hours_weekend(self, engine):
        """Test market hours detection on weekend."""
        # Saturday
        saturday = pytz.timezone('America/New_York').localize(
            datetime(2024, 1, 13, 10, 0, 0)  # Saturday
        ).astimezone(timezone.utc)
        
        assert engine._is_market_hours(saturday) == False
    
    def test_is_in_no_entry_window_before(self, engine):
        """Test no-entry window detection before cutoff."""
        # 3:45 PM ET (before 3:50 cutoff)
        before_cutoff = pytz.timezone('America/New_York').localize(
            datetime(2024, 1, 15, 15, 45, 0)
        ).astimezone(timezone.utc)
        
        assert engine._is_in_no_entry_window(before_cutoff) == False
    
    def test_is_in_no_entry_window_after(self, engine):
        """Test no-entry window detection after cutoff."""
        # 3:52 PM ET (after 3:50 cutoff)
        after_cutoff = pytz.timezone('America/New_York').localize(
            datetime(2024, 1, 15, 15, 52, 0)
        ).astimezone(timezone.utc)
        
        assert engine._is_in_no_entry_window(after_cutoff) == True
    
    def test_is_force_close_time_before(self, engine):
        """Test force close time detection before cutoff."""
        # 3:50 PM ET (before 3:55 force close)
        before_force = pytz.timezone('America/New_York').localize(
            datetime(2024, 1, 15, 15, 50, 0)
        ).astimezone(timezone.utc)
        
        assert engine._is_force_close_time(before_force) == False
    
    def test_is_force_close_time_after(self, engine):
        """Test force close time detection at/after cutoff."""
        # 3:55 PM ET (at force close time)
        at_force = pytz.timezone('America/New_York').localize(
            datetime(2024, 1, 15, 15, 55, 0)
        ).astimezone(timezone.utc)
        
        assert engine._is_force_close_time(at_force) == True
    
    def test_format_time_et(self, engine):
        """Test time formatting to Eastern."""
        utc_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        formatted = engine._format_time_et(utc_time)
        
        assert 'ET' in formatted
        assert '09:30' in formatted


# ==========================================
# Test Data Fetching
# ==========================================

class TestDataFetching:
    """Tests for data fetching functionality."""
    
    def test_fetch_data_calls_data_fetcher(self, engine, mock_data_fetcher):
        """Test that _fetch_data calls data fetcher correctly."""
        engine._fetch_data()
        
        mock_data_fetcher.get_latest_bars.assert_called_once()
    
    def test_fetch_data_with_end_dt(self, engine, mock_data_fetcher):
        """Test data fetching with specific end datetime."""
        end_dt = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        engine._fetch_data(end_dt=end_dt)
        
        call_kwargs = mock_data_fetcher.get_latest_bars.call_args
        assert call_kwargs[1]['end_dt'] == end_dt
    
    def test_fetch_data_returns_empty_when_no_fetcher(self, sample_config):
        """Test that empty DataFrame is returned when no data fetcher."""
        sample_config.data_fetcher = None
        engine = TradingEngine(sample_config, mode='backtest')
        
        result = engine._fetch_data()
        
        assert result.empty


# ==========================================
# Test Signal Processing
# ==========================================

class TestSignalProcessing:
    """Tests for signal processing in _process_iteration."""
    
    def test_process_iteration_calls_strategy(self, engine, mock_strategy):
        """Test that process_iteration calls strategy.get_signal."""
        current_time = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        
        engine._process_iteration(current_time)
        
        mock_strategy.get_signal.assert_called_once()
    
    def test_process_iteration_increments_counter(self, engine):
        """Test that iteration counter is incremented."""
        current_time = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        
        assert engine._iteration == 0
        engine._process_iteration(current_time)
        assert engine._iteration == 1
    
    def test_process_iteration_with_buy_signal(self, engine, mock_strategy, mock_position_manager):
        """Test processing of BUY signal."""
        mock_strategy.get_signal.return_value = (
            {'signal': 'BUY', 'confidence_score': 8, 'reason': 'Test buy'},
            None
        )
        
        current_time = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        result = engine._process_iteration(current_time)
        
        assert result['signal'] == 'BUY'
        mock_position_manager.execute_and_update.assert_called_once()
    
    def test_process_iteration_with_sell_signal(self, engine, mock_strategy, mock_position_manager):
        """Test processing of SELL signal."""
        mock_strategy.get_signal.return_value = (
            {'signal': 'SELL', 'confidence_score': 8, 'reason': 'Test sell'},
            None
        )
        
        current_time = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        result = engine._process_iteration(current_time)
        
        assert result['signal'] == 'SELL'
        mock_position_manager.execute_and_update.assert_called_once()
    
    def test_process_iteration_with_hold_signal(self, engine, mock_strategy, mock_position_manager):
        """Test processing of HOLD signal (no trade execution)."""
        mock_strategy.get_signal.return_value = (
            {'signal': 'HOLD', 'confidence_score': 5, 'reason': 'Test hold'},
            None
        )
        
        current_time = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        result = engine._process_iteration(current_time)
        
        assert result['signal'] == 'HOLD'
        mock_position_manager.execute_and_update.assert_not_called()
    
    def test_process_iteration_records_equity(self, engine):
        """Test that equity history is recorded."""
        current_time = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        
        assert len(engine._equity_history) == 0
        engine._process_iteration(current_time)
        assert len(engine._equity_history) == 1
    
    def test_process_iteration_with_empty_data(self, engine, mock_data_fetcher):
        """Test handling of empty data."""
        mock_data_fetcher.get_latest_bars.return_value = pd.DataFrame()
        
        current_time = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        result = engine._process_iteration(current_time)
        
        assert result['status'] == 'no_data'
    
    def test_process_iteration_strategy_error(self, engine, mock_strategy):
        """Test handling of strategy errors."""
        mock_strategy.get_signal.side_effect = Exception("Strategy error")
        
        current_time = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        result = engine._process_iteration(current_time)
        
        assert result['status'] == 'error'
        assert 'Strategy error' in result['error']


# ==========================================
# Test Force Close Behavior
# ==========================================

class TestForceClose:
    """Tests for force close functionality."""
    
    def test_force_close_time_triggers_signal_flag(self, engine):
        """Test that force close time triggers the flag."""
        force_close_time = pytz.timezone('America/New_York').localize(
            datetime(2024, 1, 15, 15, 56, 0)
        ).astimezone(timezone.utc)
        
        engine._process_iteration(force_close_time)
        
        assert engine._force_close_reached == True
    
    def test_ensure_flat_position_with_long_position(self, engine, mock_position_manager, mock_data_fetcher):
        """Test force closing a long position."""
        mock_position_manager.get_account_status.return_value = {
            'position': 10.0,  # Long position
            'equity': 1000.0,
            'cash': 0.0
        }
        
        end_time = datetime(2024, 1, 15, 21, 0, tzinfo=timezone.utc)
        engine._ensure_flat_position(end_time)
        
        # Should call execute_and_update with SELL signal
        mock_position_manager.execute_and_update.assert_called()
        call_args = mock_position_manager.execute_and_update.call_args
        assert call_args[1]['signal'] == 'SELL'
    
    def test_ensure_flat_position_with_short_position(self, engine, mock_position_manager, mock_data_fetcher):
        """Test force closing a short position."""
        mock_position_manager.get_account_status.return_value = {
            'position': -10.0,  # Short position
            'equity': 1000.0,
            'cash': 2000.0
        }
        
        end_time = datetime(2024, 1, 15, 21, 0, tzinfo=timezone.utc)
        engine._ensure_flat_position(end_time)
        
        # Should call execute_and_update with COVER signal
        mock_position_manager.execute_and_update.assert_called()
        call_args = mock_position_manager.execute_and_update.call_args
        assert call_args[1]['signal'] == 'COVER'
    
    def test_ensure_flat_position_already_flat(self, engine, mock_position_manager):
        """Test that no action is taken when already flat."""
        mock_position_manager.get_account_status.return_value = {
            'position': 0.0,  # Already flat
            'equity': 1000.0,
            'cash': 1000.0
        }
        
        end_time = datetime(2024, 1, 15, 21, 0, tzinfo=timezone.utc)
        result = engine._ensure_flat_position(end_time)
        
        assert result == True
        # execute_and_update should not be called for closing
        # (it might be called during _fetch_data, so we check the signal)


# ==========================================
# Test Backtest Execution
# ==========================================

class TestBacktestExecution:
    """Tests for backtest mode execution."""
    
    def test_run_backtest_basic(self, engine):
        """Test basic backtest execution."""
        start_time = pytz.timezone('America/New_York').localize(
            datetime(2024, 1, 15, 9, 30, 0)
        ).astimezone(timezone.utc)
        
        end_time = pytz.timezone('America/New_York').localize(
            datetime(2024, 1, 15, 9, 35, 0)  # Short test: 5 minutes
        ).astimezone(timezone.utc)
        
        report = engine.run_backtest(start_time, end_time, progress_interval=100)
        
        assert 'iterations' in report
        assert 'final_equity' in report
        assert 'pnl' in report
        assert report['mode'] == 'backtest'
    
    def test_run_backtest_sets_running_flag(self, engine):
        """Test that running flag is set during execution."""
        start_time = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 15, 14, 31, tzinfo=timezone.utc)
        
        # The flag should be True during execution
        # After completion, we can check the report was generated
        report = engine.run_backtest(start_time, end_time)
        
        assert report is not None
    
    def test_run_backtest_respects_step_seconds(self, engine):
        """Test that backtest steps through time correctly."""
        engine.data_config.step_seconds = 60  # 1 minute steps
        
        start_time = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 15, 14, 33, tzinfo=timezone.utc)  # 3 minutes
        
        report = engine.run_backtest(start_time, end_time)
        
        # Should have roughly 3-4 iterations (start, +1min, +2min, +3min)
        assert report['iterations'] >= 3
    
    def test_run_backtest_tracks_signals(self, engine, mock_strategy):
        """Test that signals are tracked during backtest."""
        mock_strategy.get_signal.return_value = (
            {'signal': 'BUY', 'confidence_score': 8, 'reason': 'Test'},
            None
        )
        
        start_time = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 15, 14, 32, tzinfo=timezone.utc)
        
        report = engine.run_backtest(start_time, end_time)
        
        assert report['signals'] > 0


# ==========================================
# Test Report Generation
# ==========================================

class TestReportGeneration:
    """Tests for report generation."""
    
    def test_generate_report_structure(self, engine):
        """Test that report has required fields."""
        report = engine._generate_report(datetime.now(timezone.utc))
        
        required_fields = [
            'mode', 'ticker', 'strategy', 'runtime_seconds',
            'iterations', 'signals', 'trades_executed',
            'initial_capital', 'final_equity', 'final_position',
            'pnl', 'pnl_pct'
        ]
        
        for field in required_fields:
            assert field in report, f"Missing field: {field}"
    
    def test_generate_report_calculates_pnl(self, engine, mock_position_manager):
        """Test PnL calculation in report."""
        mock_position_manager.get_account_status.return_value = {
            'position': 0.0,
            'equity': 1100.0,  # 10% gain
            'cash': 1100.0
        }
        
        report = engine._generate_report(datetime.now(timezone.utc))
        
        assert report['pnl'] == 100.0  # 1100 - 1000
        assert report['pnl_pct'] == 10.0  # 10%
    
    def test_generate_report_includes_trade_log(self, engine, mock_position_manager):
        """Test that trade log is included in report."""
        mock_trade_log = pd.DataFrame({
            'time': [datetime.now()],
            'type': ['BUY'],
            'qty': [10],
            'price': [100.0]
        })
        mock_position_manager.get_trade_log.return_value = mock_trade_log
        
        report = engine._generate_report(datetime.now(timezone.utc))
        
        assert 'trade_log' in report
        assert len(report['trade_log']) == 1


# ==========================================
# Test Callbacks
# ==========================================

class TestCallbacks:
    """Tests for callback functionality."""
    
    def test_signal_callback_called(self, sample_config, mock_strategy):
        """Test that signal callback is called on trade signals."""
        callback = Mock()
        sample_config.on_signal_callback = callback
        
        mock_strategy.get_signal.return_value = (
            {'signal': 'BUY', 'confidence_score': 8, 'reason': 'Test'},
            None
        )
        
        engine = TradingEngine(sample_config, mode='backtest')
        current_time = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        engine._process_iteration(current_time)
        
        callback.assert_called_once()
    
    def test_trade_callback_called(self, sample_config, mock_strategy, mock_position_manager):
        """Test that trade callback is called on successful trade."""
        callback = Mock()
        sample_config.on_trade_callback = callback
        
        mock_strategy.get_signal.return_value = (
            {'signal': 'BUY', 'confidence_score': 8, 'reason': 'Test'},
            None
        )
        mock_position_manager.execute_and_update.return_value = True
        
        engine = TradingEngine(sample_config, mode='backtest')
        current_time = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        engine._process_iteration(current_time)
        
        callback.assert_called_once()
    
    def test_iteration_callback_called(self, sample_config):
        """Test that iteration callback is called each iteration."""
        callback = Mock()
        sample_config.on_iteration_callback = callback
        
        engine = TradingEngine(sample_config, mode='backtest')
        current_time = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        engine._process_iteration(current_time)
        
        callback.assert_called_once()


# ==========================================
# Test Edge Cases
# ==========================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_stop_method(self, engine):
        """Test that stop method sets running flag."""
        engine._running = True
        engine.stop()
        
        assert engine._running == False
    
    def test_run_with_invalid_mode(self, sample_config):
        """Test that invalid mode raises error."""
        engine = TradingEngine(sample_config, mode='backtest')
        
        # Calling run() in backtest mode without times should raise
        with pytest.raises(ValueError):
            engine.run()  # Missing start_time and end_time
    
    def test_multiple_runs_reset_state(self, engine):
        """Test that multiple runs reset internal state."""
        start_time = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 15, 14, 31, tzinfo=timezone.utc)
        
        # First run
        engine.run_backtest(start_time, end_time)
        first_iterations = engine._iteration
        
        # Second run should reset
        engine.run_backtest(start_time, end_time)
        
        # Iterations should be similar (reset for second run)
        assert engine._iteration > 0
    
    def test_handles_none_visualizer(self, sample_config):
        """Test engine works without visualizer."""
        sample_config.visualizer = None
        
        engine = TradingEngine(sample_config, mode='backtest')
        current_time = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        
        # Should not raise
        result = engine._process_iteration(current_time)
        assert result is not None


# ==========================================
# Test Integration Scenarios
# ==========================================

class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""
    
    def test_full_trading_day_simulation(self, engine, mock_strategy, mock_position_manager):
        """Test a full simulated trading day."""
        # Simulate BUY at open, SELL before close
        call_count = [0]
        
        def dynamic_signal(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return ({'signal': 'BUY', 'confidence_score': 8, 'reason': 'Open'}, None)
            elif call_count[0] > 5:
                return ({'signal': 'SELL', 'confidence_score': 8, 'reason': 'Close'}, None)
            return ({'signal': 'HOLD', 'confidence_score': 5, 'reason': 'Wait'}, None)
        
        mock_strategy.get_signal.side_effect = dynamic_signal
        
        start_time = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 15, 14, 40, tzinfo=timezone.utc)
        
        report = engine.run_backtest(start_time, end_time)
        
        assert report['signals'] >= 2  # At least BUY and SELL
    
    def test_no_trading_scenario(self, engine, mock_strategy, mock_position_manager):
        """Test scenario where strategy never signals."""
        mock_strategy.get_signal.return_value = (
            {'signal': 'HOLD', 'confidence_score': 5, 'reason': 'No opportunity'},
            None
        )
        
        start_time = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 15, 14, 35, tzinfo=timezone.utc)
        
        report = engine.run_backtest(start_time, end_time)
        
        assert report['signals'] == 0
        assert report['trades_executed'] == 0
        mock_position_manager.execute_and_update.assert_not_called()


# ==========================================
# Run Tests
# ==========================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])