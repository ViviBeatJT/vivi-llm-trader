# tests/test_local_data_fetcher.py

"""
Unit Tests for LocalDataFetcher

Tests cover:
1. Initialization and data loading
2. Date range filtering (simulating backtest behavior)
3. Timeframe resampling
4. Edge cases and error handling

Run with:
    pytest tests/test_local_data_fetcher.py -v
"""

import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from src.data_fetcher.local_data_fetcher import LocalDataFetcher
from src.data_fetcher.base_data_fetcher import BaseDataFetcher


# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def sample_csv_content():
    """Sample CSV content matching expected format."""
    return """symbol,timestamp,open,high,low,close,volume,trade_count,vwap
AAPL,2024-12-11 09:30:00+00:00,247.77,248.00,247.50,247.80,1000.0,50.0,247.77
AAPL,2024-12-11 09:35:00+00:00,247.80,248.10,247.70,248.00,1200.0,60.0,247.90
AAPL,2024-12-11 09:40:00+00:00,248.00,248.20,247.90,248.10,1100.0,55.0,248.05
AAPL,2024-12-11 09:45:00+00:00,248.10,248.30,248.00,248.20,1300.0,65.0,248.15
AAPL,2024-12-11 09:50:00+00:00,248.20,248.40,248.10,248.30,1400.0,70.0,248.25
AAPL,2024-12-11 09:55:00+00:00,248.30,248.50,248.20,248.40,1500.0,75.0,248.35
AAPL,2024-12-11 10:00:00+00:00,248.40,248.60,248.30,248.50,1600.0,80.0,248.45
AAPL,2024-12-11 10:05:00+00:00,248.50,248.70,248.40,248.60,1700.0,85.0,248.55
AAPL,2024-12-11 10:10:00+00:00,248.60,248.80,248.50,248.70,1800.0,90.0,248.65
AAPL,2024-12-11 10:15:00+00:00,248.70,248.90,248.60,248.80,1900.0,95.0,248.75"""


@pytest.fixture
def sample_tsla_csv():
    """Sample CSV content for TSLA."""
    return """symbol,timestamp,open,high,low,close,volume,trade_count,vwap
TSLA,2024-12-11 09:30:00+00:00,350.00,352.00,349.00,351.00,5000.0,200.0,350.50
TSLA,2024-12-11 09:35:00+00:00,351.00,353.00,350.00,352.00,5500.0,220.0,351.50
TSLA,2024-12-11 09:40:00+00:00,352.00,354.00,351.00,353.00,6000.0,240.0,352.50"""


@pytest.fixture
def data_dir(tmp_path, sample_csv_content, sample_tsla_csv):
    """Create a temporary data directory with CSV files."""
    aapl_file = tmp_path / "AAPL.csv"
    aapl_file.write_text(sample_csv_content)
    
    tsla_file = tmp_path / "TSLA.csv"
    tsla_file.write_text(sample_tsla_csv)
    
    return tmp_path


# ==========================================
# Test Initialization
# ==========================================

class TestInitialization:
    """Tests for LocalDataFetcher initialization."""
    
    def test_loads_data_on_init(self, data_dir):
        """Test that data is loaded during initialization."""
        fetcher = LocalDataFetcher(
            data_dir=str(data_dir), 
            ticker='AAPL', 
            verbose=False
        )
        
        assert fetcher._data is not None
        assert not fetcher._data.empty
        assert fetcher.ticker == 'AAPL'
    
    def test_no_load_without_ticker(self, data_dir):
        """Test that no data is loaded if ticker not specified."""
        fetcher = LocalDataFetcher(
            data_dir=str(data_dir), 
            verbose=False
        )
        
        assert fetcher._data is None
    
    def test_implements_base_interface(self, data_dir):
        """Test that LocalDataFetcher implements BaseDataFetcher."""
        fetcher = LocalDataFetcher(
            data_dir=str(data_dir), 
            ticker='AAPL', 
            verbose=False
        )
        assert isinstance(fetcher, BaseDataFetcher)
    
    def test_nonexistent_ticker(self, data_dir):
        """Test loading non-existent ticker."""
        fetcher = LocalDataFetcher(
            data_dir=str(data_dir), 
            ticker='INVALID', 
            verbose=False
        )
        
        assert fetcher._data is None
        assert not fetcher.is_available()


# ==========================================
# Test Data Loading
# ==========================================

class TestDataLoading:
    """Tests for data loading functionality."""
    
    def test_load_ticker_switches_data(self, data_dir):
        """Test that load_ticker switches to new ticker."""
        fetcher = LocalDataFetcher(
            data_dir=str(data_dir), 
            ticker='AAPL', 
            verbose=False
        )
        
        assert fetcher.ticker == 'AAPL'
        
        fetcher.load_ticker('TSLA')
        
        assert fetcher.ticker == 'TSLA'
        assert len(fetcher._data) == 3  # TSLA has 3 rows
    
    def test_data_has_correct_columns(self, data_dir):
        """Test that loaded data has required columns."""
        fetcher = LocalDataFetcher(
            data_dir=str(data_dir), 
            ticker='AAPL', 
            verbose=False
        )
        
        assert 'open' in fetcher._data.columns
        assert 'high' in fetcher._data.columns
        assert 'low' in fetcher._data.columns
        assert 'close' in fetcher._data.columns
        assert 'volume' in fetcher._data.columns
    
    def test_index_is_datetime(self, data_dir):
        """Test that index is datetime."""
        fetcher = LocalDataFetcher(
            data_dir=str(data_dir), 
            ticker='AAPL', 
            verbose=False
        )
        
        assert isinstance(fetcher._data.index, pd.DatetimeIndex)
        assert fetcher._data.index.tzinfo is not None
    
    def test_data_sorted_by_timestamp(self, data_dir):
        """Test that data is sorted by timestamp."""
        fetcher = LocalDataFetcher(
            data_dir=str(data_dir), 
            ticker='AAPL', 
            verbose=False
        )
        
        assert fetcher._data.index.is_monotonic_increasing


# ==========================================
# Test get_latest_bars
# ==========================================

class TestGetLatestBars:
    """Tests for get_latest_bars method."""
    
    def test_returns_dataframe(self, data_dir):
        """Test that get_latest_bars returns DataFrame."""
        fetcher = LocalDataFetcher(
            data_dir=str(data_dir), 
            ticker='AAPL', 
            verbose=False
        )
        
        df = fetcher.get_latest_bars(lookback_minutes=60)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
    
    def test_end_dt_filters_future_data(self, data_dir):
        """Test that end_dt excludes future data."""
        fetcher = LocalDataFetcher(
            data_dir=str(data_dir), 
            ticker='AAPL', 
            verbose=False
        )
        
        # Set end_dt to 09:45 - should only get first 4 bars
        end_dt = datetime(2024, 12, 11, 9, 45, tzinfo=timezone.utc)
        
        df = fetcher.get_latest_bars(lookback_minutes=60, end_dt=end_dt)
        
        # All data should be <= end_dt
        assert df.index.max() <= end_dt
    
    def test_lookback_minutes_respected(self, data_dir):
        """Test that lookback_minutes is respected."""
        fetcher = LocalDataFetcher(
            data_dir=str(data_dir), 
            ticker='AAPL', 
            verbose=False
        )
        
        end_dt = datetime(2024, 12, 11, 10, 15, tzinfo=timezone.utc)
        
        # 15 minute lookback
        df = fetcher.get_latest_bars(lookback_minutes=15, end_dt=end_dt)
        
        expected_start = end_dt - timedelta(minutes=15)
        assert df.index.min() >= expected_start
    
    def test_no_data_in_range(self, data_dir):
        """Test when no data exists in requested range."""
        fetcher = LocalDataFetcher(
            data_dir=str(data_dir), 
            ticker='AAPL', 
            verbose=False
        )
        
        # Set end_dt before any data exists
        end_dt = datetime(2024, 12, 10, 9, 0, tzinfo=timezone.utc)
        
        df = fetcher.get_latest_bars(lookback_minutes=60, end_dt=end_dt)
        
        assert df.empty
    
    def test_none_end_dt_uses_latest(self, data_dir):
        """Test that None end_dt uses latest available data."""
        fetcher = LocalDataFetcher(
            data_dir=str(data_dir), 
            ticker='AAPL', 
            verbose=False
        )
        
        df = fetcher.get_latest_bars(lookback_minutes=60, end_dt=None)
        
        assert not df.empty
        # Should include the last bar (10:15)
        assert df.index.max().hour == 10
        assert df.index.max().minute == 15
    
    def test_switches_ticker_if_different(self, data_dir):
        """Test that ticker parameter switches data."""
        fetcher = LocalDataFetcher(
            data_dir=str(data_dir), 
            ticker='AAPL', 
            verbose=False
        )
        
        # Request TSLA data
        df = fetcher.get_latest_bars(ticker='TSLA', lookback_minutes=60)
        
        assert fetcher.ticker == 'TSLA'
        assert len(df) == 3


# ==========================================
# Test Timeframe Resampling
# ==========================================

class TestTimeframeResampling:
    """Tests for timeframe resampling."""
    
    def test_resample_to_10min(self, data_dir):
        """Test resampling to 10-minute bars."""
        fetcher = LocalDataFetcher(
            data_dir=str(data_dir), 
            ticker='AAPL', 
            verbose=False
        )
        
        df_5min = fetcher.get_latest_bars(
            lookback_minutes=60,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute)
        )
        
        df_10min = fetcher.get_latest_bars(
            lookback_minutes=60,
            timeframe=TimeFrame(10, TimeFrameUnit.Minute)
        )
        
        # 10-min bars should have fewer rows than 5-min
        assert len(df_10min) < len(df_5min)
    
    def test_resampled_ohlc_correct(self, data_dir):
        """Test that resampled OHLC values are valid."""
        fetcher = LocalDataFetcher(
            data_dir=str(data_dir), 
            ticker='AAPL', 
            verbose=False
        )
        
        df = fetcher.get_latest_bars(
            lookback_minutes=60,
            timeframe=TimeFrame(10, TimeFrameUnit.Minute)
        )
        
        if not df.empty:
            # High >= all other prices
            assert (df['high'] >= df['open']).all()
            assert (df['high'] >= df['close']).all()
            # Low <= all other prices
            assert (df['low'] <= df['open']).all()
            assert (df['low'] <= df['close']).all()


# ==========================================
# Test Other Methods
# ==========================================

class TestOtherMethods:
    """Tests for other methods."""
    
    def test_get_latest_price(self, data_dir):
        """Test get_latest_price returns correct value."""
        fetcher = LocalDataFetcher(
            data_dir=str(data_dir), 
            ticker='AAPL', 
            verbose=False
        )
        
        price = fetcher.get_latest_price()
        
        # Should be the last close (248.80)
        assert price == pytest.approx(248.80, rel=0.01)
    
    def test_get_latest_price_no_data(self, data_dir):
        """Test get_latest_price returns 0 when no data."""
        fetcher = LocalDataFetcher(
            data_dir=str(data_dir), 
            verbose=False
        )
        
        price = fetcher.get_latest_price()
        
        assert price == 0.0
    
    def test_get_available_tickers(self, data_dir):
        """Test get_available_tickers finds CSV files."""
        fetcher = LocalDataFetcher(
            data_dir=str(data_dir), 
            verbose=False
        )
        
        tickers = fetcher.get_available_tickers()
        
        assert 'AAPL' in tickers
        assert 'TSLA' in tickers
        assert len(tickers) == 2
    
    def test_get_date_range(self, data_dir):
        """Test get_date_range returns correct range."""
        fetcher = LocalDataFetcher(
            data_dir=str(data_dir), 
            ticker='AAPL', 
            verbose=False
        )
        
        date_range = fetcher.get_date_range()
        
        assert date_range is not None
        assert 'start' in date_range
        assert 'end' in date_range
        assert date_range['start'] < date_range['end']
    
    def test_is_available(self, data_dir):
        """Test is_available returns correct status."""
        fetcher = LocalDataFetcher(
            data_dir=str(data_dir), 
            ticker='AAPL', 
            verbose=False
        )
        
        assert fetcher.is_available() == True
        
        fetcher2 = LocalDataFetcher(
            data_dir=str(data_dir), 
            verbose=False
        )
        
        assert fetcher2.is_available() == False


# ==========================================
# Test Edge Cases
# ==========================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_malformed_csv(self, tmp_path):
        """Test handling of malformed CSV."""
        malformed_file = tmp_path / "BAD.csv"
        malformed_file.write_text("not,valid,csv\ndata")
        
        fetcher = LocalDataFetcher(
            data_dir=str(tmp_path), 
            ticker='BAD', 
            verbose=False
        )
        
        assert not fetcher.is_available()
    
    def test_missing_columns(self, tmp_path):
        """Test handling of CSV with missing required columns."""
        incomplete_csv = """symbol,timestamp,open,close
INCOMPLETE,2024-12-11 09:30:00+00:00,100.0,101.0"""
        
        incomplete_file = tmp_path / "INCOMPLETE.csv"
        incomplete_file.write_text(incomplete_csv)
        
        fetcher = LocalDataFetcher(
            data_dir=str(tmp_path), 
            ticker='INCOMPLETE', 
            verbose=False
        )
        
        assert not fetcher.is_available()
    
    def test_empty_csv(self, tmp_path):
        """Test handling of empty CSV."""
        empty_file = tmp_path / "EMPTY.csv"
        empty_file.write_text("symbol,timestamp,open,high,low,close,volume\n")
        
        fetcher = LocalDataFetcher(
            data_dir=str(tmp_path), 
            ticker='EMPTY', 
            verbose=False
        )
        
        # Should load but be empty
        df = fetcher.get_latest_bars(lookback_minutes=60)
        assert df.empty
    
    def test_naive_datetime_handling(self, data_dir):
        """Test handling of naive datetime (no timezone)."""
        fetcher = LocalDataFetcher(
            data_dir=str(data_dir), 
            ticker='AAPL', 
            verbose=False
        )
        
        # Naive datetime should be treated as UTC
        end_dt = datetime(2024, 12, 11, 10, 0)  # No timezone
        
        df = fetcher.get_latest_bars(lookback_minutes=30, end_dt=end_dt)
        
        # Should not raise and should filter correctly
        assert df.index.max() <= end_dt.replace(tzinfo=timezone.utc)


# ==========================================
# Run Tests
# ==========================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])