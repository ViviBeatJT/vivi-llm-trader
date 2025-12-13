# src/data_fetcher/local_data_fetcher.py

"""
Local Data Fetcher - Reads historical data from local CSV files.

This fetcher is designed for backtesting with pre-downloaded data.
It provides the same interface as AlpacaDataFetcher but reads from local files.

Data is loaded once during initialization for efficiency.

Expected CSV format:
    symbol,timestamp,open,high,low,close,volume,trade_count,vwap
    AAPL,2024-12-11 00:00:00+00:00,247.77,247.77,247.77,247.77,142.0,9.0,247.77
    ...

Directory structure:
    data/
    ├── AAPL.csv
    ├── TSLA.csv
    └── ...

Usage:
    fetcher = LocalDataFetcher(data_dir='data/', ticker='AAPL')
    df = fetcher.get_latest_bars(lookback_minutes=300, end_dt=some_datetime)
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List
from pathlib import Path
import pandas as pd

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from src.data_fetcher.base_data_fetcher import BaseDataFetcher


class LocalDataFetcher(BaseDataFetcher):
    """
    Local data fetcher that reads from CSV files.
    
    Data is loaded once during initialization for maximum efficiency.
    Supports filtering by date range to simulate API behavior for backtesting.
    """
    
    def __init__(self, 
                 data_dir: str = 'data/',
                 ticker: str = None,
                 file_pattern: str = '{ticker}_1Min_1year_data.csv',
                 verbose: bool = True):
        """
        Initialize the local data fetcher and load data.
        
        Args:
            data_dir: Directory containing CSV files
            ticker: Ticker to load (required for data operations)
            file_pattern: Pattern for file names. Use {ticker} as placeholder.
            verbose: Whether to print status messages
        """
        self.data_dir = Path(data_dir)
        self.file_pattern = file_pattern
        self.verbose = verbose
        self.ticker = ticker
        
        # Data storage - loaded once
        self._data: Optional[pd.DataFrame] = None
        self._latest_data: Optional[pd.DataFrame] = None
        
        if verbose:
            print(f"📁 LocalDataFetcher initialized:")
            print(f"   Data directory: {self.data_dir}")
        
        # Load data immediately if ticker specified
        if ticker:
            self._load_data(ticker)
    
    def _get_file_path(self, ticker: str) -> Path:
        """Get the file path for a ticker."""
        filename = self.file_pattern.format(ticker=ticker)
        return self.data_dir / filename
    
    def _load_data(self, ticker: str) -> bool:
        """
        Load data from CSV file for a ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            True if data loaded successfully
        """
        file_path = self._get_file_path(ticker)
        
        if not file_path.exists():
            if self.verbose:
                print(f"❌ Data file not found: {file_path}")
            return False
        
        try:
            # Read CSV
            df = pd.read_csv(file_path)
            
            # Parse timestamp column
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
            # Set timestamp as index
            df = df.set_index('timestamp')
            
            # Sort by timestamp
            df = df.sort_index()
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                if self.verbose:
                    print(f"⚠️ Missing columns {missing_cols} in {file_path}")
                return False
            
            # Store data
            self._data = df
            self.ticker = ticker
            
            if self.verbose:
                print(f"✅ Loaded {ticker}: {len(df)} bars "
                      f"({df.index.min().strftime('%Y-%m-%d')} to "
                      f"{df.index.max().strftime('%Y-%m-%d')})")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error loading {file_path}: {e}")
            return False
    
    def load_ticker(self, ticker: str) -> bool:
        """
        Load data for a specific ticker (replaces current data).
        
        Args:
            ticker: Stock symbol to load
            
        Returns:
            True if loaded successfully
        """
        return self._load_data(ticker)
    
    def _resample_data(self, 
                       df: pd.DataFrame, 
                       timeframe: TimeFrame) -> pd.DataFrame:
        """
        Resample data to a different timeframe.
        
        Args:
            df: Source DataFrame with OHLCV data
            timeframe: Target timeframe
            
        Returns:
            Resampled DataFrame
        """
        if df.empty:
            return df
        
        # Convert timeframe to pandas resample string
        if timeframe.unit == TimeFrameUnit.Minute:
            freq = f'{timeframe.amount}min'
        elif timeframe.unit == TimeFrameUnit.Hour:
            freq = f'{timeframe.amount}h'
        elif timeframe.unit == TimeFrameUnit.Day:
            freq = f'{timeframe.amount}D'
        elif timeframe.unit == TimeFrameUnit.Week:
            freq = f'{timeframe.amount}W'
        elif timeframe.unit == TimeFrameUnit.Month:
            freq = f'{timeframe.amount}ME'
        else:
            freq = f'{timeframe.amount}min'
        
        # Resample OHLCV
        resampled = df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def get_latest_bars(self,
                        ticker: str = None,
                        lookback_minutes: int = 60,
                        timeframe: TimeFrame = None,
                        end_dt: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get historical bars for a ticker.
        
        Simulates API behavior for backtesting:
        - Returns data up to end_dt (not including future data)
        - Looks back lookback_minutes from end_dt
        
        Args:
            ticker: Stock symbol (uses loaded ticker if None)
            lookback_minutes: How many minutes of data to look back
            timeframe: Bar timeframe. If None, uses raw data (5min)
            end_dt: End datetime. If None, returns most recent data.
            
        Returns:
            DataFrame with OHLCV data
        """
        # Load ticker if different from current
        if ticker and ticker != self.ticker:
            self._load_data(ticker)
        
        if self._data is None or self._data.empty:
            return pd.DataFrame()
        
        df = self._data
        
        # Default timeframe to 5 minutes
        if timeframe is None:
            timeframe = TimeFrame(5, TimeFrameUnit.Minute)
        
        # Resample if needed (source data is 5-minute bars)
        source_minutes = 5
        target_minutes = timeframe.amount
        if timeframe.unit == TimeFrameUnit.Hour:
            target_minutes = timeframe.amount * 60
        elif timeframe.unit == TimeFrameUnit.Day:
            target_minutes = timeframe.amount * 60 * 24
        
        if target_minutes != source_minutes:
            df = self._resample_data(df, timeframe)
        
        # Determine end time
        if end_dt is None:
            end_time = df.index.max()
        else:
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            end_time = end_dt
        
        # Calculate start time
        start_time = end_time - timedelta(minutes=lookback_minutes)
        
        # Filter data
        mask = (df.index >= start_time) & (df.index <= end_time)
        filtered_df = df.loc[mask].copy()
        
        # Select only OHLCV columns
        output_cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [c for c in output_cols if c in filtered_df.columns]
        filtered_df = filtered_df[available_cols]
        self._latest_data = filtered_df
        
        return filtered_df
    
    def get_latest_price(self, ticker: str = None,  current_time: Optional[datetime] = None) -> float:
        """
        Get the most recent closing price for a ticker.
        
        Args:
            ticker: Stock symbol (uses loaded ticker if None)
            
        Returns:
            Latest closing price, or 0.0 if unavailable
        """
        if ticker and ticker != self.ticker:
            self._load_data(ticker)
        
        if self._latest_data is None or self._latest_data.empty:
            return 0.0
        
        return float(self._latest_data['close'].iloc[-1])
    
    def get_available_tickers(self) -> List[str]:
        """
        Get list of tickers available in the data directory.
        
        Returns:
            List of ticker symbols
        """
        if not self.data_dir.exists():
            return []
        
        tickers = []
        for file_path in self.data_dir.glob('*.csv'):
            tickers.append(file_path.stem)
        
        return sorted(tickers)
    
    def get_date_range(self, ticker: str = None) -> Optional[Dict[str, datetime]]:
        """
        Get the available date range for a ticker.
        
        Args:
            ticker: Stock symbol (uses loaded ticker if None)
            
        Returns:
            Dict with 'start' and 'end' datetime keys
        """
        if ticker and ticker != self.ticker:
            self._load_data(ticker)
        
        if self._data is None or self._data.empty:
            return None
        
        return {
            'start': self._data.index.min().to_pydatetime(),
            'end': self._data.index.max().to_pydatetime()
        }
    
    def is_available(self) -> bool:
        """Check if data is loaded."""
        return self._data is not None and not self._data.empty


# ==========================================
# Main (for testing)
# ==========================================

if __name__ == '__main__':
    # Test the local data fetcher
    print("--- Available Tickers ---")
    fetcher = LocalDataFetcher(data_dir='data/', verbose=True)
    tickers = fetcher.get_available_tickers()
    print(f"Found {len(tickers)} tickers: {tickers[:10]}...")
    
    if tickers:
        ticker = tickers[0]
        
        # Create fetcher with specific ticker (loads on init)
        fetcher = LocalDataFetcher(data_dir='data/', ticker=ticker, verbose=True)
        
        print(f"\n--- Date Range ---")
        date_range = fetcher.get_date_range()
        if date_range:
            print(f"Start: {date_range['start']}")
            print(f"End: {date_range['end']}")
        
        print(f"\n--- Latest Price ---")
        price = fetcher.get_latest_price()
        print(f"Price: ${price:.2f}")
        
        print(f"\n--- Latest Bars ---")
        df = fetcher.get_latest_bars(lookback_minutes=300)
        print(f"Shape: {df.shape}")
        if not df.empty:
            print(df.tail())