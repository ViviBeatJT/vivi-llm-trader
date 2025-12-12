# src/data_fetcher/base_data_fetcher.py

"""
Base Data Fetcher Interface

This abstract base class defines the interface that all data fetchers must implement.
This ensures consistent behavior across different data sources:
- AlpacaDataFetcher: Fetches from Alpaca API
- LocalDataFetcher: Fetches from local CSV files
- Future: Could add Yahoo, Polygon, etc.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import pandas as pd

from alpaca.data.timeframe import TimeFrame


class BaseDataFetcher(ABC):
    """
    Abstract base class for data fetchers.
    
    All data fetchers must implement these methods to ensure
    consistent behavior across the trading system.
    """
    
    @abstractmethod
    def get_latest_bars(self,
                        ticker: str,
                        lookback_minutes: int = 60,
                        timeframe: TimeFrame = None,
                        end_dt: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch historical bar data for a ticker.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL', 'TSLA')
            lookback_minutes: How many minutes of data to fetch
            timeframe: Bar timeframe (e.g., 1min, 5min, 1hour)
            end_dt: End datetime for the data. If None, use current time.
                   For backtesting, this simulates "current time".
        
        Returns:
            pd.DataFrame with columns:
                - open: Opening price
                - high: High price
                - low: Low price
                - close: Closing price
                - volume: Trading volume
            Index should be datetime (timezone-aware, UTC)
            
            Returns empty DataFrame if no data available.
        """
        pass
    
    @abstractmethod
    def get_latest_price(self, ticker: str) -> float:
        """
        Get the most recent closing price for a ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            float: Latest closing price, or 0.0 if unavailable
        """
        pass
    
    def get_available_tickers(self) -> List[str]:
        """
        Get list of available tickers from this data source.
        
        Returns:
            List of ticker symbols. Default implementation returns empty list.
        """
        return []
    
    def get_date_range(self, ticker: str) -> Optional[Dict[str, datetime]]:
        """
        Get the available date range for a ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict with 'start' and 'end' datetime keys, or None if unknown
        """
        return None
    
    def is_available(self) -> bool:
        """
        Check if this data fetcher is available/configured.
        
        Returns:
            bool: True if the data source is accessible
        """
        return True
    
    def _format_timestamp(self, dt: Optional[datetime]) -> str:
        """Helper to format timestamp for logging."""
        if dt is None:
            return "now"
        return dt.strftime('%Y-%m-%d %H:%M UTC')