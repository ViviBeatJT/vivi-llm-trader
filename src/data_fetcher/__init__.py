# src/data_fetcher/__init__.py

"""
Data Fetcher Module

Provides interfaces for fetching market data from different sources:
- BaseDataFetcher: Abstract interface
- AlpacaDataFetcher: Fetches from Alpaca API (live/paper)
- LocalDataFetcher: Fetches from local CSV files (backtesting)

Usage:
    # For live trading
    from src.data_fetcher import AlpacaDataFetcher
    fetcher = AlpacaDataFetcher(paper=True)
    
    # For backtesting with local data
    from src.data_fetcher import LocalDataFetcher
    fetcher = LocalDataFetcher(data_dir='data/', ticker='AAPL')
    
    # Both have the same interface
    df = fetcher.get_latest_bars(lookback_minutes=300, end_dt=some_time)
"""

from src.data_fetcher.base_data_fetcher import BaseDataFetcher
from src.data_fetcher.local_data_fetcher import LocalDataFetcher

# AlpacaDataFetcher is imported conditionally (requires API keys)
try:
    from src.data_fetcher.alpaca_data_fetcher import AlpacaDataFetcher
except ImportError:
    AlpacaDataFetcher = None

__all__ = [
    'BaseDataFetcher',
    'LocalDataFetcher',
    'AlpacaDataFetcher',
]