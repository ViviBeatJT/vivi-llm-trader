# src/factory/component_factory.py

"""
Component Factory - Centralized creation of trading components.

This module provides factory functions and a registry for creating
strategies, executors, and other components in a consistent way.
"""

from dataclasses import dataclass
from typing import Dict, Any, Type, Optional, Callable
from enum import Enum


# ==========================================
# Trading Mode Enum
# ==========================================

class TradingMode(Enum):
    """Trading mode enumeration."""
    SIMULATION = 'simulation'  # Local simulation (no API)
    PAPER = 'paper'           # Paper trading (API)
    LIVE = 'live'             # Live trading (real money)


# ==========================================
# Strategy Registry
# ==========================================

@dataclass
class StrategyConfig:
    """Strategy configuration."""
    strategy_class: Type
    name: str
    description: str
    default_params: Dict[str, Any]
    chart_file: Optional[str] = None


class StrategyRegistry:
    """
    Registry for available strategies.
    
    Usage:
        # Register a strategy
        StrategyRegistry.register('my_strategy', MyStrategyClass, {...})
        
        # Create a strategy instance
        strategy = StrategyRegistry.create('my_strategy')
        
        # Get strategy info
        info = StrategyRegistry.get_info('my_strategy')
    """
    
    _strategies: Dict[str, StrategyConfig] = {}
    
    @classmethod
    def register(cls, 
                 key: str, 
                 strategy_class: Type,
                 name: str,
                 description: str,
                 default_params: Dict[str, Any],
                 chart_file: Optional[str] = None):
        """Register a strategy."""
        cls._strategies[key] = StrategyConfig(
            strategy_class=strategy_class,
            name=name,
            description=description,
            default_params=default_params,
            chart_file=chart_file
        )
    
    @classmethod
    def create(cls, key: str, **override_params) -> Any:
        """Create a strategy instance."""
        if key not in cls._strategies:
            raise ValueError(f"Unknown strategy: {key}. Available: {list(cls._strategies.keys())}")
        
        config = cls._strategies[key]
        params = {**config.default_params, **override_params}
        
        print(f"\n📊 Creating Strategy: {config.name}")
        print(f"   Description: {config.description}")
        print(f"   Parameters:")
        for k, v in params.items():
            if isinstance(v, float):
                print(f"      {k}: {v:.2f}")
            else:
                print(f"      {k}: {v}")
        
        return config.strategy_class(**params)
    
    @classmethod
    def get_info(cls, key: str) -> StrategyConfig:
        """Get strategy configuration info."""
        if key not in cls._strategies:
            raise ValueError(f"Unknown strategy: {key}")
        return cls._strategies[key]
    
    @classmethod
    def list_strategies(cls) -> Dict[str, str]:
        """List all registered strategies."""
        return {k: v.name for k, v in cls._strategies.items()}
    
    @classmethod
    def get_all_keys(cls) -> list:
        """Get all strategy keys."""
        return list(cls._strategies.keys())


# ==========================================
# Component Factory
# ==========================================

class ComponentFactory:
    """
    Factory for creating trading components.
    
    Centralizes the creation of:
    - DataFetcher
    - Executor (Simulation or Alpaca)
    - PositionManager
    - Cache
    - Visualizer
    """
    
    @staticmethod
    def create_data_fetcher(mode: TradingMode):
        """Create data fetcher based on mode."""
        from src.data_fetcher.alpaca_data_fetcher import AlpacaDataFetcher
        
        is_paper = mode in [TradingMode.SIMULATION, TradingMode.PAPER]
        return AlpacaDataFetcher(paper=is_paper)
    
    @staticmethod
    def create_executor(mode: TradingMode, finance_params: Dict[str, Any]):
        """Create executor based on mode."""
        
        if mode == TradingMode.SIMULATION:
            from src.executor.simulation_executor import SimulationExecutor
            print("🔧 Executor: Local Simulation")
            return SimulationExecutor(finance_params)
        
        elif mode == TradingMode.PAPER:
            from src.executor.alpaca_trade_executor import AlpacaExecutor
            print("🔧 Executor: Alpaca Paper Trading")
            return AlpacaExecutor(paper=True, max_allocation_rate=finance_params.get('MAX_ALLOCATION', 0.95))
        
        elif mode == TradingMode.LIVE:
            from src.executor.alpaca_trade_executor import AlpacaExecutor
            print("🔧 Executor: Alpaca LIVE Trading ⚠️")
            return AlpacaExecutor(paper=False, max_allocation_rate=finance_params.get('MAX_ALLOCATION', 0.95))
        
        else:
            raise ValueError(f"Unknown trading mode: {mode}")
    
    @staticmethod
    def create_position_manager(executor, 
                                finance_params: Dict[str, Any], 
                                data_fetcher=None):
        """Create position manager."""
        from src.manager.position_manager import PositionManager
        return PositionManager(executor, finance_params, data_fetcher=data_fetcher)
    
    @staticmethod
    def create_cache(cache_file: Optional[str] = None):
        """Create trading cache."""
        from src.cache.trading_cache import TradingCache
        return TradingCache(cache_file) if cache_file else TradingCache()
    
    @staticmethod
    def create_visualizer(ticker: str, 
                         output_file: str, 
                         auto_open: bool = True,
                         initial_capital: float = 1000.0):
        """Create chart visualizer."""
        from src.visualization.simple_chart_visualizer import SimpleChartVisualizer
        
        visualizer = SimpleChartVisualizer(
            ticker=ticker,
            output_file=output_file,
            auto_open=auto_open
        )
        visualizer.set_initial_capital(initial_capital)
        return visualizer


# ==========================================
# Register Default Strategies
# ==========================================

def register_default_strategies():
    """Register all default strategies."""
    
    # Import strategy classes
    try:
        from src.strategies.moderate_aggressive_strategy import ModerateAggressiveStrategy
        StrategyRegistry.register(
            'moderate',
            ModerateAggressiveStrategy,
            name='Moderate Aggressive Strategy',
            description='Trades near Bollinger Bands, captures more opportunities',
            default_params={
                'bb_period': 20,
                'bb_std_dev': 2.0,
                'entry_threshold': 0.85,
                'exit_threshold': 0.60,
                'stop_loss_threshold': 0.10,
                'monitor_interval_seconds': 60,
            },
            chart_file='backtest_moderate.html'
        )
    except ImportError:
        pass
    
    try:
        from src.strategies.trend_aware_strategy import TrendAwareStrategy
        StrategyRegistry.register(
            'trend_aware',
            TrendAwareStrategy,
            name='Trend Aware Strategy',
            description='ADX-based trend detection with adaptive behavior',
            default_params={
                'bb_period': 20,
                'bb_std_dev': 2.0,
                'adx_period': 14,
                'adx_trend_threshold': 25,
                'adx_range_threshold': 20,
                'ema_fast_period': 12,
                'ema_slow_period': 26,
                'mean_reversion_entry': 0.85,
                'mean_reversion_exit': 0.60,
                'trend_entry_pullback': 0.50,
                'trend_exit_profit': 0.03,
                'stop_loss_threshold': 0.01,
                'monitor_interval_seconds': 60,
                'max_history_bars': 500
            },
            chart_file='backtest_trend_aware.html'
        )
    except ImportError:
        pass
    
    try:
        from src.strategies.mean_reversion_strategy import MeanReversionStrategy
        StrategyRegistry.register(
            'mean_reversion',
            MeanReversionStrategy,
            name='Mean Reversion Strategy',
            description='Classic Bollinger Band + RSI mean reversion',
            default_params={
                'bb_period': 20,
                'bb_std_dev': 2,
                'rsi_window': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'max_history_bars': 500
            },
            chart_file='backtest_mean_reversion.html'
        )
    except ImportError:
        pass


# Auto-register on import
register_default_strategies()