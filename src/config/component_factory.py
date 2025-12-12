# src/factory/component_factory.py

"""
Component Factory - Centralized creation of trading components.

整合配置系统:
    from src.config.trading_config import TradingConfig, get_full_config
    from src.factory.component_factory import ComponentFactory
    
    config = get_full_config(ticker='TSLA', mode='paper')
    components = ComponentFactory.create_all_from_config(config)
"""

from dataclasses import dataclass
from typing import Dict, Any, Type, Optional, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from src.config.trading_config import TradingConfig


# ==========================================
# Trading Mode Enum
# ==========================================

class TradingMode(Enum):
    """Trading mode enumeration."""
    SIMULATION = 'simulation'
    PAPER = 'paper'
    LIVE = 'live'


# ==========================================
# Strategy Registry
# ==========================================

@dataclass
class StrategyInfo:
    """Strategy metadata."""
    strategy_class: Type
    name: str
    description: str
    config_key: str  # 对应 TradingConfig 中的配置键


class StrategyRegistry:
    """
    Registry for available strategies.

    Usage:
        # Create from config (推荐)
        strategy = StrategyRegistry.create_from_config(config)

        # Create with manual params
        strategy = StrategyRegistry.create('simple_trend', stop_loss=0.02)
    """

    _strategies: Dict[str, StrategyInfo] = {}

    @classmethod
    def register(cls,
                 key: str,
                 strategy_class: Type,
                 name: str,
                 description: str,
                 config_key: str = None):
        """Register a strategy."""
        cls._strategies[key] = StrategyInfo(
            strategy_class=strategy_class,
            name=name,
            description=description,
            config_key=config_key or key
        )

    @classmethod
    def create(cls, key: str, **params) -> Any:
        """Create a strategy instance with manual params."""
        if key not in cls._strategies:
            raise ValueError(
                f"Unknown strategy: {key}. Available: {list(cls._strategies.keys())}")

        info = cls._strategies[key]

        print(f"\n📊 Creating Strategy: {info.name}")
        print(f"   Description: {info.description}")

        return info.strategy_class(**params)

    @classmethod
    def create_from_config(cls, config: 'TradingConfig') -> Any:
        """
        Create strategy from TradingConfig object.

        Args:
            config: TradingConfig 配置对象

        Returns:
            策略实例
        """
        strategy_key = config.system.strategy

        if strategy_key not in cls._strategies:
            raise ValueError(
                f"Unknown strategy: {strategy_key}. Available: {list(cls._strategies.keys())}")

        info = cls._strategies[strategy_key]
        strategy_params = config.get_strategy_config()

        print(f"\n📊 Creating Strategy: {info.name}")
        print(f"   Description: {info.description}")
        print(f"   Config: TradingConfig.{info.config_key}")

        return info.strategy_class(**strategy_params)

    @classmethod
    def get_info(cls, key: str) -> StrategyInfo:
        """Get strategy info."""
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

    可以单独创建组件，也可以从 TradingConfig 一次性创建所有组件。
    """

    @staticmethod
    def create_data_fetcher(mode: TradingMode,
                            use_local: bool = False,
                            local_data_dir: str = 'data/'):
        """Create data fetcher based on mode."""
        if use_local:
            from src.data_fetcher.local_data_fetcher import LocalDataFetcher
            print("🔧 DataFetcher: Local CSV files")
            return LocalDataFetcher(data_dir=local_data_dir, verbose=True)
        else:
            from src.data_fetcher.alpaca_data_fetcher import AlpacaDataFetcher
            is_paper = mode in [TradingMode.SIMULATION, TradingMode.PAPER]
            print(
                f"🔧 DataFetcher: Alpaca API ({'Paper' if is_paper else 'Live'})")
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

    @staticmethod
    def create_strategy_from_config(config: 'TradingConfig'):
        """
        Shortcut to create strategy from config.

        Args:
            config: TradingConfig 对象

        Returns:
            策略实例
        """
        return StrategyRegistry.create_from_config(config)

    @classmethod
    def create_all_from_config(cls, config: 'TradingConfig') -> Dict[str, Any]:
        """
        从 TradingConfig 创建所有组件

        Args:
            config: TradingConfig 配置对象

        Returns:
            包含所有组件的字典
        """
        from pathlib import Path

        mode = TradingMode(config.system.mode)
        finance_params = config.finance.to_dict()

        # Data fetcher
        data_fetcher = cls.create_data_fetcher(mode)

        # Executor
        executor = cls.create_executor(mode, finance_params)

        # Position manager
        position_manager = cls.create_position_manager(
            executor,
            finance_params,
            data_fetcher=data_fetcher if mode != TradingMode.SIMULATION else None
        )

        # Strategy
        strategy = cls.create_strategy_from_config(config)

        # Visualizer
        visualizer = None
        chart_file = None
        if config.system.enable_chart:
            charts_dir = Path(config.system.output_dir) / "charts"
            charts_dir.mkdir(parents=True, exist_ok=True)
            chart_file = str(
                charts_dir / f"{config.system.ticker}_{config.system.strategy}_{config.system.mode}.html")

            visualizer = cls.create_visualizer(
                ticker=config.system.ticker,
                output_file=chart_file,
                auto_open=config.system.auto_open_browser,
                initial_capital=config.finance.initial_capital
            )

        return {
            'data_fetcher': data_fetcher,
            'executor': executor,
            'position_manager': position_manager,
            'strategy': strategy,
            'visualizer': visualizer,
            'chart_file': chart_file,
        }


# ==========================================
# Register Default Strategies
# ==========================================

def register_default_strategies():
    """Register all default strategies."""

    # Moderate Aggressive Strategy
    try:
        from src.strategies.moderate_aggressive_strategy import ModerateAggressiveStrategy
        StrategyRegistry.register(
            'moderate',
            ModerateAggressiveStrategy,
            name='Moderate Aggressive Strategy',
            description='Trades near Bollinger Bands, captures more opportunities',
            config_key='moderate'
        )
    except ImportError:
        pass

    # Churn Moderate Strategy
    try:
        from src.strategies.churn_moderate_aggressive_strategy import ChurnModerateAggressiveStrategy
        StrategyRegistry.register(
            'churn_moderate',
            ChurnModerateAggressiveStrategy,
            name='Churn Moderate Aggressive Strategy',
            description='Churning version of moderate strategy',
            config_key='churn_moderate'
        )
    except ImportError:
        pass

    # Simple Up Trend Strategy
    try:
        from src.strategies.simple_uptrend_strategy import SimpleUpTrendStrategy
        StrategyRegistry.register(
            'up_trend_aware',
            SimpleUpTrendStrategy,
            name='Simple Up Trend Strategy',
            description='ADX-based up trend detection',
            config_key='up_trend_aware'
        )
    except ImportError:
        pass

    # Mean Reversion Strategy
    try:
        from src.strategies.mean_reversion_strategy import MeanReversionStrategy
        StrategyRegistry.register(
            'mean_reversion',
            MeanReversionStrategy,
            name='Mean Reversion Strategy',
            description='Classic Bollinger Band + RSI mean reversion',
            config_key='mean_reversion'
        )
    except ImportError:
        pass


# Auto-register on import
register_default_strategies()


# ==========================================
# Convenience Functions
# ==========================================

def create_trading_engine(config: 'TradingConfig', mode: str = None):
    """
    便捷函数：从 TradingConfig 创建完整的 TradingEngine

    Args:
        config: TradingConfig 配置对象
        mode: 覆盖配置中的模式 ('backtest' or 'live')

    Returns:
        TradingEngine 实例
    """
    from src.engine.trading_engine import TradingEngine
    return TradingEngine.from_config(config, mode=mode)
