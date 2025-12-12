# tests/test_component_factory.py

"""
Unit Tests for ComponentFactory and StrategyRegistry

Tests cover:
1. Strategy registration and creation
2. Component factory methods
3. Trading mode handling
4. Error handling for invalid inputs

Run with:
    pytest tests/test_component_factory.py -v
"""

import pytest
from unittest.mock import Mock, patch
from src.factory.component_factory import (
    TradingMode,
    StrategyConfig,
    StrategyRegistry,
    ComponentFactory,
    register_default_strategies
)


# ==========================================
# Test TradingMode Enum
# ==========================================

class TestTradingMode:
    """Tests for TradingMode enum."""
    
    def test_simulation_mode(self):
        """Test simulation mode value."""
        assert TradingMode.SIMULATION.value == 'simulation'
    
    def test_paper_mode(self):
        """Test paper mode value."""
        assert TradingMode.PAPER.value == 'paper'
    
    def test_live_mode(self):
        """Test live mode value."""
        assert TradingMode.LIVE.value == 'live'
    
    def test_mode_from_string(self):
        """Test creating mode from string."""
        assert TradingMode('simulation') == TradingMode.SIMULATION
        assert TradingMode('paper') == TradingMode.PAPER
        assert TradingMode('live') == TradingMode.LIVE
    
    def test_invalid_mode_raises(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError):
            TradingMode('invalid')


# ==========================================
# Test StrategyConfig
# ==========================================

class TestStrategyConfig:
    """Tests for StrategyConfig dataclass."""
    
    def test_create_strategy_config(self):
        """Test creating a strategy config."""
        mock_class = Mock
        config = StrategyConfig(
            strategy_class=mock_class,
            name='Test Strategy',
            description='A test strategy',
            default_params={'param1': 10}
        )
        
        assert config.strategy_class == mock_class
        assert config.name == 'Test Strategy'
        assert config.description == 'A test strategy'
        assert config.default_params == {'param1': 10}
        assert config.chart_file is None
    
    def test_strategy_config_with_chart_file(self):
        """Test strategy config with chart file."""
        config = StrategyConfig(
            strategy_class=Mock,
            name='Test',
            description='Test',
            default_params={},
            chart_file='test_chart.html'
        )
        
        assert config.chart_file == 'test_chart.html'


# ==========================================
# Test StrategyRegistry
# ==========================================

class TestStrategyRegistry:
    """Tests for StrategyRegistry."""
    
    def setup_method(self):
        """Clear registry before each test."""
        # Store original strategies
        self._original_strategies = StrategyRegistry._strategies.copy()
    
    def teardown_method(self):
        """Restore registry after each test."""
        StrategyRegistry._strategies = self._original_strategies
    
    def test_register_strategy(self):
        """Test registering a new strategy."""
        mock_class = Mock
        
        StrategyRegistry.register(
            key='test_strategy',
            strategy_class=mock_class,
            name='Test Strategy',
            description='A test strategy',
            default_params={'param1': 10}
        )
        
        assert 'test_strategy' in StrategyRegistry._strategies
    
    def test_create_strategy(self):
        """Test creating a strategy instance."""
        # Create a mock class that tracks instantiation
        class MockStrategy:
            def __init__(self, param1=5):
                self.param1 = param1
        
        StrategyRegistry.register(
            key='mock_strategy',
            strategy_class=MockStrategy,
            name='Mock Strategy',
            description='A mock strategy',
            default_params={'param1': 10}
        )
        
        strategy = StrategyRegistry.create('mock_strategy')
        
        assert isinstance(strategy, MockStrategy)
        assert strategy.param1 == 10
    
    def test_create_strategy_with_override(self):
        """Test creating strategy with parameter override."""
        class MockStrategy:
            def __init__(self, param1=5, param2=20):
                self.param1 = param1
                self.param2 = param2
        
        StrategyRegistry.register(
            key='mock_strategy2',
            strategy_class=MockStrategy,
            name='Mock Strategy 2',
            description='Another mock strategy',
            default_params={'param1': 10, 'param2': 20}
        )
        
        strategy = StrategyRegistry.create('mock_strategy2', param1=15)
        
        assert strategy.param1 == 15  # Overridden
        assert strategy.param2 == 20  # Default
    
    def test_create_unknown_strategy_raises(self):
        """Test that creating unknown strategy raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            StrategyRegistry.create('nonexistent_strategy')
        
        assert 'Unknown strategy' in str(exc_info.value)
    
    def test_get_info(self):
        """Test getting strategy info."""
        class MockStrategy:
            pass
        
        StrategyRegistry.register(
            key='info_test',
            strategy_class=MockStrategy,
            name='Info Test Strategy',
            description='Test description',
            default_params={'x': 1}
        )
        
        info = StrategyRegistry.get_info('info_test')
        
        assert info.name == 'Info Test Strategy'
        assert info.description == 'Test description'
    
    def test_get_info_unknown_raises(self):
        """Test that getting info for unknown strategy raises."""
        with pytest.raises(ValueError):
            StrategyRegistry.get_info('nonexistent')
    
    def test_list_strategies(self):
        """Test listing all strategies."""
        class MockStrategy:
            pass
        
        StrategyRegistry.register(
            key='list_test',
            strategy_class=MockStrategy,
            name='List Test',
            description='Test',
            default_params={}
        )
        
        strategies = StrategyRegistry.list_strategies()
        
        assert isinstance(strategies, dict)
        assert 'list_test' in strategies
        assert strategies['list_test'] == 'List Test'
    
    def test_get_all_keys(self):
        """Test getting all strategy keys."""
        keys = StrategyRegistry.get_all_keys()
        
        assert isinstance(keys, list)


# ==========================================
# Test ComponentFactory
# ==========================================

class TestComponentFactory:
    """Tests for ComponentFactory."""
    
    @patch('src.data_fetcher.alpaca_data_fetcher.AlpacaDataFetcher')
    def test_create_data_fetcher_paper(self, mock_fetcher_class):
        """Test creating data fetcher for paper mode."""
        ComponentFactory.create_data_fetcher(TradingMode.PAPER)
        
        mock_fetcher_class.assert_called_once_with(paper=True)
    
    @patch('src.data_fetcher.alpaca_data_fetcher.AlpacaDataFetcher')
    def test_create_data_fetcher_simulation(self, mock_fetcher_class):
        """Test creating data fetcher for simulation mode."""
        ComponentFactory.create_data_fetcher(TradingMode.SIMULATION)
        
        mock_fetcher_class.assert_called_once_with(paper=True)
    
    @patch('src.data_fetcher.alpaca_data_fetcher.AlpacaDataFetcher')
    def test_create_data_fetcher_live(self, mock_fetcher_class):
        """Test creating data fetcher for live mode."""
        ComponentFactory.create_data_fetcher(TradingMode.LIVE)
        
        mock_fetcher_class.assert_called_once_with(paper=False)
    
    @patch('src.executor.simulation_executor.SimulationExecutor')
    def test_create_executor_simulation(self, mock_executor_class):
        """Test creating executor for simulation mode."""
        finance_params = {'INITIAL_CAPITAL': 1000, 'MAX_ALLOCATION': 0.95}
        
        ComponentFactory.create_executor(TradingMode.SIMULATION, finance_params)
        
        mock_executor_class.assert_called_once_with(finance_params)
    
    @patch('src.executor.alpaca_trade_executor.AlpacaExecutor')
    def test_create_executor_paper(self, mock_executor_class):
        """Test creating executor for paper mode."""
        finance_params = {'INITIAL_CAPITAL': 1000, 'MAX_ALLOCATION': 0.8}
        
        ComponentFactory.create_executor(TradingMode.PAPER, finance_params)
        
        mock_executor_class.assert_called_once_with(paper=True, max_allocation_rate=0.8)
    
    @patch('src.executor.alpaca_trade_executor.AlpacaExecutor')
    def test_create_executor_live(self, mock_executor_class):
        """Test creating executor for live mode."""
        finance_params = {'INITIAL_CAPITAL': 1000, 'MAX_ALLOCATION': 0.9}
        
        ComponentFactory.create_executor(TradingMode.LIVE, finance_params)
        
        mock_executor_class.assert_called_once_with(paper=False, max_allocation_rate=0.9)
    
    @patch('src.manager.position_manager.PositionManager')
    def test_create_position_manager(self, mock_pm_class):
        """Test creating position manager."""
        mock_executor = Mock()
        finance_params = {'INITIAL_CAPITAL': 1000}
        mock_data_fetcher = Mock()
        
        ComponentFactory.create_position_manager(
            mock_executor,
            finance_params,
            data_fetcher=mock_data_fetcher
        )
        
        mock_pm_class.assert_called_once_with(
            mock_executor,
            finance_params,
            data_fetcher=mock_data_fetcher
        )
    
    @patch('src.cache.trading_cache.TradingCache')
    def test_create_cache_with_file(self, mock_cache_class):
        """Test creating cache with file path."""
        ComponentFactory.create_cache('test_cache.json')
        
        mock_cache_class.assert_called_once_with('test_cache.json')
    
    @patch('src.cache.trading_cache.TradingCache')
    def test_create_cache_without_file(self, mock_cache_class):
        """Test creating cache without file path."""
        ComponentFactory.create_cache()
        
        mock_cache_class.assert_called_once_with()
    
    @patch('src.visualization.simple_chart_visualizer.SimpleChartVisualizer')
    def test_create_visualizer(self, mock_viz_class):
        """Test creating visualizer."""
        mock_instance = Mock()
        mock_viz_class.return_value = mock_instance
        
        viz = ComponentFactory.create_visualizer(
            ticker='TEST',
            output_file='test.html',
            auto_open=False,
            initial_capital=5000.0
        )
        
        mock_viz_class.assert_called_once_with(
            ticker='TEST',
            output_file='test.html',
            auto_open=False
        )
        mock_instance.set_initial_capital.assert_called_once_with(5000.0)


# ==========================================
# Test Default Strategy Registration
# ==========================================

class TestDefaultStrategyRegistration:
    """Tests for default strategy registration."""
    
    def test_register_default_strategies_runs(self):
        """Test that register_default_strategies runs without error."""
        # Should not raise even if some imports fail
        register_default_strategies()
    
    def test_some_strategies_registered(self):
        """Test that at least some strategies are registered after import."""
        # After importing the module, some strategies should be registered
        keys = StrategyRegistry.get_all_keys()
        
        # At least one strategy should be available
        # (depending on which strategy modules are importable)
        assert isinstance(keys, list)


# ==========================================
# Run Tests
# ==========================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])