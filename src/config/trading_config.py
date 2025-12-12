# src/config/trading_config.py

"""
交易系统配置中心 - Trading Configuration Center

所有参数集中管理，分为以下几类：
1. FinanceConfig - 资金和费用参数
2. TimeConfig - 时间控制参数
3. DataConfig - 数据获取参数
4. StrategyConfig - 策略参数（每个策略有自己的配置）
5. SystemConfig - 系统运行参数

使用方式：
    from src.config.trading_config import (
        DEFAULT_FINANCE_CONFIG,
        DEFAULT_TIME_CONFIG,
        SimpleTrendConfig,
        get_full_config
    )
    
    # 使用默认配置
    config = get_full_config()
    
    # 自定义配置
    config = get_full_config(
        initial_capital=5000,
        strategy='up_trend_aware',
        ticker='AAPL'
    )
"""

from dataclasses import dataclass, field
from datetime import time as dt_time
from typing import Dict, Any, Optional, Literal
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


# ============================================================
# 1. 资金和费用配置 - Finance Config
# ============================================================

@dataclass
class FinanceConfig:
    """
    资金和费用参数
    
    用于: PositionManager, TradingEngine
    """
    # 初始资金
    initial_capital: float = 1000.0
    
    # 费用参数
    commission_rate: float = 0.0003    # 佣金率 0.03%
    slippage_rate: float = 0.0001      # 滑点率 0.01%
    stamp_duty_rate: float = 0.001     # 印花税（A股）0.1%
    
    # 仓位控制
    min_lot_size: int = 1              # 最小交易单位
    max_allocation: float = 0.95       # 最大仓位比例 95%
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（兼容旧代码）"""
        return {
            'INITIAL_CAPITAL': self.initial_capital,
            'COMMISSION_RATE': self.commission_rate,
            'SLIPPAGE_RATE': self.slippage_rate,
            'MIN_LOT_SIZE': self.min_lot_size,
            'MAX_ALLOCATION': self.max_allocation,
            'STAMP_DUTY_RATE': self.stamp_duty_rate,
        }


# 默认配置
DEFAULT_FINANCE_CONFIG = FinanceConfig()

# 预设配置
FINANCE_PRESETS = {
    'small': FinanceConfig(initial_capital=1000, max_allocation=0.95),
    'medium': FinanceConfig(initial_capital=5000, max_allocation=0.90),
    'large': FinanceConfig(initial_capital=25000, max_allocation=0.80),
    'paper': FinanceConfig(initial_capital=100000, max_allocation=0.50),
}


# ============================================================
# 2. 时间控制配置 - Time Config
# ============================================================

@dataclass
class TimeConfig:
    """
    交易时间参数
    
    用于: TradingEngine
    """
    # 美股交易时间
    market_open_time: dt_time = field(default_factory=lambda: dt_time(9, 30))
    market_close_time: dt_time = field(default_factory=lambda: dt_time(16, 0))
    
    # 交易控制
    last_entry_time: dt_time = field(default_factory=lambda: dt_time(15, 50))   # 最后开仓时间
    force_close_time: dt_time = field(default_factory=lambda: dt_time(15, 55))  # 强制平仓时间
    
    # 时区
    timezone: str = 'America/New_York'


DEFAULT_TIME_CONFIG = TimeConfig()


# ============================================================
# 3. 数据获取配置 - Data Config
# ============================================================

@dataclass
class DataConfig:
    """
    数据获取参数
    
    用于: DataFetcher, TradingEngine
    """
    # K线参数
    timeframe_value: int = 5
    timeframe_unit: str = 'Minute'     # 'Minute', 'Hour', 'Day'
    
    # 回溯参数
    lookback_minutes: int = 300        # 获取多少分钟的历史数据
    
    # 运行参数
    step_seconds: int = 30             # 每次迭代间隔（秒）
    
    @property
    def timeframe(self) -> TimeFrame:
        """获取 Alpaca TimeFrame 对象"""
        unit_map = {
            'Minute': TimeFrameUnit.Minute,
            'Hour': TimeFrameUnit.Hour,
            'Day': TimeFrameUnit.Day,
        }
        return TimeFrame(self.timeframe_value, unit_map.get(self.timeframe_unit, TimeFrameUnit.Minute))


DEFAULT_DATA_CONFIG = DataConfig()

# 预设配置
DATA_PRESETS = {
    'scalping': DataConfig(timeframe_value=1, lookback_minutes=60, step_seconds=10),
    'intraday': DataConfig(timeframe_value=5, lookback_minutes=300, step_seconds=30),
    'swing': DataConfig(timeframe_value=15, lookback_minutes=1000, step_seconds=60),
}


# ============================================================
# 4. 策略配置 - Strategy Configs
# ============================================================

@dataclass
class SimpleUpTrendConfig:
    """
    SimpleTrendStrategy 参数配置
    
    用于: SimpleTrendStrategy
    """
    # ---------- 技术指标参数 ----------
    # 布林带
    bb_period: int = 20
    bb_std_dev: float = 2.0
    
    # ADX（趋势强度）
    adx_period: int = 14
    adx_trend_threshold: float = 25    # > 此值 = 趋势市
    adx_range_threshold: float = 20    # < 此值 = 震荡市
    
    # EMA（趋势方向）
    ema_fast: int = 12
    ema_slow: int = 26
    
    # ---------- 交易参数 ----------
    # 上升趋势
    uptrend_buy_low: float = 0.40      # BB 位置下限
    uptrend_buy_high: float = 0.60     # BB 位置上限
    uptrend_take_profit: float = 0.03  # 止盈 3%
    
    # 震荡市场
    range_buy_threshold: float = 0.20  # BB < 20% 买入
    range_sell_threshold: float = 0.55 # BB > 55% 卖出
    
    # ---------- 止损参数 ----------
    quick_stop_loss: float = 0.0005     # 快速止损 0.05%（下降趋势）
    normal_stop_loss: float = 0.001     # 正常止损 0.1%
    
    # ---------- 动态仓位管理 ----------
    reduce_allocation_threshold: float = 0.001   # 亏损 0.1% 触发减仓
    reduce_allocation_ratio: float = 0.5        # 减到原来的 50%
    recovery_threshold: float = 0.005           # 盈利 0.5% 开始恢复
    recovery_step: float = 0.1                  # 每次恢复 10%
    min_allocation: float = 0.25                # 最小仓位 25%
    max_allocation: float = 1.0                 # 最大仓位 100%
    
    # ---------- 其他 ----------
    max_history_bars: int = 500
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'bb_period': self.bb_period,
            'bb_std_dev': self.bb_std_dev,
            'adx_period': self.adx_period,
            'adx_trend_threshold': self.adx_trend_threshold,
            'adx_range_threshold': self.adx_range_threshold,
            'ema_fast': self.ema_fast,
            'ema_slow': self.ema_slow,
            'uptrend_buy_low': self.uptrend_buy_low,
            'uptrend_buy_high': self.uptrend_buy_high,
            'uptrend_take_profit': self.uptrend_take_profit,
            'range_buy_threshold': self.range_buy_threshold,
            'range_sell_threshold': self.range_sell_threshold,
            'quick_stop_loss': self.quick_stop_loss,
            'normal_stop_loss': self.normal_stop_loss,
            'reduce_allocation_threshold': self.reduce_allocation_threshold,
            'reduce_allocation_ratio': self.reduce_allocation_ratio,
            'recovery_threshold': self.recovery_threshold,
            'recovery_step': self.recovery_step,
            'min_allocation': self.min_allocation,
            'max_allocation': self.max_allocation,
            'max_history_bars': self.max_history_bars,
        }


@dataclass
class TrendAwareConfig:
    """
    TrendAwareStrategy 参数配置
    
    用于: TrendAwareStrategy
    """
    # 布林带
    bb_period: int = 20
    bb_std_dev: float = 2.0
    
    # ADX
    adx_period: int = 14
    adx_trend_threshold: float = 25
    adx_range_threshold: float = 20
    
    # EMA
    ema_fast_period: int = 12
    ema_slow_period: int = 26
    
    # 均值回归（震荡市）
    mean_reversion_entry: float = 0.9
    mean_reversion_exit: float = 0.60
    
    # 趋势跟踪
    trend_entry_pullback: float = 0.50
    trend_exit_profit: float = 0.03
    
    # 波动率过滤
    min_bb_width_pct: float = 0.02
    
    # 冷却期
    cooldown_minutes: int = 15
    
    # 止损
    stop_loss_threshold: float = 0.02
    
    # 其他
    monitor_interval_seconds: int = 60
    max_history_bars: int = 500
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {k: v for k, v in self.__dict__.items()}


# 默认策略配置
DEFAULT_SIMPLE_UPTREND_CONFIG = SimpleUpTrendConfig()
DEFAULT_TREND_AWARE_CONFIG = TrendAwareConfig()

# 策略预设
SIMPLE_UP_TREND_PRESETS = {
    'conservative': SimpleUpTrendConfig(
        quick_stop_loss=0.0003,
        normal_stop_loss=0.001,
        uptrend_take_profit=0.02,
        reduce_allocation_threshold=0.005,
    ),
    'moderate': SimpleUpTrendConfig(),  # 默认配置
    'aggressive': SimpleUpTrendConfig(
        quick_stop_loss=0.001,
        normal_stop_loss=0.003,
        uptrend_take_profit=0.05,
        uptrend_buy_low=0.30,
        uptrend_buy_high=0.70,
    ),
}


# ============================================================
# 其他策略配置（兼容旧策略）
# ============================================================

@dataclass
class ModerateConfig:
    """ModerateAggressiveStrategy 配置"""
    bb_period: int = 20
    bb_std_dev: float = 2.0
    entry_threshold: float = 0.85
    exit_threshold: float = 0.60
    stop_loss_threshold: float = 0.10
    monitor_interval_seconds: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass  
class MeanReversionConfig:
    """MeanReversionStrategy 配置"""
    bb_period: int = 20
    bb_std_dev: float = 2.0
    rsi_window: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    max_history_bars: int = 500
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


DEFAULT_MODERATE_CONFIG = ModerateConfig()
DEFAULT_MEAN_REVERSION_CONFIG = MeanReversionConfig()

# ============================================================
# 5. 系统运行配置 - System Config
# ============================================================

@dataclass
class SystemConfig:
    """
    系统运行参数
    
    用于: LiveRunner, BacktestRunner
    """
    # 运行模式
    mode: Literal['simulation', 'paper', 'live'] = 'paper'
    
    # 交易标的
    ticker: str = 'TSLA'
    
    # 策略选择
    strategy: str = 'up_trend_aware'
    
    # 运行时间
    max_runtime_minutes: Optional[int] = None  # None = 无限
    
    # 图表
    enable_chart: bool = True
    auto_open_browser: bool = True
    chart_update_interval: int = 30
    
    # API 同步
    sync_position_on_start: bool = True
    
    # 输出
    output_dir: str = 'live_trading'
    verbose: bool = True
    
    # 是否遵循市场时间
    respect_market_hours: bool = True


DEFAULT_SYSTEM_CONFIG = SystemConfig()


# ============================================================
# 6. 完整配置 - Full Config
# ============================================================

@dataclass
class TradingConfig:
    """
    完整交易配置
    
    整合所有配置到一个对象
    """
    finance: FinanceConfig = field(default_factory=FinanceConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    # 策略配置（每个策略一个配置对象）
    up_trend_aware: SimpleUpTrendConfig = field(default_factory=SimpleUpTrendConfig)
    trend_aware: TrendAwareConfig = field(default_factory=TrendAwareConfig)
    moderate: ModerateConfig = field(default_factory=ModerateConfig)
    mean_reversion: MeanReversionConfig = field(default_factory=MeanReversionConfig)
    
    def get_strategy_config(self) -> Dict[str, Any]:
        """获取当前策略的配置"""
        strategy_map = {
            'up_trend_aware': self.up_trend_aware,
            'trend_aware': self.trend_aware,
            'moderate': self.moderate,
            'mean_reversion': self.mean_reversion,
        }
        config = strategy_map.get(self.system.strategy)
        return config.to_dict() if config else {}
    
    def summary(self) -> str:
        """打印配置摘要"""
        lines = [
            "\n" + "=" * 60,
            "📋 Trading Configuration Summary",
            "=" * 60,
            "",
            "💰 Finance:",
            f"   Initial Capital: ${self.finance.initial_capital:,.2f}",
            f"   Max Allocation: {self.finance.max_allocation * 100:.0f}%",
            f"   Commission: {self.finance.commission_rate * 100:.2f}%",
            "",
            "⏰ Time:",
            f"   Market: {self.time.market_open_time} - {self.time.market_close_time}",
            f"   Last Entry: {self.time.last_entry_time}",
            f"   Force Close: {self.time.force_close_time}",
            "",
            "📊 Data:",
            f"   Timeframe: {self.data.timeframe_value} {self.data.timeframe_unit}",
            f"   Lookback: {self.data.lookback_minutes} minutes",
            f"   Step: {self.data.step_seconds} seconds",
            "",
            "🖥️ System:",
            f"   Mode: {self.system.mode.upper()}",
            f"   Ticker: {self.system.ticker}",
            f"   Strategy: {self.system.strategy}",
            "",
        ]
        
        # 添加策略配置
        if self.system.strategy == 'up_trend_aware':
            lines.extend([
                "📈 Simple Trend Strategy:",
                f"   Stop Loss: {self.up_trend_aware.normal_stop_loss * 100:.1f}% / {self.up_trend_aware.quick_stop_loss * 100:.1f}% (quick)",
                f"   Take Profit: {self.up_trend_aware.uptrend_take_profit * 100:.1f}%",
                f"   Reduce Allocation: at {self.up_trend_aware.reduce_allocation_threshold * 100:.1f}% loss",
            ])
        
        lines.append("=" * 60 + "\n")
        return "\n".join(lines)


def get_full_config(
    # 快捷参数
    initial_capital: Optional[float] = None,
    ticker: Optional[str] = None,
    strategy: Optional[str] = None,
    mode: Optional[str] = None,
    
    # 使用预设
    finance_preset: Optional[str] = None,
    data_preset: Optional[str] = None,
    strategy_preset: Optional[str] = None,
    
    # 完整覆盖
    finance: Optional[FinanceConfig] = None,
    time: Optional[TimeConfig] = None,
    data: Optional[DataConfig] = None,
    system: Optional[SystemConfig] = None,
    up_trend_aware: Optional[SimpleUpTrendConfig] = None,
    trend_aware: Optional[TrendAwareConfig] = None,
) -> TradingConfig:
    """
    获取完整配置
    
    优先级: 参数 > 预设 > 默认
    
    Examples:
        # 使用默认配置
        config = get_full_config()
        
        # 快速修改几个参数
        config = get_full_config(
            initial_capital=5000,
            ticker='AAPL',
            mode='paper'
        )
        
        # 使用预设
        config = get_full_config(
            finance_preset='medium',
            strategy_preset='conservative'
        )
    """
    # 加载预设或默认
    fin = FINANCE_PRESETS.get(finance_preset, DEFAULT_FINANCE_CONFIG) if finance is None else finance
    dat = DATA_PRESETS.get(data_preset, DEFAULT_DATA_CONFIG) if data is None else data
    tim = DEFAULT_TIME_CONFIG if time is None else time
    sys = DEFAULT_SYSTEM_CONFIG if system is None else system
    
    # 策略配置
    st = SIMPLE_UP_TREND_PRESETS.get(strategy_preset, DEFAULT_SIMPLE_UPTREND_CONFIG) if up_trend_aware is None else up_trend_aware
    ta = DEFAULT_TREND_AWARE_CONFIG if trend_aware is None else trend_aware
    
    # 应用快捷参数
    if initial_capital is not None:
        fin = FinanceConfig(**{**fin.__dict__, 'initial_capital': initial_capital})
    
    if ticker is not None or strategy is not None or mode is not None:
        sys_dict = sys.__dict__.copy()
        if ticker is not None:
            sys_dict['ticker'] = ticker
        if strategy is not None:
            sys_dict['strategy'] = strategy
        if mode is not None:
            sys_dict['mode'] = mode
        sys = SystemConfig(**sys_dict)
    
    return TradingConfig(
        finance=fin,
        time=tim,
        data=dat,
        system=sys,
        up_trend_aware=st,
        trend_aware=ta,
    )


# ============================================================
# 快速配置函数
# ============================================================

def quick_config(
    capital: float = 1000,
    ticker: str = 'TSLA',
    strategy: str = 'up_trend_aware',
    mode: str = 'paper',
    stop_loss: float = 0.02,
    take_profit: float = 0.03,
) -> TradingConfig:
    """
    快速创建配置（最常用参数）
    
    Example:
        config = quick_config(capital=5000, ticker='AAPL', stop_loss=0.015)
    """
    return TradingConfig(
        finance=FinanceConfig(initial_capital=capital),
        system=SystemConfig(ticker=ticker, strategy=strategy, mode=mode),
        simple_uptrend=SimpleUpTrendConfig(
            normal_stop_loss=stop_loss,
            uptrend_take_profit=take_profit,
        ),
    )


# ============================================================
# 测试
# ============================================================

if __name__ == '__main__':
    # 测试默认配置
    config = get_full_config()
    print(config.summary())
    
    # 测试快速配置
    config2 = quick_config(capital=5000, ticker='AAPL', stop_loss=0.015)
    print(config2.summary())
    
    # 测试预设
    config3 = get_full_config(
        finance_preset='medium',
        strategy_preset='conservative',
        ticker='NVDA'
    )
    print(config3.summary())