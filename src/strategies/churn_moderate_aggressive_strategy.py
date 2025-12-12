# src/strategies/moderate_aggressive_strategy.py (FIXED VERSION)

"""
温和进取策略 - Moderate Aggressive Mean Reversion
修复版 - 防止重复开仓 + 强化收盘管理

🔥 关键修复：
1. **防止重复开仓**：有持仓时优先考虑平仓，不会重复开仓
2. **逻辑优先级**：止损 > 时间窗口 > 平仓 > 开仓
3. **持仓状态检查**：有持仓时返回 HOLD，等待平仓信号
4. **收盘管理**：15:50后禁止开新仓，15:55强制平仓

核心改进：
1. 接近布林带边界就开仓（不必完全突破）
2. 回调到 60% 位置就平仓（不必回到中线）
3. 可调节的灵敏度参数
4. 强化的收盘时间管理
"""

from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from src.strategies.moderate_aggressive_strategy import ModerateAggressiveStrategy


class ChurnModerateAggressiveStrategy:
    """
    温和进取型均值回归策略（修复版 - 防止重复开仓）

    交易规则：
    - 价格 > 布林带宽度 85% → SHORT（例：接近上轨）
    - 空仓价格回落到 60% → COVER
    - 价格 < 布林带宽度 15% → BUY（例：接近下轨）
    - 多仓价格上涨到 40% → SELL

    🔥 防止重复开仓逻辑：
    - 有空仓时：只能 COVER 或 HOLD，不能再 SHORT
    - 有多仓时：只能 SELL 或 HOLD，不能再 BUY
    - 无持仓时：才允许 BUY 或 SHORT

    收盘管理：
    - 15:50后：禁止开新仓（BUY/SHORT），只允许平仓（SELL/COVER）
    - 15:55后：强制平仓所有持仓
    - 16:00前：确保持仓为0
    """

    def __init__(self,
                 bb_period: int = 20,
                 bb_std_dev: float = 2.0,
                 entry_threshold: float = 0.85,    # 开仓阈值（0.85 = 接近 85%）
                 exit_threshold: float = 0.60,     # 平仓阈值（0.60 = 回到 60%）
                 stop_loss_threshold: float = 0.10,
                 monitor_interval_seconds: int = 60,
                 max_history_bars: int = 500,
                 # 收盘时间控制
                 no_new_entry_time: int = 15 * 60 +
                 50,  # 15:50 (minutes from midnight)
                 force_close_time: int = 15 * 60 + 55):  # 15:55 (minutes from midnight)
        """
        参数说明：
            entry_threshold: 开仓阈值（0-1）
                - 0.85 = 价格接近布林带 85% 时开仓（温和）
                - 0.90 = 更接近边界（保守）
                - 0.80 = 更早开仓（激进）

            exit_threshold: 平仓阈值（0-1）
                - 0.60 = 价格回到 60% 位置平仓
                - 0.50 = 回到中线平仓（保守）
                - 0.70 = 快速平仓（激进）

            no_new_entry_time: 禁止开新仓时间（分钟，从午夜算起）
                - 默认 950 = 15:50

            force_close_time: 强制平仓时间（分钟）
                - 默认 955 = 15:55
        """
        self.moderate_aggressive_strategy = ModerateAggressiveStrategy(
            bb_period, bb_std_dev, entry_threshold, exit_threshold, stop_loss_threshold, monitor_interval_seconds, max_history_bars, no_new_entry_time, force_close_time)

    def _get_churn_signal(self,signal: str):
        if signal == 'SHORT':
            return 'BUY'
        if signal == 'BUY':
            return 'SHORT'
        if signal == 'COVER':
            return 'SELL'
        if signal == 'SELL':
            return 'COVER'
    def get_signal(self,
                   ticker: str,
                   new_data: pd.DataFrame,
                   current_position: float = 0.0,
                   avg_cost: float = 0.0,
                   verbose: bool = True,
                   is_market_close: bool = False,
                   current_time_et: pd.Timestamp = None) -> Tuple[Dict, float]:
        """
        获取交易信号（修复版 - 防止重复开仓）

        Args:
            ticker: 股票代码
            new_data: 新的 OHLCV DataFrame
            current_position: 当前持仓
            avg_cost: 平均成本
            verbose: 是否打印详细信息
            is_market_close: 是否是强制平仓时间（15:55+）
            current_time_et: 当前东部时间

        Returns:
            (signal_dict, current_price)
        """
        orig_signal, price = self.moderate_aggressive_strategy.get_signal(
            ticker, new_data, current_position, avg_cost, verbose, is_market_close, current_time_et)
        
        churn_signal = self._get_churn_signal(orig_signal['signal'])
        return {
            "signal": churn_signal,
            "confidence_score": orig_signal['confidence_score'],
            "reason": orig_signal['reason']
        }, price
        
