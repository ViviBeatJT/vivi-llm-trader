# src/strategies/simple_uptrend_strategy_v5.py

"""
简单趋势策略 v5 - Simple Trend Strategy (Long Only) + K线反转形态

核心改进：
1. 在布林带判断的基础上，加入K线反转形态确认
2. 只有当K线出现看涨反转形态时才买入
3. 支持多种经典反转形态：锤子线、吞没、早晨之星、刺透等

K线反转形态：
- 锤子线 (Hammer): 下影线长，实体小，出现在下跌后
- 看涨吞没 (Bullish Engulfing): 阳线完全吞没前一根阴线
- 早晨之星 (Morning Star): 三根K线组合，底部反转信号
- 刺透形态 (Piercing Line): 阳线深入前一根阴线实体50%以上
- 十字星 (Doji): 开盘价接近收盘价，表示犹豫
- 看涨孕线 (Bullish Harami): 小阳线包含在前一根大阴线实体内
"""

from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta, time
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


# ==================== K线形态定义 ====================

class CandlePattern(Enum):
    """K线形态枚举"""
    NONE = "无形态"
    HAMMER = "锤子线"
    INVERTED_HAMMER = "倒锤子"
    BULLISH_ENGULFING = "看涨吞没"
    MORNING_STAR = "早晨之星"
    PIERCING_LINE = "刺透形态"
    DOJI = "十字星"
    BULLISH_HARAMI = "看涨孕线"
    DRAGONFLY_DOJI = "蜻蜓十字"
    THREE_WHITE_SOLDIERS = "三白兵"


@dataclass
class PatternResult:
    """形态识别结果"""
    pattern: CandlePattern
    strength: float  # 形态强度 0-1
    description: str
    is_bullish: bool = True


class CandlePatternRecognizer:
    """
    K线形态识别器

    识别常见的看涨反转形态，用于确认买入信号
    """

    def __init__(self,
                 # 锤子线参数
                 hammer_body_ratio: float = 0.3,      # 实体占比上限
                 hammer_shadow_ratio: float = 2.0,   # 下影线是实体的倍数

                 # 吞没形态参数
                 engulfing_min_body: float = 0.005,  # 最小实体比例

                 # 十字星参数
                 doji_body_ratio: float = 0.1,       # 实体占振幅比例

                 # 刺透形态参数
                 piercing_min_penetration: float = 0.5,  # 最小穿透比例

                 # 通用参数
                 min_candle_size: float = 0.001,     # 最小K线振幅（相对价格）
                 ):

        self.hammer_body_ratio = hammer_body_ratio
        self.hammer_shadow_ratio = hammer_shadow_ratio
        self.engulfing_min_body = engulfing_min_body
        self.doji_body_ratio = doji_body_ratio
        self.piercing_min_penetration = piercing_min_penetration
        self.min_candle_size = min_candle_size

    def _get_candle_metrics(self, open_p: float, high: float,
                            low: float, close: float) -> Dict:
        """计算单根K线的各项指标"""
        body = abs(close - open_p)
        upper_shadow = high - max(open_p, close)
        lower_shadow = min(open_p, close) - low
        total_range = high - low

        is_bullish = close > open_p
        mid_price = (high + low) / 2

        # 防止除零
        body_ratio = body / total_range if total_range > 0 else 0

        return {
            'body': body,
            'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow,
            'total_range': total_range,
            'is_bullish': is_bullish,
            'body_ratio': body_ratio,
            'mid_price': mid_price,
            'open': open_p,
            'high': high,
            'low': low,
            'close': close,
        }

    def _is_hammer(self, metrics: Dict) -> Tuple[bool, float]:
        """
        检测锤子线

        特征：
        - 小实体在K线上部
        - 长下影线（至少是实体的2倍）
        - 几乎没有上影线
        """
        if metrics['total_range'] == 0:
            return False, 0.0

        body = metrics['body']
        lower_shadow = metrics['lower_shadow']
        upper_shadow = metrics['upper_shadow']
        total_range = metrics['total_range']

        # 实体占比小
        body_ratio = body / total_range
        if body_ratio > self.hammer_body_ratio:
            return False, 0.0

        # 下影线长
        if body > 0 and lower_shadow < body * self.hammer_shadow_ratio:
            return False, 0.0

        # 上影线短
        if upper_shadow > body * 0.5:
            return False, 0.0

        # 计算强度
        strength = min(lower_shadow / (body + 0.001) /
                       self.hammer_shadow_ratio, 1.0)
        return True, strength

    def _is_inverted_hammer(self, metrics: Dict) -> Tuple[bool, float]:
        """
        检测倒锤子

        特征：
        - 小实体在K线下部
        - 长上影线
        - 几乎没有下影线
        """
        if metrics['total_range'] == 0:
            return False, 0.0

        body = metrics['body']
        lower_shadow = metrics['lower_shadow']
        upper_shadow = metrics['upper_shadow']
        total_range = metrics['total_range']

        # 实体占比小
        body_ratio = body / total_range
        if body_ratio > self.hammer_body_ratio:
            return False, 0.0

        # 上影线长
        if body > 0 and upper_shadow < body * self.hammer_shadow_ratio:
            return False, 0.0

        # 下影线短
        if lower_shadow > body * 0.5:
            return False, 0.0

        strength = min(upper_shadow / (body + 0.001) /
                       self.hammer_shadow_ratio, 1.0)
        return True, strength

    def _is_doji(self, metrics: Dict) -> Tuple[bool, float]:
        """
        检测十字星

        特征：
        - 开盘价接近收盘价（实体极小）
        """
        if metrics['total_range'] == 0:
            return False, 0.0

        body_ratio = metrics['body_ratio']

        if body_ratio <= self.doji_body_ratio:
            strength = 1.0 - body_ratio / self.doji_body_ratio
            return True, strength

        return False, 0.0

    def _is_dragonfly_doji(self, metrics: Dict) -> Tuple[bool, float]:
        """
        检测蜻蜓十字（T字线）

        特征：
        - 十字星
        - 长下影线
        - 几乎没有上影线
        """
        is_doji, doji_strength = self._is_doji(metrics)
        if not is_doji:
            return False, 0.0

        lower_shadow = metrics['lower_shadow']
        upper_shadow = metrics['upper_shadow']
        total_range = metrics['total_range']

        # 下影线占主导
        if total_range > 0 and lower_shadow / total_range > 0.6 and upper_shadow / total_range < 0.1:
            return True, doji_strength

        return False, 0.0

    def _is_bullish_engulfing(self, prev_metrics: Dict, curr_metrics: Dict) -> Tuple[bool, float]:
        """
        检测看涨吞没

        特征：
        - 前一根是阴线
        - 当前是阳线
        - 阳线实体完全吞没阴线实体
        """
        # 前一根必须是阴线
        if prev_metrics['is_bullish']:
            return False, 0.0

        # 当前必须是阳线
        if not curr_metrics['is_bullish']:
            return False, 0.0

        # 阳线实体必须大于阴线实体
        if curr_metrics['body'] <= prev_metrics['body']:
            return False, 0.0

        # 阳线开盘低于阴线收盘，阳线收盘高于阴线开盘
        if curr_metrics['open'] >= prev_metrics['close']:
            return False, 0.0

        if curr_metrics['close'] <= prev_metrics['open']:
            return False, 0.0

        # 计算吞没程度
        engulf_ratio = curr_metrics['body'] / (prev_metrics['body'] + 0.001)
        strength = min(engulf_ratio / 2.0, 1.0)

        return True, strength

    def _is_piercing_line(self, prev_metrics: Dict, curr_metrics: Dict) -> Tuple[bool, float]:
        """
        检测刺透形态

        特征：
        - 前一根是大阴线
        - 当前是阳线
        - 阳线开盘低于阴线最低价
        - 阳线收盘深入阴线实体50%以上
        """
        # 前一根必须是阴线
        if prev_metrics['is_bullish']:
            return False, 0.0

        # 当前必须是阳线
        if not curr_metrics['is_bullish']:
            return False, 0.0

        # 阳线开盘低于阴线最低价（跳空低开）
        if curr_metrics['open'] >= prev_metrics['low']:
            return False, 0.0

        # 计算穿透比例
        prev_body_mid = (prev_metrics['open'] + prev_metrics['close']) / 2
        penetration = (curr_metrics['close'] - prev_metrics['close']) / \
            prev_metrics['body'] if prev_metrics['body'] > 0 else 0

        if penetration < self.piercing_min_penetration:
            return False, 0.0

        strength = min(penetration, 1.0)
        return True, strength

    def _is_bullish_harami(self, prev_metrics: Dict, curr_metrics: Dict) -> Tuple[bool, float]:
        """
        检测看涨孕线

        特征：
        - 前一根是大阴线
        - 当前是小阳线
        - 阳线实体完全在阴线实体内部
        """
        # 前一根必须是阴线
        if prev_metrics['is_bullish']:
            return False, 0.0

        # 当前必须是阳线
        if not curr_metrics['is_bullish']:
            return False, 0.0

        # 当前实体必须小于前一根
        if curr_metrics['body'] >= prev_metrics['body'] * 0.6:
            return False, 0.0

        # 阳线实体必须在阴线实体内部
        if curr_metrics['open'] <= prev_metrics['close'] or curr_metrics['close'] >= prev_metrics['open']:
            return False, 0.0

        strength = 1.0 - curr_metrics['body'] / prev_metrics['body']
        return True, strength

    def _is_morning_star(self, candles: List[Dict]) -> Tuple[bool, float]:
        """
        检测早晨之星

        特征：
        - 第一根：大阴线
        - 第二根：小实体（十字星或小K线），跳空低开
        - 第三根：大阳线，深入第一根实体50%以上
        """
        if len(candles) < 3:
            return False, 0.0

        first, second, third = candles[-3], candles[-2], candles[-1]

        # 第一根必须是大阴线
        if first['is_bullish'] or first['body_ratio'] < 0.5:
            return False, 0.0

        # 第二根是小实体
        if second['body_ratio'] > 0.3:
            return False, 0.0

        # 第三根必须是阳线
        if not third['is_bullish']:
            return False, 0.0

        # 第三根深入第一根实体
        penetration = (third['close'] - first['close']) / \
            first['body'] if first['body'] > 0 else 0

        if penetration < 0.5:
            return False, 0.0

        strength = min(penetration, 1.0)
        return True, strength

    def _is_three_white_soldiers(self, candles: List[Dict]) -> Tuple[bool, float]:
        """
        检测三白兵

        特征：
        - 连续三根阳线
        - 每根收盘价高于前一根
        - 实体较大，影线较短
        """
        if len(candles) < 3:
            return False, 0.0

        last_three = candles[-3:]

        # 都是阳线
        if not all(c['is_bullish'] for c in last_three):
            return False, 0.0

        # 收盘价递增
        closes = [c['close'] for c in last_three]
        if not (closes[0] < closes[1] < closes[2]):
            return False, 0.0

        # 实体较大
        avg_body_ratio = sum(c['body_ratio'] for c in last_three) / 3
        if avg_body_ratio < 0.4:
            return False, 0.0

        strength = min(avg_body_ratio / 0.7, 1.0)
        return True, strength

    def recognize(self, df: pd.DataFrame, lookback: int = 5) -> List[PatternResult]:
        """
        识别K线形态

        Args:
            df: 包含 open, high, low, close 的 DataFrame
            lookback: 回看K线数量

        Returns:
            识别到的形态列表（按强度排序）
        """
        if len(df) < 3:
            return []

        # 获取最近的K线数据
        recent = df.tail(lookback)

        # 计算每根K线的指标
        candles = []
        for idx in range(len(recent)):
            row = recent.iloc[idx]
            metrics = self._get_candle_metrics(
                row['open'], row['high'], row['low'], row['close']
            )
            candles.append(metrics)

        results = []
        current = candles[-1]

        # 检测单K线形态
        is_hammer, strength = self._is_hammer(current)
        if is_hammer:
            results.append(PatternResult(
                CandlePattern.HAMMER, strength,
                f"锤子线 (强度: {strength:.0%})"
            ))

        is_inv_hammer, strength = self._is_inverted_hammer(current)
        if is_inv_hammer:
            results.append(PatternResult(
                CandlePattern.INVERTED_HAMMER, strength,
                f"倒锤子 (强度: {strength:.0%})"
            ))

        is_dragonfly, strength = self._is_dragonfly_doji(current)
        if is_dragonfly:
            results.append(PatternResult(
                CandlePattern.DRAGONFLY_DOJI, strength,
                f"蜻蜓十字 (强度: {strength:.0%})"
            ))
        elif not is_dragonfly:
            is_doji, strength = self._is_doji(current)
            if is_doji:
                results.append(PatternResult(
                    CandlePattern.DOJI, strength * 0.5,  # 普通十字星强度减半
                    f"十字星 (强度: {strength:.0%})"
                ))

        # 检测双K线形态
        if len(candles) >= 2:
            prev = candles[-2]

            is_engulfing, strength = self._is_bullish_engulfing(prev, current)
            if is_engulfing:
                results.append(PatternResult(
                    CandlePattern.BULLISH_ENGULFING, strength,
                    f"看涨吞没 (强度: {strength:.0%})"
                ))

            is_piercing, strength = self._is_piercing_line(prev, current)
            if is_piercing:
                results.append(PatternResult(
                    CandlePattern.PIERCING_LINE, strength,
                    f"刺透形态 (强度: {strength:.0%})"
                ))

            is_harami, strength = self._is_bullish_harami(prev, current)
            if is_harami:
                results.append(PatternResult(
                    CandlePattern.BULLISH_HARAMI, strength,
                    f"看涨孕线 (强度: {strength:.0%})"
                ))

        # 检测三K线形态
        if len(candles) >= 3:
            is_morning, strength = self._is_morning_star(candles)
            if is_morning:
                results.append(PatternResult(
                    CandlePattern.MORNING_STAR, strength,
                    f"早晨之星 (强度: {strength:.0%})"
                ))

            is_soldiers, strength = self._is_three_white_soldiers(candles)
            if is_soldiers:
                results.append(PatternResult(
                    CandlePattern.THREE_WHITE_SOLDIERS, strength,
                    f"三白兵 (强度: {strength:.0%})"
                ))

        # 按强度排序
        results.sort(key=lambda x: x.strength, reverse=True)

        return results

    def get_strongest_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """获取最强的反转形态"""
        patterns = self.recognize(df)
        return patterns[0] if patterns else None

    def has_reversal_pattern(self, df: pd.DataFrame,
                             min_strength: float = 0.3) -> Tuple[bool, Optional[PatternResult]]:
        """
        检查是否有反转形态

        Args:
            df: K线数据
            min_strength: 最小强度阈值

        Returns:
            (是否有反转形态, 最强形态)
        """
        patterns = self.recognize(df)
        valid_patterns = [p for p in patterns if p.strength >= min_strength]

        if valid_patterns:
            return True, valid_patterns[0]
        return False, None


# ==================== 主策略类 ====================

class SimpleUpTrendStrategy:
    """
    简单趋势策略 v5 - 只做多 + 动态仓位管理 + K线反转形态确认

    改进点：
    - 在布林带条件满足时，额外检查K线反转形态
    - 只有同时满足布林带位置和反转形态时才买入
    - 可配置是否强制要求反转形态

    市场状态判断：
    - ADX > 25 且 EMA快 > EMA慢 → 上升趋势 ✅ 可交易
    - ADX > 25 且 EMA快 < EMA慢 → 下降趋势 ⚠️ 持仓观望
    - ADX < 20 → 震荡市场 ✅ 可交易

    买入条件（v5新增）：
    1. 布林带位置满足条件（原有逻辑）
    2. K线出现看涨反转形态（新增）
    3. 形态强度超过阈值（可配置）
    """

    def __init__(self,
                 # ===== 布林带参数 =====
                 bb_period: int = 20,
                 bb_std_dev: float = 2.0,

                 # ===== ADX 参数（趋势强度）=====
                 adx_period: int = 14,
                 adx_trend_threshold: float = 25,
                 adx_range_threshold: float = 20,

                 # ===== EMA 参数（趋势方向）=====
                 ema_fast: int = 12,
                 ema_slow: int = 26,

                 # ===== 上升趋势交易参数 =====
                 uptrend_buy_low: float = 0.40,
                 uptrend_buy_high: float = 0.60,
                 uptrend_take_profit: float = 0.005,

                 # ===== 震荡市交易参数 =====
                 range_buy_threshold: float = 0.20,
                 range_sell_threshold: float = 0.55,
                 range_take_profit: float = 0.003,

                 # ===== 下降趋势交易参数 =====
                 downtrend_buy_threshold: float = 0.05,
                 downtrend_sell_threshold: float = 0.40,
                 downtrend_take_profit: float = 0.001,

                 # ===== 动态仓位管理参数 =====
                 quick_stop_loss: float = 0.0005,
                 normal_stop_loss: float = 0.001,
                 reduce_allocation_threshold: float = 0.01,
                 reduce_allocation_ratio: float = 0.5,
                 recovery_threshold: float = 0.005,
                 recovery_step: float = 0.1,
                 min_allocation: float = 0.25,
                 max_allocation: float = 1.0,

                 # ===== 冷却期参数 =====
                 cooldown_bars: int = 0,
                 cooldown_minutes: int = 0,
                 consecutive_loss_multiplier: float = 1.5,
                 max_cooldown_multiplier: float = 4.0,
                 consecutive_loss_reset_after_profit: bool = True,

                 # ===== 收盘保护参数 =====
                 no_new_position_minutes: int = 10,
                 market_close_time: time = time(16, 0),

                 # ===== 布林带保护参数 =====
                 bb_narrow_threshold: float = 0.02,
                 bb_narrow_action: str = 'BLOCK',

                 # ===== 🆕 K线形态参数 =====
                 require_candle_pattern: bool = True,           # 是否强制要求K线形态
                 pattern_min_strength: float = 0.3,             # 最小形态强度
                 pattern_lookback: int = 5,                     # 形态回看K线数
                 pattern_boost_confidence: bool = True,         # 形态是否提升信心值
                 pattern_confidence_boost: int = 2,             # 形态提升的信心值

                 # K线形态识别参数
                 hammer_body_ratio: float = 0.3,
                 hammer_shadow_ratio: float = 2.0,
                 engulfing_min_body: float = 0.005,
                 doji_body_ratio: float = 0.1,
                 piercing_min_penetration: float = 0.5,

                 # ===== 其他 =====
                 max_history_bars: int = 500,
                 verbose_init: bool = True):

        # 保存所有原有参数
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.uptrend_buy_low = uptrend_buy_low
        self.uptrend_buy_high = uptrend_buy_high
        self.uptrend_take_profit = uptrend_take_profit
        self.range_buy_threshold = range_buy_threshold
        self.range_sell_threshold = range_sell_threshold
        self.range_take_profit = range_take_profit
        self.downtrend_buy_threshold = downtrend_buy_threshold
        self.downtrend_sell_threshold = downtrend_sell_threshold
        self.downtrend_take_profit = downtrend_take_profit

        self.quick_stop_loss = quick_stop_loss
        self.normal_stop_loss = normal_stop_loss
        self.reduce_allocation_threshold = reduce_allocation_threshold
        self.reduce_allocation_ratio = reduce_allocation_ratio
        self.recovery_threshold = recovery_threshold
        self.recovery_step = recovery_step
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation

        self.cooldown_bars = cooldown_bars
        self.cooldown_minutes = cooldown_minutes
        self.consecutive_loss_multiplier = consecutive_loss_multiplier
        self.max_cooldown_multiplier = max_cooldown_multiplier
        self.consecutive_loss_reset_after_profit = consecutive_loss_reset_after_profit

        self.no_new_position_minutes = no_new_position_minutes
        self.market_close_time = market_close_time

        self.bb_narrow_threshold = bb_narrow_threshold
        self.bb_narrow_action = bb_narrow_action

        self.max_history_bars = max_history_bars

        # 🆕 K线形态参数
        self.require_candle_pattern = require_candle_pattern
        self.pattern_min_strength = pattern_min_strength
        self.pattern_lookback = pattern_lookback
        self.pattern_boost_confidence = pattern_boost_confidence
        self.pattern_confidence_boost = pattern_confidence_boost

        # 🆕 创建K线形态识别器
        self.pattern_recognizer = CandlePatternRecognizer(
            hammer_body_ratio=hammer_body_ratio,
            hammer_shadow_ratio=hammer_shadow_ratio,
            engulfing_min_body=engulfing_min_body,
            doji_body_ratio=doji_body_ratio,
            piercing_min_penetration=piercing_min_penetration,
        )

        # 数据存储
        self._history_data: Dict[str, pd.DataFrame] = {}

        # 动态仓位状态
        self._current_allocation: Dict[str, float] = {}
        self._peak_equity: Dict[str, float] = {}
        self._last_pnl_state: Dict[str, str] = {}

        # 冷却期状态
        self._stop_loss_time: Dict[str, datetime] = {}
        self._stop_loss_bar_count: Dict[str, int] = {}
        self._bar_count: Dict[str, int] = {}
        self._consecutive_losses: Dict[str, int] = {}
        self._current_cooldown_multiplier: Dict[str, float] = {}

        # 交易状态跟踪
        self._last_signal: Dict[str, str] = {}
        self._total_invested: Dict[str, float] = {}

        # 🆕 形态识别状态
        self._last_pattern: Dict[str, Optional[PatternResult]] = {}

        # 打印配置
        if verbose_init:
            self._print_config()

    def _print_config(self):
        """打印策略配置"""
        print(f"\n{'='*60}")
        print(f"📈 简单趋势策略 v5 (布林带 + K线反转形态)")
        print(f"{'='*60}")
        print(f"\n📊 K线形态配置:")
        print(f"  🆕 是否强制要求反转形态: {'是' if self.require_candle_pattern else '否'}")
        print(f"  🆕 最小形态强度: {self.pattern_min_strength*100:.0f}%")
        print(f"  🆕 形态回看K线: {self.pattern_lookback} 根")
        print(f"  🆕 形态提升信心: {'是' if self.pattern_boost_confidence else '否'}")
        print(f"\n趋势判断:")
        print(f"  ADX > {self.adx_trend_threshold} = 趋势市")
        print(f"  ADX < {self.adx_range_threshold} = 震荡市")
        print(f"\n交易参数:")
        print(
            f"  上升趋势买入: BB {self.uptrend_buy_low*100:.0f}%-{self.uptrend_buy_high*100:.0f}%")
        print(f"  震荡买入: BB < {self.range_buy_threshold*100:.0f}%")
        print(f"\n动态仓位管理:")
        print(f"  🛑 快速止损: {self.quick_stop_loss*100:.4f}%")
        print(f"  🛑 正常止损: {self.normal_stop_loss*100:.4f}%")
        print(f"  🛑 止盈: {self.uptrend_take_profit*100:.4f}%")
        print(f"{'='*60}\n")

    # ==================== K线形态相关方法 ====================

    def _check_candle_pattern(self, df: pd.DataFrame, verbose: bool = False) -> Tuple[bool, Optional[PatternResult]]:
        """
        检查是否有看涨反转形态

        Args:
            df: K线数据
            verbose: 是否打印详细信息

        Returns:
            (是否有形态, 形态结果)
        """
        has_pattern, pattern = self.pattern_recognizer.has_reversal_pattern(
            df,
            min_strength=self.pattern_min_strength
        )

        if verbose and has_pattern:
            print(f"   🕯️ 检测到K线形态: {pattern.description}")

        return has_pattern, pattern

    def _get_all_patterns(self, df: pd.DataFrame) -> List[PatternResult]:
        """获取所有识别到的形态"""
        return self.pattern_recognizer.recognize(df, lookback=self.pattern_lookback)

    # ==================== 收盘保护方法 ====================

    def _is_last_n_minutes(self, current_time, minutes: int = None) -> bool:
        """检查是否在收盘前N分钟内"""
        if minutes is None:
            minutes = self.no_new_position_minutes

        if current_time is None:
            return False

        if isinstance(current_time, datetime):
            current_time_only = current_time.time()
        elif isinstance(current_time, time):
            current_time_only = current_time
        else:
            return False

        close_minutes = self.market_close_time.hour * 60 + self.market_close_time.minute
        cutoff_minutes = close_minutes - minutes

        if cutoff_minutes < 0:
            cutoff_minutes = 0

        cutoff_hour = cutoff_minutes // 60
        cutoff_minute = cutoff_minutes % 60
        cutoff_time = time(cutoff_hour, cutoff_minute)

        return cutoff_time <= current_time_only < self.market_close_time

    # ==================== 布林带保护方法 ====================

    def _calculate_bb_width(self, bb_upper: float, bb_lower: float, price: float) -> float:
        """计算布林带宽度占价格的百分比"""
        if price <= 0:
            return 0.0
        return (bb_upper - bb_lower) / price

    def _is_bb_narrow(self, bb_upper: float, bb_lower: float, price: float) -> Tuple[bool, float]:
        """检查布林带是否过窄"""
        width_pct = self._calculate_bb_width(bb_upper, bb_lower, price)
        return width_pct < self.bb_narrow_threshold, width_pct

    # ==================== 连续亏损管理方法 ====================

    def _record_loss(self, ticker: str):
        """记录一次亏损"""
        self._consecutive_losses[ticker] = self._consecutive_losses.get(
            ticker, 0) + 1
        losses = self._consecutive_losses[ticker]
        new_multiplier = min(
            self.consecutive_loss_multiplier ** (losses - 1),
            self.max_cooldown_multiplier
        )
        self._current_cooldown_multiplier[ticker] = new_multiplier
        return losses, new_multiplier

    def _record_profit(self, ticker: str):
        """记录一次盈利"""
        if self.consecutive_loss_reset_after_profit:
            self._consecutive_losses[ticker] = 0
            self._current_cooldown_multiplier[ticker] = 1.0

    def _get_effective_cooldown(self, ticker: str) -> Tuple[int, int]:
        """获取有效冷却期"""
        multiplier = self._current_cooldown_multiplier.get(ticker, 1.0)
        effective_bars = int(self.cooldown_bars * multiplier)
        effective_minutes = int(self.cooldown_minutes * multiplier)
        return effective_bars, effective_minutes

    # ==================== 冷却期管理方法 ====================

    def _start_cooldown(self, ticker: str, current_time: datetime = None, is_stop_loss: bool = True):
        """开始冷却期"""
        if current_time is None:
            current_time = datetime.now()

        self._stop_loss_time[ticker] = current_time
        self._stop_loss_bar_count[ticker] = self._bar_count.get(ticker, 0)

        if is_stop_loss:
            losses, multiplier = self._record_loss(ticker)
            effective_bars, effective_minutes = self._get_effective_cooldown(
                ticker)

            if self.cooldown_minutes > 0:
                print(f"   ⏳ [冷却期开始] {ticker}: 等待 {effective_minutes} 分钟 "
                      f"(连续亏损{losses}次, {multiplier:.1f}x)")
            else:
                print(f"   ⏳ [冷却期开始] {ticker}: 等待 {effective_bars} 根K线 "
                      f"(连续亏损{losses}次, {multiplier:.1f}x)")

    def _is_in_cooldown(self, ticker: str, current_time: datetime = None) -> Tuple[bool, str]:
        """检查是否在冷却期内"""
        if ticker not in self._stop_loss_time:
            return False, ""

        if current_time is None:
            current_time = datetime.now()

        effective_bars, effective_minutes = self._get_effective_cooldown(
            ticker)

        if self.cooldown_minutes > 0:
            time_since_stop = current_time - self._stop_loss_time[ticker]
            cooldown_duration = timedelta(minutes=effective_minutes)

            if time_since_stop < cooldown_duration:
                remaining = cooldown_duration - time_since_stop
                remaining_mins = remaining.total_seconds() / 60
                multiplier = self._current_cooldown_multiplier.get(ticker, 1.0)
                return True, f"⏳ 冷却期中，还需 {remaining_mins:.1f} 分钟 ({multiplier:.1f}x)"
            else:
                del self._stop_loss_time[ticker]
                if ticker in self._stop_loss_bar_count:
                    del self._stop_loss_bar_count[ticker]
                return False, ""
        else:
            current_bar = self._bar_count.get(ticker, 0)
            stop_bar = self._stop_loss_bar_count.get(ticker, 0)
            bars_passed = current_bar - stop_bar

            if bars_passed < effective_bars:
                remaining = effective_bars - bars_passed
                multiplier = self._current_cooldown_multiplier.get(ticker, 1.0)
                return True, f"⏳ 冷却期中，还需 {remaining} 根K线 ({multiplier:.1f}x)"
            else:
                if ticker in self._stop_loss_time:
                    del self._stop_loss_time[ticker]
                if ticker in self._stop_loss_bar_count:
                    del self._stop_loss_bar_count[ticker]
                return False, ""

    # ==================== 仓位管理方法 ====================

    def get_current_allocation(self, ticker: str) -> float:
        """获取当前仓位比例"""
        if ticker not in self._current_allocation:
            self._current_allocation[ticker] = self.max_allocation
        return self._current_allocation[ticker]

    def _reduce_allocation(self, ticker: str, reason: str = "") -> float:
        """减少仓位"""
        current = self.get_current_allocation(ticker)
        new_allocation = max(
            current * self.reduce_allocation_ratio, self.min_allocation)
        self._current_allocation[ticker] = new_allocation
        print(
            f"   📉 [减仓] {ticker}: {current*100:.0f}% → {new_allocation*100:.0f}% ({reason})")
        return new_allocation

    def _recover_allocation(self, ticker: str) -> float:
        """恢复仓位"""
        current = self.get_current_allocation(ticker)
        if current >= self.max_allocation:
            return current
        new_allocation = min(current + self.recovery_step, self.max_allocation)
        self._current_allocation[ticker] = new_allocation
        print(
            f"   📈 [恢复仓位] {ticker}: {current*100:.0f}% → {new_allocation*100:.0f}%")
        return new_allocation

    def _reset_allocation(self, ticker: str):
        """重置仓位到最大"""
        self._current_allocation[ticker] = self.max_allocation

    def _update_allocation_based_on_pnl(self, ticker: str, pnl_pct: float, market_state: str):
        """根据盈亏情况动态调整仓位"""
        current_state = 'neutral'

        if pnl_pct <= -self.reduce_allocation_threshold:
            current_state = 'loss'
        elif pnl_pct >= self.recovery_threshold:
            current_state = 'profit'

        last_state = self._last_pnl_state.get(ticker, 'neutral')

        if current_state == 'loss' and last_state != 'loss':
            self._reduce_allocation(ticker, f"亏损 {pnl_pct*100:.2f}%")
        elif current_state == 'profit' and last_state == 'loss':
            self._recover_allocation(ticker)
        elif current_state == 'profit' and pnl_pct >= self.uptrend_take_profit:
            if self.get_current_allocation(ticker) < self.max_allocation:
                self._reset_allocation(ticker)

        self._last_pnl_state[ticker] = current_state

    def _can_open_position(self, ticker: str, requested_allocation: float) -> Tuple[bool, str]:
        """检查是否可以开仓"""
        current_invested = self._total_invested.get(ticker, 0.0)
        if current_invested >= self.max_allocation:
            return False, f"🚫 已达最大仓位 {self.max_allocation*100:.0f}%"
        if current_invested + requested_allocation > self.max_allocation:
            available = self.max_allocation - current_invested
            return True, f"⚠️ 只能再投入 {available*100:.0f}%"
        return True, ""

    def _update_invested(self, ticker: str, delta: float):
        """更新已投资比例"""
        current = self._total_invested.get(ticker, 0.0)
        self._total_invested[ticker] = max(
            0.0, min(current + delta, self.max_allocation))

    def _reset_invested(self, ticker: str):
        """重置已投资比例"""
        self._total_invested[ticker] = 0.0

    # ==================== 技术指标计算 ====================

    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """计算 EMA"""
        return series.ewm(span=period, adjust=False).mean()

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """计算 ADX"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        n = len(df)

        up_move = np.zeros(n)
        down_move = np.zeros(n)
        up_move[1:] = high[1:] - high[:-1]
        down_move[1:] = low[:-1] - low[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) &
                            (down_move > 0), down_move, 0)

        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i-1]),
                        abs(low[i] - close[i-1]))

        atr = pd.Series(tr).rolling(window=period, min_periods=1).mean().values
        plus_dm_smooth = pd.Series(plus_dm).rolling(
            window=period, min_periods=1).mean().values
        minus_dm_smooth = pd.Series(minus_dm).rolling(
            window=period, min_periods=1).mean().values

        atr_safe = np.where(atr == 0, 1, atr)
        plus_di = 100 * plus_dm_smooth / atr_safe
        minus_di = 100 * minus_dm_smooth / atr_safe

        di_sum = plus_di + minus_di
        di_sum_safe = np.where(di_sum == 0, 1, di_sum)
        dx = 100 * np.abs(plus_di - minus_di) / di_sum_safe
        adx = pd.Series(dx).rolling(window=period, min_periods=1).mean().values

        return adx

    def _get_market_state(self, adx: float, ema_fast: float, ema_slow: float) -> str:
        """判断市场状态"""
        if adx >= self.adx_trend_threshold:
            if ema_fast > ema_slow:
                return 'UPTREND'
            else:
                return 'DOWNTREND'
        elif adx <= self.adx_range_threshold:
            return 'RANGING'
        else:
            return 'UNCLEAR'

    def _calculate_bb_position(self, price: float, bb_upper: float, bb_lower: float) -> float:
        """计算 BB 位置 (0-1)"""
        bb_range = bb_upper - bb_lower
        if bb_range <= 0:
            return 0.5
        return (price - bb_lower) / bb_range

    # ==================== 主信号函数 ====================

    def get_signal(self,
                   ticker: str,
                   new_data: pd.DataFrame,
                   current_position: float = 0.0,
                   current_price: float = 0.0,
                   avg_cost: float = 0.0,
                   verbose: bool = False,
                   is_market_close: bool = False,
                   current_time_et=None,
                   **kwargs) -> Tuple[Dict, pd.DataFrame]:
        """获取交易信号"""

        # ========== 1. 更新历史数据 ==========
        if ticker not in self._history_data or self._history_data[ticker].empty:
            self._history_data[ticker] = new_data.copy()
        else:
            combined = pd.concat([self._history_data[ticker], new_data])
            combined = combined[~combined.index.duplicated(keep='last')]
            self._history_data[ticker] = combined.tail(self.max_history_bars)

        df = self._history_data[ticker]
        self._bar_count[ticker] = len(df)

        # ========== 2. 计算技术指标 ==========
        close = df['close']
        if current_price == 0.0:
            current_price = close.iloc[-1]

        # 布林带
        bb_middle = close.rolling(window=self.bb_period, min_periods=1).mean()
        bb_std = close.rolling(window=self.bb_period, min_periods=1).std()
        bb_upper = bb_middle + self.bb_std_dev * bb_std
        bb_lower = bb_middle - self.bb_std_dev * bb_std

        current_bb_upper = bb_upper.iloc[-1]
        current_bb_lower = bb_lower.iloc[-1]

        # EMA
        ema_fast_series = self._calculate_ema(close, self.ema_fast)
        ema_slow_series = self._calculate_ema(close, self.ema_slow)
        current_ema_fast = ema_fast_series.iloc[-1]
        current_ema_slow = ema_slow_series.iloc[-1]

        # ADX
        adx_values = self._calculate_adx(df, self.adx_period)
        current_adx = adx_values[-1] if len(adx_values) > 0 else 0

        # 市场状态
        market_state = self._get_market_state(
            current_adx, current_ema_fast, current_ema_slow)

        # BB 位置
        bb_position = self._calculate_bb_position(
            current_price, current_bb_upper, current_bb_lower)

        # 当前仓位比例
        current_allocation = self.get_current_allocation(ticker)

        # 获取当前时间
        if current_time_et is not None:
            current_time = current_time_et
        elif len(df) > 0 and hasattr(df.index[-1], 'to_pydatetime'):
            current_time = df.index[-1].to_pydatetime()
        else:
            current_time = datetime.now()

        # 检查布林带宽度
        is_bb_narrow, bb_width = self._is_bb_narrow(
            current_bb_upper, current_bb_lower, current_price)

        # 🆕 检查K线反转形态
        has_pattern, pattern = self._check_candle_pattern(df, verbose=verbose)
        self._last_pattern[ticker] = pattern

        # ========== 3. 计算盈亏 ==========
        pnl_pct = 0.0
        if current_position > 0 and avg_cost > 0:
            pnl_pct = (current_price - avg_cost) / avg_cost
            self._update_allocation_based_on_pnl(ticker, pnl_pct, market_state)

        # ========== 4. 生成信号 ==========
        signal = 'HOLD'
        confidence = 5
        reason = ""

        # --- 收盘强制平仓 ---
        if is_market_close and current_position > 0:
            signal = 'SELL'
            confidence = 10
            reason = "⏰ 收盘平仓"
            self._reset_invested(ticker)
            return self._make_result(signal, confidence, reason, current_price,
                                     market_state, current_adx, bb_position,
                                     current_allocation, bb_width, is_bb_narrow,
                                     pattern), df

        # --- 最后N分钟只允许平仓 ---
        is_last_n_min = self._is_last_n_minutes(current_time)
        if is_last_n_min and current_position == 0:
            reason = f"⏰ 收盘前{self.no_new_position_minutes}分钟，不开新仓"
            if verbose:
                print(f"   {reason}")
            return self._make_result('HOLD', 5, reason, current_price,
                                     market_state, current_adx, bb_position,
                                     current_allocation, bb_width, is_bb_narrow,
                                     pattern), df

        # --- 止损检查 ---
        if current_position > 0 and avg_cost > 0:
            stop_loss = self.quick_stop_loss if market_state == 'DOWNTREND' else self.normal_stop_loss

            if pnl_pct <= -stop_loss:
                signal = 'SELL'
                confidence = 10
                reason = f"🛑 止损! 亏损 {pnl_pct*100:.4f}% (阈值: {stop_loss*100:.4f}%)"

                self._reduce_allocation(ticker, "止损触发")
                self._start_cooldown(ticker, current_time, is_stop_loss=True)
                self._reset_invested(ticker)

                if verbose:
                    print(f"🛑 [止损] {ticker}: {reason}")

                return self._make_result(signal, confidence, reason, current_price,
                                         market_state, current_adx, bb_position,
                                         self.get_current_allocation(
                                             ticker), bb_width, is_bb_narrow,
                                         pattern), df

        # --- 检查冷却期 ---
        if current_position == 0:
            is_cooling, cooldown_reason = self._is_in_cooldown(
                ticker, current_time)
            if is_cooling:
                if verbose:
                    print(f"   {cooldown_reason}")
                return self._make_result('HOLD', 5, cooldown_reason, current_price,
                                         market_state, current_adx, bb_position,
                                         current_allocation, bb_width, is_bb_narrow,
                                         pattern), df

        # --- 检查布林带是否过窄 ---
        if is_bb_narrow and current_position == 0:
            if self.bb_narrow_action == 'BLOCK' and market_state != 'UPTREND':
                reason = f"📊 布林带过窄 ({bb_width*100:.2f}%)，暂停交易"
                if verbose:
                    print(f"   {reason}")
                return self._make_result('HOLD', 5, reason, current_price,
                                         market_state, current_adx, bb_position,
                                         current_allocation, bb_width, is_bb_narrow,
                                         pattern), df

        # --- 根据市场状态交易 ---
        if market_state == 'UPTREND':
            signal, confidence, reason = self._uptrend_strategy(
                current_position, avg_cost, current_price, bb_position, pnl_pct,
                has_pattern, pattern, verbose
            )

        elif market_state == 'RANGING':
            signal, confidence, reason = self._ranging_strategy(
                current_position, current_price, bb_position, pnl_pct,
                has_pattern, pattern, verbose
            )

        elif market_state == 'DOWNTREND':
            signal, confidence, reason = self._downtrend_strategy(
                current_position, current_price, bb_position, pnl_pct,
                has_pattern, pattern, verbose
            )

        else:  # UNCLEAR
            reason = "⚪ 市场不明朗，观望"

        # --- 🆕 形态提升信心 ---
        if has_pattern and signal == 'BUY' and self.pattern_boost_confidence:
            confidence = min(confidence + self.pattern_confidence_boost, 10)
            reason += f" 🕯️{pattern.pattern.value}"

        # --- 布林带窄幅时降低信心 ---
        if is_bb_narrow and signal == 'BUY' and self.bb_narrow_action == 'WARN':
            confidence = max(confidence - 2, 1)
            reason += f" ⚠️BB窄({bb_width*100:.1f}%)"

        # --- 检查是否可以开仓 ---
        if signal == 'BUY':
            can_open, open_reason = self._can_open_position(
                ticker, current_allocation)
            if not can_open:
                signal = 'HOLD'
                reason = open_reason
            elif open_reason:
                reason += f" {open_reason}"

        # --- 记录盈利 ---
        if signal == 'SELL' and current_position > 0 and pnl_pct > 0:
            self._record_profit(ticker)
            self._reset_invested(ticker)

        # --- 更新已投资比例 ---
        if signal == 'BUY':
            self._update_invested(ticker, current_allocation)
        elif signal == 'SELL':
            self._reset_invested(ticker)

        self._last_signal[ticker] = signal

        # ========== 5. 输出调试信息 ==========
        if verbose:
            state_emoji = {'UPTREND': '🟢', 'DOWNTREND': '🔴',
                           'RANGING': '🟡', 'UNCLEAR': '⚪'}
            signal_emoji = {'BUY': '💰', 'SELL': '💸', 'HOLD': '⏸️'}

            pos_str = f"持仓 {int(current_position)} 股" if current_position > 0 else "空仓"
            pnl_str = f" ({pnl_pct*100:+.2f}%)" if current_position > 0 else ""
            time_warning = f" ⚠️收盘前{self.no_new_position_minutes}分钟" if is_last_n_min else ""
            bb_warning = f" 📊BB窄" if is_bb_narrow else ""
            pattern_str = f" 🕯️{pattern.pattern.value}" if has_pattern else " 🕯️无形态"

            print(f"\nTIME: {current_time}")
            print(f"\n{state_emoji.get(market_state, '⚪')} [{market_state}] {ticker} | "
                  f"{pos_str}{pnl_str}{time_warning}{bb_warning}{pattern_str}")
            print(f"   价格: ${current_price:.2f} | BB: {bb_position*100:.0f}% | "
                  f"BB宽: {bb_width*100:.2f}% | ADX: {current_adx:.1f}")
            print(f"   📊 当前仓位比例: {current_allocation*100:.0f}% | "
                  f"已投资: {self._total_invested.get(ticker, 0)*100:.0f}%")
            print(f"   {signal_emoji.get(signal, '❓')} {signal} - {reason}")

        return self._make_result(signal, confidence, reason, current_price,
                                 market_state, current_adx, bb_position,
                                 current_allocation, bb_width, is_bb_narrow,
                                 pattern), df

    # ==================== 🆕 各市场状态策略（加入形态判断）====================

    def _uptrend_strategy(self, position: float, avg_cost: float,
                          price: float, bb_pos: float, pnl_pct: float,
                          has_pattern: bool, pattern: Optional[PatternResult],
                          verbose: bool = False) -> Tuple[str, int, str]:
        """上升趋势策略 - 需要K线反转形态确认"""
        if position == 0:
            # 布林带条件
            bb_condition = self.uptrend_buy_low <= bb_pos <= self.uptrend_buy_high

            if bb_condition:
                # 🆕 检查是否需要反转形态
                if self.require_candle_pattern:
                    if has_pattern:
                        return 'BUY', 8, f"🟢 上升趋势回调 + {pattern.pattern.value} (BB {bb_pos*100:.0f}%)"
                    else:
                        return 'HOLD', 5, f"⏳ BB位置OK，等待K线反转形态 (BB {bb_pos*100:.0f}%)"
                else:
                    # 不强制要求形态，有形态提升信心
                    base_reason = f"🟢 上升趋势回调买入 (BB {bb_pos*100:.0f}%)"
                    return 'BUY', 8, base_reason
            elif bb_pos < self.uptrend_buy_low:
                return 'HOLD', 5, f"回调过深，等待企稳"
            else:
                return 'HOLD', 5, f"等待回调"
        else:
            if avg_cost > 0 and pnl_pct >= self.uptrend_take_profit:
                return 'SELL', 8, f"🎯 止盈 +{pnl_pct*100:.1f}%"
            return 'HOLD', 5, f"持仓中 ({pnl_pct*100:+.1f}%)"

    def _ranging_strategy(self, position: float, price: float,
                          bb_pos: float, pnl_pct: float,
                          has_pattern: bool, pattern: Optional[PatternResult],
                          verbose: bool = False) -> Tuple[str, int, str]:
        """震荡市策略 - 需要K线反转形态确认"""
        if position == 0:
            bb_condition = bb_pos <= self.range_buy_threshold

            if bb_condition:
                # 🆕 检查是否需要反转形态
                if self.require_candle_pattern:
                    if has_pattern:
                        return 'BUY', 7, f"🟡 震荡低点 + {pattern.pattern.value} (BB {bb_pos*100:.0f}%)"
                    else:
                        return 'HOLD', 5, f"⏳ BB位置OK，等待K线反转形态 (BB {bb_pos*100:.0f}%)"
                else:
                    return 'BUY', 7, f"🟡 震荡低点买入 (BB {bb_pos*100:.0f}%)"
            else:
                return 'HOLD', 5, f"等待低点"
        else:
            if pnl_pct >= self.range_take_profit or bb_pos >= self.range_sell_threshold:
                return 'SELL', 7, f"🟡 震荡高点卖出 (BB {bb_pos*100:.0f}%)"
            else:
                return 'HOLD', 5, f"持仓等待高点"

    def _downtrend_strategy(self, position: float, price: float,
                            bb_pos: float, pnl_pct: float,
                            has_pattern: bool, pattern: Optional[PatternResult],
                            verbose: bool = False) -> Tuple[str, int, str]:
        """下降趋势策略 - 需要K线反转形态确认"""
        if position == 0:
            bb_condition = bb_pos <= self.downtrend_buy_threshold

            if bb_condition:
                # 🆕 下降趋势更严格，必须有形态
                if has_pattern:
                    return 'BUY', 6, f"🔴 下降趋势抄底 + {pattern.pattern.value} (BB {bb_pos*100:.0f}%)"
                else:
                    return 'HOLD', 5, f"⏳ 下降趋势，需要反转形态确认 (BB {bb_pos*100:.0f}%)"
            else:
                return 'HOLD', 5, f"⚠️ 下降趋势，等待极低点"
        else:
            if pnl_pct >= self.downtrend_take_profit or bb_pos >= self.downtrend_sell_threshold:
                return 'SELL', 7, f"🔴 下降趋势快速止盈 (BB {bb_pos*100:.0f}%)"
            else:
                return 'HOLD', 5, f"持仓观望"

    def _make_result(self, signal: str, confidence: int, reason: str,
                     price: float, market_state: str, adx: float,
                     bb_position: float, allocation: float = 1.0,
                     bb_width: float = 0.0, is_bb_narrow: bool = False,
                     pattern: Optional[PatternResult] = None) -> Dict:
        """构建返回结果"""
        result = {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'price': price,
            'market_state': market_state,
            'adx': adx,
            'bb_position': bb_position,
            'allocation': allocation,
            'bb_width': bb_width,
            'is_bb_narrow': is_bb_narrow,
            # 🆕 添加形态信息
            'pattern': pattern.pattern.value if pattern else None,
            'pattern_strength': pattern.strength if pattern else 0.0,
        }
        return result

    def get_history_data(self, ticker: str) -> pd.DataFrame:
        """获取带指标的历史数据"""
        if ticker not in self._history_data or self._history_data[ticker].empty:
            return pd.DataFrame()

        df = self._history_data[ticker].copy()

        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='last')]

        close = df['close']

        bb_middle = close.rolling(window=self.bb_period, min_periods=1).mean()
        bb_std = close.rolling(window=self.bb_period, min_periods=1).std()

        df['SMA'] = bb_middle.values
        df['BB_UPPER'] = (bb_middle + self.bb_std_dev * bb_std).values
        df['BB_LOWER'] = (bb_middle - self.bb_std_dev * bb_std).values
        df['EMA_FAST'] = self._calculate_ema(close, self.ema_fast).values
        df['EMA_SLOW'] = self._calculate_ema(close, self.ema_slow).values
        df['ADX'] = self._calculate_adx(df, self.adx_period)

        for col in ['SMA', 'BB_UPPER', 'BB_LOWER', 'EMA_FAST', 'EMA_SLOW']:
            df[col] = df[col].bfill()
        df['ADX'] = df['ADX'].fillna(0)

        return df

    def get_last_pattern(self, ticker: str) -> Optional[PatternResult]:
        """获取最后识别的K线形态"""
        return self._last_pattern.get(ticker)

    def reset_state(self, ticker: str = None):
        """重置策略状态"""
        if ticker:
            for d in [self._history_data, self._current_allocation, self._peak_equity,
                      self._last_pnl_state, self._stop_loss_time, self._stop_loss_bar_count,
                      self._bar_count, self._consecutive_losses, self._current_cooldown_multiplier,
                      self._last_signal, self._total_invested, self._last_pattern]:
                if ticker in d:
                    del d[ticker]
        else:
            self._history_data.clear()
            self._current_allocation.clear()
            self._peak_equity.clear()
            self._last_pnl_state.clear()
            self._stop_loss_time.clear()
            self._stop_loss_bar_count.clear()
            self._bar_count.clear()
            self._consecutive_losses.clear()
            self._current_cooldown_multiplier.clear()
            self._last_signal.clear()
            self._total_invested.clear()
            self._last_pattern.clear()


# ==================== 测试代码 ====================

if __name__ == '__main__':
    # 创建测试数据
    import numpy as np

    print("=" * 60)
    print("测试 K线形态识别")
    print("=" * 60)

    # 创建模拟数据
    dates = pd.date_range('2024-01-01', periods=20, freq='5min')

    # 模拟一个锤子线形态
    np.random.seed(42)
    data = {
        'open': [100, 99, 98, 97, 96, 95, 94, 93, 92, 91,
                 90, 89, 88, 87, 86, 85, 84, 83, 82, 81],
        'high': [101, 100, 99, 98, 97, 96, 95, 94, 93, 92,
                 91, 90, 89, 88, 87, 86, 85, 84, 83, 85],  # 最后一根高点
        'low':  [99, 98, 97, 96, 95, 94, 93, 92, 91, 90,
                 89, 88, 87, 86, 85, 84, 83, 82, 78, 79],   # 最后一根长下影线
        'close': [99.5, 98.5, 97.5, 96.5, 95.5, 94.5, 93.5, 92.5, 91.5, 90.5,
                  89.5, 88.5, 87.5, 86.5, 85.5, 84.5, 83.5, 82.5, 81.5, 84.5],  # 锤子线收盘
        'volume': [1000] * 20
    }

    df = pd.DataFrame(data, index=dates)

    # 测试形态识别器
    recognizer = CandlePatternRecognizer()
    patterns = recognizer.recognize(df)

    print(f"\n识别到 {len(patterns)} 个形态:")
    for p in patterns:
        print(f"  - {p.pattern.value}: 强度 {p.strength:.0%}")

    # 测试策略
    print("\n" + "=" * 60)
    print("测试策略信号")
    print("=" * 60)

    strategy = SimpleUpTrendStrategy(
        require_candle_pattern=True,
        pattern_min_strength=0.3,
        verbose_init=True
    )

    signal_data, _ = strategy.get_signal(
        ticker='TEST',
        new_data=df,
        current_position=0,
        current_price=84.5,
        verbose=True
    )

    print(f"\n最终信号: {signal_data['signal']}")
    print(f"原因: {signal_data['reason']}")
    if signal_data['pattern']:
        print(
            f"识别形态: {signal_data['pattern']} (强度: {signal_data['pattern_strength']:.0%})")
