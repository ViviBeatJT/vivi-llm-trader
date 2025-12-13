# src/strategies/simple_trend_strategy.py

"""
简单趋势策略 - Simple Trend Strategy (Long Only)

核心思想：
1. 只做多（BUY/SELL），不做空
2. 检测市场状态（上升趋势 / 震荡 / 下降趋势）
3. 上升趋势 → 回调买入
4. 震荡市场 → 低买高卖
5. 下降趋势 → 持仓观望，动态止损

动态仓位管理：
- 亏损超过 quick_stop_loss → 立即止损
- 亏损超过 reduce_allocation_threshold → 下次交易减仓
- 盈利后逐步恢复仓位

v2 改进：
- 添加止损冷却期，止损后不会立即开仓

v3 改进：
- 最后10分钟只允许平仓，禁止开新仓

v4 改进：
- 🆕 连续亏损冷却期加长
- 🆕 布林带窄幅时谨慎交易
- 🆕 防止连续开仓超过 max_allocation
- 🆕 更完善的状态跟踪
"""

from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np


class SimpleUpTrendStrategy:
    """
    简单趋势策略 - 只做多 + 动态仓位管理

    市场状态判断：
    - ADX > 25 且 EMA快 > EMA慢 → 上升趋势 ✅ 可交易
    - ADX > 25 且 EMA快 < EMA慢 → 下降趋势 ⚠️ 持仓观望
    - ADX < 20 → 震荡市场 ✅ 可交易

    动态仓位管理：
    - 亏损 > quick_stop_loss (0.5%) → 立即止损
    - 亏损 > reduce_threshold (1%) → allocation 减半
    - 盈利 > recovery_threshold (0.5%) → allocation 恢复一档

    冷却期机制：
    - 止损后进入冷却期，期间不开新仓
    - 连续亏损时冷却期加长
    - 冷却期可以按时间或K线数量计算

    收盘保护：
    - 最后10分钟只允许平仓，不允许开新仓

    布林带保护：
    - 布林带过窄时谨慎交易（波动率低，可能即将突破）
    """

    def __init__(self,
                 # 布林带参数
                 bb_period: int = 20,
                 bb_std_dev: float = 2.0,

                 # ADX 参数（趋势强度）
                 adx_period: int = 14,
                 adx_trend_threshold: float = 25,
                 adx_range_threshold: float = 20,

                 # EMA 参数（趋势方向）
                 ema_fast: int = 12,
                 ema_slow: int = 26,

                 # 上升趋势交易参数
                 uptrend_buy_low: float = 0.40,
                 uptrend_buy_high: float = 0.60,
                 uptrend_take_profit: float = 0.005,

                 # 震荡市交易参数
                 range_buy_threshold: float = 0.20,
                 range_sell_threshold: float = 0.55,
                 range_take_profit: float = 0.003,

                # 下降趋势交易参数
                 downtrend_buy_threshold : float = 0.05,
                 downtrend_sell_threshold : float = 0.40,
                 downtrend_take_profit: float = 0.001,

                 # ===== 动态仓位管理参数 =====
                 # 止损参数
                 quick_stop_loss: float = 0.0005,   # 0.5% 快速止损（下降趋势时）
                 normal_stop_loss: float = 0.001,   # 0.1% 正常止损

                 # 仓位调整参数
                 reduce_allocation_threshold: float = 0.01,  # 亏损 1% 时减仓
                 reduce_allocation_ratio: float = 0.5,       # 减到原来的 50%
                 recovery_threshold: float = 0.005,          # 盈利 0.5% 开始恢复
                 recovery_step: float = 0.1,                 # 每次恢复 10%
                 min_allocation: float = 0.25,               # 最小仓位 25%
                 max_allocation: float = 1.0,                # 最大仓位 100%

                 # ===== 冷却期参数 =====
                 cooldown_bars: int = 5,                     # 止损后冷却 5 根K线
                 cooldown_minutes: int = 0,                  # 或者冷却 N 分钟（0表示用K线数）

                 # 🆕 连续亏损冷却期加长参数
                 consecutive_loss_multiplier: float = 1.5,   # 每次连续亏损，冷却期乘以这个系数
                 max_cooldown_multiplier: float = 4.0,       # 最大冷却期倍数
                 consecutive_loss_reset_after_profit: bool = True,  # 盈利后重置连续亏损计数

                 # ===== 收盘保护参数 =====
                 no_new_position_minutes: int = 10,          # 收盘前N分钟禁止开新仓
                 market_close_time: time = time(16, 0),      # 美股收盘时间 (ET)

                 # ===== 🆕 布林带保护参数 =====
                 bb_narrow_threshold: float = 0.02,          # BB宽度 < 价格的1% 视为过窄
                 bb_narrow_action: str = 'BLOCK',             # 'WARN' 降低信心, 'BLOCK' 禁止交易

                 # 其他
                 max_history_bars: int = 500,
                 verbose_init: bool = True):                 # 是否打印初始化信息

        # 保存参数
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
        
        # 止损参数
        self.quick_stop_loss = quick_stop_loss
        self.normal_stop_loss = normal_stop_loss

        # 仓位管理参数
        self.reduce_allocation_threshold = reduce_allocation_threshold
        self.reduce_allocation_ratio = reduce_allocation_ratio
        self.recovery_threshold = recovery_threshold
        self.recovery_step = recovery_step
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation

        # 冷却期参数
        self.cooldown_bars = cooldown_bars
        self.cooldown_minutes = cooldown_minutes

        # 🆕 连续亏损参数
        self.consecutive_loss_multiplier = consecutive_loss_multiplier
        self.max_cooldown_multiplier = max_cooldown_multiplier
        self.consecutive_loss_reset_after_profit = consecutive_loss_reset_after_profit

        # 收盘保护参数
        self.no_new_position_minutes = no_new_position_minutes
        self.market_close_time = market_close_time

        # 🆕 布林带保护参数
        self.bb_narrow_threshold = bb_narrow_threshold
        self.bb_narrow_action = bb_narrow_action

        self.max_history_bars = max_history_bars

        # 数据存储
        self._history_data: Dict[str, pd.DataFrame] = {}

        # ===== 动态仓位状态 =====
        self._current_allocation: Dict[str, float] = {}  # 当前仓位比例
        self._peak_equity: Dict[str, float] = {}         # 最高权益（用于计算回撤）
        self._last_pnl_state: Dict[str, str] = {}        # 上次盈亏状态

        # ===== 冷却期状态 =====
        self._stop_loss_time: Dict[str, datetime] = {}   # 止损时间
        self._stop_loss_bar_count: Dict[str, int] = {}   # 止损时的K线计数
        self._bar_count: Dict[str, int] = {}             # 当前K线计数

        # 🆕 连续亏损状态
        self._consecutive_losses: Dict[str, int] = {}    # 连续亏损次数
        self._current_cooldown_multiplier: Dict[str, float] = {}  # 当前冷却期倍数

        # 🆕 交易状态跟踪（防止连续开仓）
        self._last_signal: Dict[str, str] = {}           # 上次信号
        self._total_invested: Dict[str, float] = {}      # 当前总投资比例

        # 打印配置
        if verbose_init:
            self._print_config()

    def _print_config(self):
        """打印策略配置"""
        print(f"\n{'='*60}")
        print(f"📈 简单趋势策略 v4 (只做多 + 动态仓位管理 + 冷却期)")
        print(f"{'='*60}")
        print(f"趋势判断:")
        print(f"  ADX > {self.adx_trend_threshold} = 趋势市")
        print(f"  ADX < {self.adx_range_threshold} = 震荡市")
        print(f"\n交易参数:")
        print(
            f"  上升趋势买入: BB {self.uptrend_buy_low*100:.0f}%-{self.uptrend_buy_high*100:.0f}%")
        print(f"  震荡买入: BB < {self.range_buy_threshold*100:.0f}%")
        print(f"\n动态仓位管理:")
        print(f"  🛑 快速止损: {self.quick_stop_loss*100:.4f}% (下降趋势)")
        print(f"  🛑 正常止损: {self.normal_stop_loss*100:.4f}%")
        print(f"  🛑 止盈: {self.uptrend_take_profit*100:.4f}%")
        print(f"  📉 减仓触发: 亏损 > {self.reduce_allocation_threshold*100:.1f}%")
        print(f"  📉 减仓比例: 减到 {self.reduce_allocation_ratio*100:.0f}%")
        print(f"  📈 恢复触发: 盈利 > {self.recovery_threshold*100:.1f}%")
        print(f"  📈 恢复步长: 每次 +{self.recovery_step*100:.0f}%")
        print(
            f"  📊 仓位范围: {self.min_allocation*100:.0f}% - {self.max_allocation*100:.0f}%")
        print(f"\n⏳ 冷却期:")
        if self.cooldown_minutes > 0:
            print(f"  止损后冷却: {self.cooldown_minutes} 分钟")
        else:
            print(f"  止损后冷却: {self.cooldown_bars} 根K线")
        print(f"  🆕 连续亏损冷却期倍数: {self.consecutive_loss_multiplier}x")
        print(f"  🆕 最大冷却期倍数: {self.max_cooldown_multiplier}x")
        print(f"\n⏰ 收盘保护:")
        print(f"  收盘前 {self.no_new_position_minutes} 分钟禁止开新仓")
        print(f"  收盘时间: {self.market_close_time.strftime('%H:%M')} ET")
        print(f"\n📊 布林带保护:")
        print(f"  窄幅阈值: {self.bb_narrow_threshold*100:.1f}%")
        print(f"  窄幅处理: {self.bb_narrow_action}")
        print(f"{'='*60}\n")

    # ==================== 收盘保护方法 ====================

    def _is_last_n_minutes(self, current_time, minutes: int = None) -> bool:
        """
        检查是否在收盘前N分钟内
        """
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

    # ==================== 🆕 布林带保护方法 ====================

    def _calculate_bb_width(self, bb_upper: float, bb_lower: float, price: float) -> float:
        """
        计算布林带宽度占价格的百分比
        """
        if price <= 0:
            return 0.0
        return (bb_upper - bb_lower) / price

    def _is_bb_narrow(self, bb_upper: float, bb_lower: float, price: float) -> Tuple[bool, float]:
        """
        检查布林带是否过窄

        Returns:
            (is_narrow, width_pct): 是否过窄，宽度百分比
        """
        width_pct = self._calculate_bb_width(bb_upper, bb_lower, price)
        return width_pct < self.bb_narrow_threshold, width_pct

    # ==================== 🆕 连续亏损管理方法 ====================

    def _record_loss(self, ticker: str):
        """记录一次亏损"""
        self._consecutive_losses[ticker] = self._consecutive_losses.get(
            ticker, 0) + 1

        # 计算新的冷却期倍数
        losses = self._consecutive_losses[ticker]
        new_multiplier = min(
            self.consecutive_loss_multiplier ** (losses - 1),
            self.max_cooldown_multiplier
        )
        self._current_cooldown_multiplier[ticker] = new_multiplier

        return losses, new_multiplier

    def _record_profit(self, ticker: str):
        """记录一次盈利，可选择重置连续亏损计数"""
        if self.consecutive_loss_reset_after_profit:
            self._consecutive_losses[ticker] = 0
            self._current_cooldown_multiplier[ticker] = 1.0

    def _get_effective_cooldown(self, ticker: str) -> Tuple[int, int]:
        """
        获取有效冷却期（考虑连续亏损）

        Returns:
            (effective_bars, effective_minutes): 有效的K线数和分钟数
        """
        multiplier = self._current_cooldown_multiplier.get(ticker, 1.0)
        effective_bars = int(self.cooldown_bars * multiplier)
        effective_minutes = int(self.cooldown_minutes * multiplier)
        return effective_bars, effective_minutes

    def get_consecutive_losses(self, ticker: str) -> int:
        """获取连续亏损次数"""
        return self._consecutive_losses.get(ticker, 0)

    def get_cooldown_multiplier(self, ticker: str) -> float:
        """获取当前冷却期倍数"""
        return self._current_cooldown_multiplier.get(ticker, 1.0)

    # ==================== 冷却期管理方法 ====================

    def _start_cooldown(self, ticker: str, current_time: datetime = None, is_stop_loss: bool = True):
        """开始冷却期"""
        if current_time is None:
            current_time = datetime.now()

        self._stop_loss_time[ticker] = current_time
        self._stop_loss_bar_count[ticker] = self._bar_count.get(ticker, 0)

        # 🆕 如果是止损，记录亏损并可能加长冷却期
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
        else:
            if self.cooldown_minutes > 0:
                print(f"   ⏳ [冷却期开始] {ticker}: 等待 {self.cooldown_minutes} 分钟")
            else:
                print(f"   ⏳ [冷却期开始] {ticker}: 等待 {self.cooldown_bars} 根K线")

    def _is_in_cooldown(self, ticker: str, current_time: datetime = None) -> Tuple[bool, str]:
        """
        检查是否在冷却期内（考虑连续亏损加长）
        """
        if ticker not in self._stop_loss_time:
            return False, ""

        if current_time is None:
            current_time = datetime.now()

        # 🆕 获取有效冷却期
        effective_bars, effective_minutes = self._get_effective_cooldown(
            ticker)

        # 按时间计算冷却期
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

        # 按K线数量计算冷却期
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

    def _clear_cooldown(self, ticker: str):
        """清除冷却期状态"""
        if ticker in self._stop_loss_time:
            del self._stop_loss_time[ticker]
        if ticker in self._stop_loss_bar_count:
            del self._stop_loss_bar_count[ticker]

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

    # ==================== 🆕 防止连续开仓方法 ====================

    def _can_open_position(self, ticker: str, requested_allocation: float) -> Tuple[bool, str]:
        """
        检查是否可以开仓（防止超过 max_allocation）

        Args:
            ticker: 股票代码
            requested_allocation: 请求的仓位比例

        Returns:
            (can_open, reason): 是否可以开仓，原因
        """
        current_invested = self._total_invested.get(ticker, 0.0)

        if current_invested >= self.max_allocation:
            return False, f"🚫 已达最大仓位 {self.max_allocation*100:.0f}%"

        # 检查加上请求的仓位后是否超过最大值
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
        """重置已投资比例（平仓后）"""
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

        # 🆕 检查布林带宽度
        is_bb_narrow, bb_width = self._is_bb_narrow(
            current_bb_upper, current_bb_lower, current_price)

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
                                     current_allocation, bb_width, is_bb_narrow), df

        # --- 最后N分钟只允许平仓 ---
        is_last_n_min = self._is_last_n_minutes(current_time)
        if is_last_n_min and current_position == 0:
            reason = f"⏰ 收盘前{self.no_new_position_minutes}分钟，不开新仓"
            if verbose:
                print(f"   {reason}")
            return self._make_result('HOLD', 5, reason, current_price,
                                     market_state, current_adx, bb_position,
                                     current_allocation, bb_width, is_bb_narrow), df

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
                                         self.get_current_allocation(ticker), bb_width, is_bb_narrow), df

        # --- 检查冷却期（只在空仓时）---
        if current_position == 0:
            is_cooling, cooldown_reason = self._is_in_cooldown(
                ticker, current_time)
            if is_cooling:
                if verbose:
                    print(f"   {cooldown_reason}")
                return self._make_result('HOLD', 5, cooldown_reason, current_price,
                                         market_state, current_adx, bb_position,
                                         current_allocation, bb_width, is_bb_narrow), df

        # --- 🆕 检查布林带是否过窄 ---
        if is_bb_narrow and current_position == 0:
            if self.bb_narrow_action == 'BLOCK':
                reason = f"📊 布林带过窄 ({bb_width*100:.2f}%)，暂停交易"
                if verbose:
                    print(f"   {reason}")
                return self._make_result('HOLD', 5, reason, current_price,
                                         market_state, current_adx, bb_position,
                                         current_allocation, bb_width, is_bb_narrow), df

        # --- 根据市场状态交易 ---
        if market_state == 'UPTREND':
            signal, confidence, reason = self._uptrend_strategy(
                current_position, avg_cost, current_price, bb_position, pnl_pct
            )

        elif market_state == 'RANGING':
            signal, confidence, reason = self._ranging_strategy(
                current_position, current_price, bb_position, pnl_pct
            )

        elif market_state == 'DOWNTREND':
            signal, confidence, reason = self._downtrend_strategy(
                current_position, current_price, bb_position, pnl_pct
            )

        else:  # UNCLEAR
            reason = "⚪ 市场不明朗，观望"

        # --- 🆕 布林带窄幅时降低信心 ---
        if is_bb_narrow and signal == 'BUY' and self.bb_narrow_action == 'WARN':
            confidence = max(confidence - 2, 1)
            reason += f" ⚠️BB窄({bb_width*100:.1f}%)"

        # --- 🆕 检查是否可以开仓（防止超过 max_allocation）---
        if signal == 'BUY':
            can_open, open_reason = self._can_open_position(
                ticker, current_allocation)
            if not can_open:
                signal = 'HOLD'
                reason = open_reason
            elif open_reason:
                reason += f" {open_reason}"

        # --- 🆕 记录盈利（用于重置连续亏损）---
        if signal == 'SELL' and current_position > 0 and pnl_pct > 0:
            self._record_profit(ticker)
            self._reset_invested(ticker)

        # --- 🆕 更新已投资比例 ---
        if signal == 'BUY':
            self._update_invested(ticker, current_allocation)
        elif signal == 'SELL':
            self._reset_invested(ticker)

        # --- 🆕 记录上次信号 ---
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
            print(f"\nTIME: {current_time}")
            print(f"\n{state_emoji.get(market_state, '⚪')} [{market_state}] {ticker} | "
                  f"{pos_str}{pnl_str}{time_warning}{bb_warning}")
            print(f"   价格: ${current_price:.2f} | BB: {bb_position*100:.0f}% | "
                  f"BB宽: {bb_width*100:.2f}% | ADX: {current_adx:.1f}")
            print(f"   📊 当前仓位比例: {current_allocation*100:.0f}% | "
                  f"已投资: {self._total_invested.get(ticker, 0)*100:.0f}%")
            print(f"   🔄 连续亏损: {self._consecutive_losses.get(ticker, 0)}次 | "
                  f"冷却倍数: {self._current_cooldown_multiplier.get(ticker, 1.0):.1f}x")
            print(f"   {signal_emoji.get(signal, '❓')} {signal} - {reason}")

        return self._make_result(signal, confidence, reason, current_price,
                                 market_state, current_adx, bb_position,
                                 current_allocation, bb_width, is_bb_narrow), df

    # ==================== 各市场状态策略 ====================

    def _uptrend_strategy(self, position: float, avg_cost: float,
                          price: float, bb_pos: float, pnl_pct: float) -> Tuple[str, int, str]:
        """上升趋势策略"""
        if position == 0:
            if self.uptrend_buy_low <= bb_pos <= self.uptrend_buy_high:
                return 'BUY', 8, f"🟢 上升趋势回调买入 (BB {bb_pos*100:.0f}%)"
            elif bb_pos < self.uptrend_buy_low:
                return 'HOLD', 5, f"回调过深，等待企稳"
            else:
                return 'HOLD', 5, f"等待回调"
        else:
            if avg_cost > 0 and pnl_pct >= self.uptrend_take_profit:
                return 'SELL', 8, f"🎯 止盈 +{pnl_pct*100:.1f}%"
            return 'HOLD', 5, f"持仓中 ({pnl_pct*100:+.1f}%)"

    def _ranging_strategy(self, position: float, price: float,
                          bb_pos: float, pnl_pct: float) -> Tuple[str, int, str]:
        """震荡市策略"""
        if position == 0:
            if bb_pos <= self.range_buy_threshold:
                return 'BUY', 7, f"🟡 震荡低点买入 (BB {bb_pos*100:.0f}%)"
            else:
                return 'HOLD', 5, f"等待低点"
        else:
            if pnl_pct >= self.range_take_profit or bb_pos >= self.range_sell_threshold:
                return 'SELL', 7, f"🟡 震荡高点卖出 (BB {bb_pos*100:.0f}%)"
            else:
                return 'HOLD', 5, f"持仓等待高点"

    def _downtrend_strategy(self, position: float, price: float,
                          bb_pos: float, pnl_pct: float) -> Tuple[str, int, str]:
        """震荡市策略"""
        if position == 0:
            if bb_pos <= self.downtrend_buy_threshold:
                return 'BUY', 7, f"🟡 震荡低点买入 (BB {bb_pos*100:.0f}%)"
            else:
                return 'HOLD', 5, f"等待低点"
        else:
            if pnl_pct >= self.downtrend_take_profit or bb_pos >= self.downtrend_sell_threshold:
                return 'SELL', 7, f"🟡 震荡高点卖出 (BB {bb_pos*100:.0f}%)"
            else:
                return 'HOLD', 5, f"持仓等待高点"

    def _make_result(self, signal: str, confidence: int, reason: str,
                     price: float, market_state: str, adx: float,
                     bb_position: float, allocation: float = 1.0,
                     bb_width: float = 0.0, is_bb_narrow: bool = False) -> Dict:
        """构建返回结果"""
        return {
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
        }

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

    # ==================== 状态重置方法（用于测试）====================

    def reset_state(self, ticker: str = None):
        """重置策略状态（用于测试）"""
        if ticker:
            # 重置特定 ticker
            for d in [self._history_data, self._current_allocation, self._peak_equity,
                      self._last_pnl_state, self._stop_loss_time, self._stop_loss_bar_count,
                      self._bar_count, self._consecutive_losses, self._current_cooldown_multiplier,
                      self._last_signal, self._total_invested]:
                if ticker in d:
                    del d[ticker]
        else:
            # 重置所有
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
