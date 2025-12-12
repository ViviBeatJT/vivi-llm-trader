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
"""

from typing import Dict, Tuple, Optional
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
                 uptrend_take_profit: float = 0.03,

                 # 震荡市交易参数
                 range_buy_threshold: float = 0.20,
                 range_sell_threshold: float = 0.55,

                 # ===== 动态仓位管理参数 =====
                 # 止损参数
                 quick_stop_loss: float = 0.0005,  # 0.05% 快速止损（下降趋势时）
                 normal_stop_loss: float = 0.001,  # 0.1% 正常止损

                 # 仓位调整参数
                 reduce_allocation_threshold: float = 0.001,  # 亏损 0.1% 时减仓
                 reduce_allocation_ratio: float = 0.5,       # 减到原来的 50%
                 recovery_threshold: float = 0.005,          # 盈利 0.5% 开始恢复
                 recovery_step: float = 0.1,                 # 每次恢复 10%
                 min_allocation: float = 0.25,               # 最小仓位 25%
                 max_allocation: float = 1.0,                # 最大仓位 100%

                 # 其他
                 max_history_bars: int = 500):

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

        self.max_history_bars = max_history_bars

        # 数据存储
        self._history_data: Dict[str, pd.DataFrame] = {}

        # ===== 动态仓位状态 =====
        self._current_allocation: Dict[str, float] = {}  # 当前仓位比例
        self._peak_equity: Dict[str, float] = {}         # 最高权益（用于计算回撤）
        # 上次盈亏状态 ('profit', 'loss', 'neutral')
        self._last_pnl_state: Dict[str, str] = {}

        # 打印配置
        print(f"\n{'='*60}")
        print(f"📈 简单趋势策略 (只做多 + 动态仓位管理)")
        print(f"{'='*60}")
        print(f"趋势判断:")
        print(f"  ADX > {adx_trend_threshold} = 趋势市")
        print(f"  ADX < {adx_range_threshold} = 震荡市")
        print(f"\n交易参数:")
        print(
            f"  上升趋势买入: BB {uptrend_buy_low*100:.0f}%-{uptrend_buy_high*100:.0f}%")
        print(f"  震荡买入: BB < {range_buy_threshold*100:.0f}%")
        print(f"\n动态仓位管理:")
        print(f"  🛑 快速止损: {quick_stop_loss*100:.1f}% (下降趋势)")
        print(f"  🛑 正常止损: {normal_stop_loss*100:.1f}%")
        print(f"  📉 减仓触发: 亏损 > {reduce_allocation_threshold*100:.1f}%")
        print(f"  📉 减仓比例: 减到 {reduce_allocation_ratio*100:.0f}%")
        print(f"  📈 恢复触发: 盈利 > {recovery_threshold*100:.1f}%")
        print(f"  📈 恢复步长: 每次 +{recovery_step*100:.0f}%")
        print(
            f"  📊 仓位范围: {min_allocation*100:.0f}% - {max_allocation*100:.0f}%")
        print(f"{'='*60}\n")

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
        print(f"   🔄 [重置仓位] {ticker}: 恢复到 {self.max_allocation*100:.0f}%")

    def _update_allocation_based_on_pnl(self, ticker: str, pnl_pct: float, market_state: str):
        """
        根据盈亏情况动态调整仓位

        Args:
            ticker: 股票代码
            pnl_pct: 当前持仓盈亏百分比
            market_state: 市场状态
        """
        current_state = 'neutral'

        if pnl_pct <= -self.reduce_allocation_threshold:
            current_state = 'loss'
        elif pnl_pct >= self.recovery_threshold:
            current_state = 'profit'

        last_state = self._last_pnl_state.get(ticker, 'neutral')

        # 状态变化时调整仓位
        if current_state == 'loss' and last_state != 'loss':
            # 进入亏损状态 → 减仓
            self._reduce_allocation(ticker, f"亏损 {pnl_pct*100:.2f}%")

        elif current_state == 'profit' and last_state == 'loss':
            # 从亏损恢复到盈利 → 逐步恢复仓位
            self._recover_allocation(ticker)

        elif current_state == 'profit' and pnl_pct >= self.uptrend_take_profit:
            # 盈利超过止盈目标 → 完全恢复仓位
            if self.get_current_allocation(ticker) < self.max_allocation:
                self._reset_allocation(ticker)

        self._last_pnl_state[ticker] = current_state

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
        """
        获取交易信号
        """

        # ========== 1. 更新历史数据 ==========
        if ticker not in self._history_data or self._history_data[ticker].empty:
            self._history_data[ticker] = new_data.copy()
        else:
            combined = pd.concat([self._history_data[ticker], new_data])
            combined = combined[~combined.index.duplicated(keep='last')]
            self._history_data[ticker] = combined.tail(self.max_history_bars)

        df = self._history_data[ticker]

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

        # ========== 3. 计算盈亏 ==========
        pnl_pct = 0.0
        if current_position > 0 and avg_cost > 0:
            pnl_pct = (current_price - avg_cost) / avg_cost
            # 动态调整仓位
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
            return self._make_result(signal, confidence, reason, current_price,
                                     market_state, current_adx, bb_position, current_allocation), df

        # --- 止损检查（根据市场状态使用不同阈值）---
        if current_position > 0 and avg_cost > 0:
            # 下降趋势使用快速止损
            stop_loss = self.quick_stop_loss if market_state == 'DOWNTREND' else self.normal_stop_loss

            if pnl_pct <= -stop_loss:
                signal = 'SELL'
                confidence = 10
                reason = f"🛑 止损! 亏损 {pnl_pct*100:.2f}% (阈值: {stop_loss*100:.1f}%)"

                # 止损后减仓
                self._reduce_allocation(ticker, "止损触发")

                if verbose:
                    print(f"🛑 [止损] {ticker}: {reason}")

                return self._make_result(signal, confidence, reason, current_price,
                                         market_state, current_adx, bb_position,
                                         self.get_current_allocation(ticker)), df

        # --- 根据市场状态交易 ---

        if market_state == 'UPTREND':
            signal, confidence, reason = self._uptrend_strategy(
                current_position, avg_cost, current_price, bb_position, pnl_pct
            )

        elif market_state == 'RANGING':
            signal, confidence, reason = self._ranging_strategy(
                current_position, current_price, bb_position
            )

        elif market_state == 'DOWNTREND':
            # 🔴 下降趋势 - 持仓观望，不急着卖
            signal, confidence, reason = self._downtrend_strategy(
                current_position, avg_cost, current_price, pnl_pct
            )

        else:  # UNCLEAR
            reason = "⚪ 市场不明朗，观望"

        # ========== 5. 输出调试信息 ==========
        if verbose:
            state_emoji = {'UPTREND': '🟢', 'DOWNTREND': '🔴',
                           'RANGING': '🟡', 'UNCLEAR': '⚪'}
            signal_emoji = {'BUY': '💰', 'SELL': '💸', 'HOLD': '⏸️'}

            pos_str = f"持仓 {int(current_position)} 股" if current_position > 0 else "空仓"
            pnl_str = f" ({pnl_pct*100:+.2f}%)" if current_position > 0 else ""

            print(
                f"\n{state_emoji.get(market_state, '⚪')} [{market_state}] {ticker} | {pos_str}{pnl_str}")
            print(
                f"   价格: ${current_price:.2f} | BB: {bb_position*100:.0f}% | ADX: {current_adx:.1f}")
            print(f"   📊 当前仓位比例: {current_allocation*100:.0f}%")
            print(f"   {signal_emoji.get(signal, '❓')} {signal} - {reason}")

        return self._make_result(signal, confidence, reason, current_price,
                                 market_state, current_adx, bb_position, current_allocation), df

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
                          bb_pos: float) -> Tuple[str, int, str]:
        """震荡市策略"""

        if position == 0:
            if bb_pos <= self.range_buy_threshold:
                return 'BUY', 7, f"🟡 震荡低点买入 (BB {bb_pos*100:.0f}%)"
            else:
                return 'HOLD', 5, f"等待低点"
        else:
            if bb_pos >= self.range_sell_threshold:
                return 'SELL', 7, f"🟡 震荡高点卖出 (BB {bb_pos*100:.0f}%)"
            else:
                return 'HOLD', 5, f"持仓等待高点"

    def _downtrend_strategy(self, position: float, avg_cost: float,
                            price: float, pnl_pct: float) -> Tuple[str, int, str]:
        """
        下降趋势策略 - 持仓观望，不急着卖

        - 不开新仓
        - 有仓位时观望，除非触发止损
        - 如果有盈利且盈利开始缩小，可以考虑保住利润
        """

        if position == 0:
            # 没有仓位 - 不开新仓
            return 'HOLD', 5, "📉 下降趋势，不开新仓"

        else:
            # 有仓位 - 观望，让止损逻辑处理
            if pnl_pct > self.uptrend_take_profit:
                # 盈利超过目标，可以卖
                return 'SELL', 7, f"📉 下降趋势，锁定利润 (+{pnl_pct*100:.1f}%)"

            elif pnl_pct > 0:
                # 小盈利，继续观望
                return 'HOLD', 5, f"📉 下降趋势，小盈利观望 (+{pnl_pct*100:.1f}%)"

            else:
                # 亏损中，等待止损触发或市场转好
                return 'HOLD', 5, f"📉 下降趋势，持仓观望 ({pnl_pct*100:.1f}%)"

    def _make_result(self, signal: str, confidence: int, reason: str,
                     price: float, market_state: str, adx: float,
                     bb_position: float, allocation: float = 1.0) -> Dict:
        """构建返回结果"""
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'price': price,
            'market_state': market_state,
            'adx': adx,
            'bb_position': bb_position,
            'allocation': allocation  # 新增：当前建议仓位比例
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


# ==================== 测试 ====================
if __name__ == '__main__':

    strategy = SimpleUpTrendStrategy(
        quick_stop_loss=0.005,      # 0.5% 快速止损
        normal_stop_loss=0.02,      # 2% 正常止损
        reduce_allocation_threshold=0.01,  # 1% 时减仓
    )

    # 模拟测试
    print("\n" + "="*50)
    print("测试动态仓位管理")
    print("="*50)

    ticker = 'TEST'

    # 模拟亏损 -> 减仓
    print("\n1. 模拟亏损触发减仓:")
    strategy._update_allocation_based_on_pnl(ticker, -0.015, 'DOWNTREND')
    print(f"   当前仓位: {strategy.get_current_allocation(ticker)*100:.0f}%")

    # 模拟盈利恢复
    print("\n2. 模拟盈利恢复仓位:")
    strategy._update_allocation_based_on_pnl(ticker, 0.01, 'UPTREND')
    print(f"   当前仓位: {strategy.get_current_allocation(ticker)*100:.0f}%")

    # 继续盈利
    print("\n3. 继续盈利:")
    strategy._update_allocation_based_on_pnl(ticker, 0.035, 'UPTREND')
    print(f"   当前仓位: {strategy.get_current_allocation(ticker)*100:.0f}%")
