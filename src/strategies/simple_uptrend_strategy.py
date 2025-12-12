# src/strategies/simple_trend_strategy.py

"""
简单趋势策略 - Simple Trend Strategy (Long Only)

核心思想：
1. 只做多（BUY/SELL），不做空
2. 检测市场状态（上升趋势 / 震荡 / 下降趋势）
3. 上升趋势 → 回调买入
4. 震荡市场 → 低买高卖
5. 下降趋势 → 不交易！

这是一个基础策略，适合新手学习和修改。
"""

from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np


class SimpleTrendStrategy:
    """
    简单趋势策略 - 只做多
    
    市场状态判断：
    - ADX > 25 且 EMA快 > EMA慢 → 上升趋势 ✅ 可交易
    - ADX > 25 且 EMA快 < EMA慢 → 下降趋势 ❌ 不交易
    - ADX < 20 → 震荡市场 ✅ 可交易
    
    交易规则：
    【上升趋势】
    - BUY: 价格回调到布林带中轨附近 (40%-60%)
    - SELL: 盈利达到目标 或 止损
    
    【震荡市场】
    - BUY: 价格接近布林带下轨 (<20%)
    - SELL: 价格回到布林带中轨以上 (>50%)
    
    【下降趋势】
    - 不开新仓
    - 如果持仓，止损出场
    """
    
    def __init__(self,
                 # 布林带参数
                 bb_period: int = 20,
                 bb_std_dev: float = 2.0,
                 
                 # ADX 参数（趋势强度）
                 adx_period: int = 14,
                 adx_trend_threshold: float = 25,  # > 25 = 趋势市
                 adx_range_threshold: float = 20,  # < 20 = 震荡市
                 
                 # EMA 参数（趋势方向）
                 ema_fast: int = 12,
                 ema_slow: int = 26,
                 
                 # 上升趋势交易参数
                 uptrend_buy_low: float = 0.40,   # 回调到 40% 以下买入
                 uptrend_buy_high: float = 0.60,  # 但不超过 60%
                 uptrend_take_profit: float = 0.03,  # 3% 止盈
                 
                 # 震荡市交易参数
                 range_buy_threshold: float = 0.20,   # 低于 20% 买入
                 range_sell_threshold: float = 0.55,  # 高于 55% 卖出
                 
                 # 止损参数
                 stop_loss_pct: float = 0.02,  # 2% 止损
                 
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
        self.stop_loss_pct = stop_loss_pct
        self.max_history_bars = max_history_bars
        
        # 数据存储
        self._history_data: Dict[str, pd.DataFrame] = {}
        
        # 打印配置
        print(f"\n{'='*50}")
        print(f"📈 简单趋势策略 (只做多)")
        print(f"{'='*50}")
        print(f"趋势判断:")
        print(f"  ADX > {adx_trend_threshold} = 趋势市")
        print(f"  ADX < {adx_range_threshold} = 震荡市")
        print(f"  EMA{ema_fast} vs EMA{ema_slow} 判断方向")
        print(f"\n交易参数:")
        print(f"  上升趋势买入: BB {uptrend_buy_low*100:.0f}%-{uptrend_buy_high*100:.0f}%")
        print(f"  上升趋势止盈: {uptrend_take_profit*100:.1f}%")
        print(f"  震荡买入: BB < {range_buy_threshold*100:.0f}%")
        print(f"  震荡卖出: BB > {range_sell_threshold*100:.0f}%")
        print(f"  止损: {stop_loss_pct*100:.1f}%")
        print(f"{'='*50}\n")
    
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """计算 EMA"""
        return series.ewm(span=period, adjust=False).mean()
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """
        计算 ADX（趋势强度指标）
        返回 numpy array 以避免索引问题
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        n = len(df)
        
        # +DM 和 -DM
        up_move = np.zeros(n)
        down_move = np.zeros(n)
        up_move[1:] = high[1:] - high[:-1]
        down_move[1:] = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # True Range
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
        
        # 平滑计算
        atr = pd.Series(tr).rolling(window=period, min_periods=1).mean().values
        plus_dm_smooth = pd.Series(plus_dm).rolling(window=period, min_periods=1).mean().values
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=period, min_periods=1).mean().values
        
        # +DI 和 -DI
        atr_safe = np.where(atr == 0, 1, atr)
        plus_di = 100 * plus_dm_smooth / atr_safe
        minus_di = 100 * minus_dm_smooth / atr_safe
        
        # DX 和 ADX
        di_sum = plus_di + minus_di
        di_sum_safe = np.where(di_sum == 0, 1, di_sum)
        dx = 100 * np.abs(plus_di - minus_di) / di_sum_safe
        adx = pd.Series(dx).rolling(window=period, min_periods=1).mean().values
        
        return adx
    
    def _get_market_state(self, adx: float, ema_fast: float, ema_slow: float) -> str:
        """
        判断市场状态
        
        Returns:
            'UPTREND' - 上升趋势（可交易）
            'DOWNTREND' - 下降趋势（不交易）
            'RANGING' - 震荡（可交易）
            'UNCLEAR' - 不明朗
        """
        if adx >= self.adx_trend_threshold:
            # 强趋势
            if ema_fast > ema_slow:
                return 'UPTREND'
            else:
                return 'DOWNTREND'
        elif adx <= self.adx_range_threshold:
            return 'RANGING'
        else:
            return 'UNCLEAR'
    
    def _calculate_bb_position(self, price: float, bb_upper: float, bb_lower: float) -> float:
        """
        计算价格在布林带中的位置 (0-1)
        0 = 下轨, 0.5 = 中轨, 1 = 上轨
        """
        bb_range = bb_upper - bb_lower
        if bb_range <= 0:
            return 0.5
        return (price - bb_lower) / bb_range
    
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
        
        Args:
            ticker: 股票代码
            new_data: 新的 K 线数据
            current_position: 当前持仓（正数=多仓）
            avg_cost: 持仓均价
            verbose: 是否打印详细信息
            is_market_close: 是否收盘强制平仓
            
        Returns:
            (signal_dict, dataframe)
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
        current_bb_middle = bb_middle.iloc[-1]
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
        market_state = self._get_market_state(current_adx, current_ema_fast, current_ema_slow)
        
        # BB 位置
        bb_position = self._calculate_bb_position(current_price, current_bb_upper, current_bb_lower)
        
        # ========== 3. 生成信号 ==========
        signal = 'HOLD'
        confidence = 5
        reason = ""
        
        # --- 收盘强制平仓 ---
        if is_market_close and current_position > 0:
            signal = 'SELL'
            confidence = 10
            reason = "⏰ 收盘平仓"
            
            return self._make_result(signal, confidence, reason, current_price, 
                                    market_state, current_adx, bb_position), df
        
        # --- 止损检查 ---
        if current_position > 0 and avg_cost > 0:
            pnl_pct = (current_price - avg_cost) / avg_cost
            
            if pnl_pct <= -self.stop_loss_pct:
                signal = 'SELL'
                confidence = 10
                reason = f"🛑 止损! 亏损 {pnl_pct*100:.2f}%"
                
                if verbose:
                    print(f"🛑 [止损] {ticker}: {reason}")
                
                return self._make_result(signal, confidence, reason, current_price,
                                        market_state, current_adx, bb_position), df
        
        # --- 根据市场状态交易 ---
        
        if market_state == 'UPTREND':
            # 🟢 上升趋势 - 回调买入
            signal, confidence, reason = self._uptrend_strategy(
                current_position, avg_cost, current_price, bb_position
            )
            
        elif market_state == 'RANGING':
            # 🟡 震荡市场 - 低买高卖
            signal, confidence, reason = self._ranging_strategy(
                current_position, current_price, bb_position
            )
            
        elif market_state == 'DOWNTREND':
            # 🔴 下降趋势 - 不交易，有仓位考虑离场
            if current_position > 0:
                # 如果还有盈利，可以考虑卖出
                if avg_cost > 0:
                    pnl_pct = (current_price - avg_cost) / avg_cost
                    if pnl_pct > 0:
                        signal = 'SELL'
                        confidence = 7
                        reason = f"📉 下降趋势，保住利润 (+{pnl_pct*100:.1f}%)"
                    else:
                        reason = f"📉 下降趋势，持仓观望"
                else:
                    reason = "📉 下降趋势，持仓观望"
            else:
                reason = "📉 下降趋势，不开仓"
        
        else:  # UNCLEAR
            reason = "⚪ 市场不明朗，观望"
        
        # ========== 4. 输出调试信息 ==========
        if verbose:
            state_emoji = {'UPTREND': '🟢', 'DOWNTREND': '🔴', 'RANGING': '🟡', 'UNCLEAR': '⚪'}
            signal_emoji = {'BUY': '💰', 'SELL': '💸', 'HOLD': '⏸️'}
            
            pos_str = f"持仓 {int(current_position)} 股" if current_position > 0 else "空仓"
            
            print(f"\n{state_emoji.get(market_state, '⚪')} [{market_state}] {ticker} | {pos_str}")
            print(f"   价格: ${current_price:.2f} | BB位置: {bb_position*100:.0f}%")
            print(f"   ADX: {current_adx:.1f} | EMA快: ${current_ema_fast:.2f} > EMA慢: ${current_ema_slow:.2f}")
            print(f"   {signal_emoji.get(signal, '❓')} {signal} - {reason}")
        
        return self._make_result(signal, confidence, reason, current_price,
                                market_state, current_adx, bb_position), df
    
    def _uptrend_strategy(self, position: float, avg_cost: float, 
                          price: float, bb_pos: float) -> Tuple[str, int, str]:
        """上升趋势策略"""
        
        if position == 0:
            # 没有仓位 - 寻找买入机会
            if self.uptrend_buy_low <= bb_pos <= self.uptrend_buy_high:
                return 'BUY', 8, f"🟢 上升趋势回调买入 (BB {bb_pos*100:.0f}%)"
            elif bb_pos < self.uptrend_buy_low:
                return 'HOLD', 5, f"回调过深，等待企稳 (BB {bb_pos*100:.0f}%)"
            else:
                return 'HOLD', 5, f"等待回调 (BB {bb_pos*100:.0f}%)"
        
        else:
            # 有仓位 - 检查止盈
            if avg_cost > 0:
                pnl_pct = (price - avg_cost) / avg_cost
                if pnl_pct >= self.uptrend_take_profit:
                    return 'SELL', 8, f"🎯 止盈 +{pnl_pct*100:.1f}%"
            
            return 'HOLD', 5, "持仓中，等待止盈"
    
    def _ranging_strategy(self, position: float, price: float, 
                          bb_pos: float) -> Tuple[str, int, str]:
        """震荡市策略"""
        
        if position == 0:
            # 没有仓位 - 等待低点买入
            if bb_pos <= self.range_buy_threshold:
                return 'BUY', 7, f"🟡 震荡低点买入 (BB {bb_pos*100:.0f}%)"
            else:
                return 'HOLD', 5, f"等待低点 (BB {bb_pos*100:.0f}%)"
        
        else:
            # 有仓位 - 等待高点卖出
            if bb_pos >= self.range_sell_threshold:
                return 'SELL', 7, f"🟡 震荡高点卖出 (BB {bb_pos*100:.0f}%)"
            else:
                return 'HOLD', 5, f"持仓等待高点 (BB {bb_pos*100:.0f}%)"
    
    def _make_result(self, signal: str, confidence: int, reason: str,
                     price: float, market_state: str, adx: float, 
                     bb_position: float) -> Dict:
        """构建返回结果"""
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'price': price,
            'market_state': market_state,
            'adx': adx,
            'bb_position': bb_position
        }
    
    def get_history_data(self, ticker: str) -> pd.DataFrame:
        """
        获取带指标的历史数据（用于图表显示）
        """
        if ticker not in self._history_data or self._history_data[ticker].empty:
            return pd.DataFrame()
        
        df = self._history_data[ticker].copy()
        
        # 去重
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='last')]
        
        close = df['close']
        
        # 布林带
        bb_middle = close.rolling(window=self.bb_period, min_periods=1).mean()
        bb_std = close.rolling(window=self.bb_period, min_periods=1).std()
        
        df['SMA'] = bb_middle.values
        df['BB_UPPER'] = (bb_middle + self.bb_std_dev * bb_std).values
        df['BB_LOWER'] = (bb_middle - self.bb_std_dev * bb_std).values
        
        # EMA
        df['EMA_FAST'] = self._calculate_ema(close, self.ema_fast).values
        df['EMA_SLOW'] = self._calculate_ema(close, self.ema_slow).values
        
        # ADX
        df['ADX'] = self._calculate_adx(df, self.adx_period)
        
        # 填充 NaN
        for col in ['SMA', 'BB_UPPER', 'BB_LOWER', 'EMA_FAST', 'EMA_SLOW']:
            df[col] = df[col].bfill()
        df['ADX'] = df['ADX'].fillna(0)
        
        return df


# ==================== 测试 ====================
if __name__ == '__main__':
    import numpy as np
    
    # 创建策略
    strategy = SimpleTrendStrategy(
        stop_loss_pct=0.02,
        uptrend_take_profit=0.03
    )
    
    # 模拟数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    df = pd.DataFrame({
        'open': prices - 0.1,
        'high': prices + 0.3,
        'low': prices - 0.3,
        'close': prices,
        'volume': np.random.randint(1000, 5000, 100)
    }, index=dates)
    
    # 测试信号
    print("\n测试信号生成:")
    result, _ = strategy.get_signal(
        ticker='TEST',
        new_data=df,
        current_position=0,
        avg_cost=0,
        verbose=True
    )
    
    print(f"\n结果: {result}")