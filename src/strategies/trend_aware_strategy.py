# src/strategies/trend_aware_strategy.py

"""
趋势感知策略 - Trend-Aware Adaptive Strategy

核心思想：
1. 检测市场状态（趋势 vs 震荡）
2. 趋势市场 → 趋势跟踪
3. 震荡市场 → 均值回归
4. 避免逆势交易

特点：
- 使用 ADX 检测趋势强度
- 使用 EMA 判断趋势方向
- 动态切换交易策略
- 保护性止损
"""

from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np


class TrendAwareStrategy:
    """
    趋势感知自适应策略
    
    技术指标：
    1. ADX (Average Directional Index) - 趋势强度
       - ADX > 25: 强趋势
       - ADX < 20: 震荡市
    
    2. EMA (Exponential Moving Average) - 趋势方向
       - 短期 EMA > 长期 EMA: 上升趋势
       - 短期 EMA < 长期 EMA: 下降趋势
    
    3. Bollinger Bands - 超买超卖
    
    交易规则：
    【上升趋势模式】(ADX > 25 && EMA快 > EMA慢)
    - ✅ BUY: 价格回调到BB中轨或下轨附近
    - ❌ 不做空！
    - SELL: 止盈/止损
    
    【下降趋势模式】(ADX > 25 && EMA快 < EMA慢)
    - ✅ SHORT: 价格反弹到BB中轨或上轨附近
    - ❌ 不做多！
    - COVER: 止盈/止损
    
    【震荡模式】(ADX < 20)
    - BUY: 价格跌破下轨
    - SELL: 价格回到中轨以上
    - SHORT: 价格突破上轨
    - COVER: 价格回到中轨以下
    """
    
    def __init__(self,
                 # Bollinger Bands 参数
                 bb_period: int = 20,
                 bb_std_dev: float = 2.0,
                 
                 # 趋势检测参数
                 adx_period: int = 14,           # ADX 周期
                 adx_trend_threshold: float = 20, # ADX > 20 = 强趋势（降低阈值更敏感）
                 adx_range_threshold: float = 15, # ADX < 15 = 震荡
                 
                 # EMA 参数（趋势方向）
                 ema_fast_period: int = 12,      # 快速 EMA
                 ema_slow_period: int = 26,      # 慢速 EMA
                 
                 # 均值回归参数（震荡市）
                 mean_reversion_entry: float = 0.85,  # 接近85%开仓
                 mean_reversion_exit: float = 0.60,   # 回到60%平仓
                 
                 # 趋势跟踪参数（趋势市）
                 trend_entry_pullback: float = 0.50,  # 回调到50%开仓
                 trend_exit_profit: float = 0.03,     # 3%止盈
                 
                 # 波动率过滤（防止布林带收窄时交易）
                 min_bb_width_pct: float = 0.02,      # 最小布林带宽度 2%（相对价格）
                 
                 # 信号冷却期（防止频繁切换）
                 cooldown_minutes: int = 15,          # 平仓后等待15分钟再开新仓（延长）
                 
                 # 风险管理
                 stop_loss_threshold: float = 0.02,  # 2%止损（更合理的阈值）
                 monitor_interval_seconds: int = 60,
                 max_history_bars: int = 500):
        
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold
        
        self.ema_fast_period = ema_fast_period
        self.ema_slow_period = ema_slow_period
        
        self.mean_reversion_entry = mean_reversion_entry
        self.mean_reversion_exit = mean_reversion_exit
        
        self.trend_entry_pullback = trend_entry_pullback
        self.trend_exit_profit = trend_exit_profit
        
        self.min_bb_width_pct = min_bb_width_pct
        self.cooldown_minutes = cooldown_minutes
        self.stop_loss_threshold = stop_loss_threshold
        self.monitor_interval_seconds = monitor_interval_seconds
        self.max_history_bars = max_history_bars
        
        self._history_data: Dict[str, pd.DataFrame] = {}
        self._last_exit_time: Dict[str, Optional[pd.Timestamp]] = {}  # 记录上次平仓时间
        
        print(f"📊 趋势感知策略初始化:")
        print(f"   ADX 趋势阈值: {adx_trend_threshold}（> 此值 = 趋势市）")
        print(f"   ADX 震荡阈值: {adx_range_threshold}（< 此值 = 震荡市）")
        print(f"   快速 EMA: {ema_fast_period} / 慢速 EMA: {ema_slow_period}")
        print(f"   震荡市策略: 均值回归（{mean_reversion_entry*100:.0f}% 开仓）")
        print(f"   趋势市策略: 趋势跟踪（{trend_entry_pullback*100:.0f}% 回调）")
        print(f"   止损阈值: {stop_loss_threshold*100:.1f}% (给策略更多空间)")
        print(f"   🔒 最小BB宽度: {min_bb_width_pct*100:.1f}% (避免低波动陷阱)")
        print(f"   ⏰ 冷却期: {cooldown_minutes} 分钟（平仓后等待）")
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        计算 ADX (Average Directional Index)
        
        ADX 衡量趋势强度（不管方向）
        - ADX > 25: 强趋势
        - ADX 20-25: 趋势形成中
        - ADX < 20: 弱趋势/震荡
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # 计算 +DM 和 -DM
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # 计算 ATR (Average True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # 计算 +DI 和 -DI
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
        
        # 计算 DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # 计算 ADX (DX 的移动平均)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """计算指数移动平均"""
        return series.ewm(span=period, adjust=False).mean()
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """计算所有技术指标"""
        
        # 1. Bollinger Bands - 使用相同的参数
        bb_middle = df['close'].rolling(window=self.bb_period, min_periods=self.bb_period).mean()
        bb_std = df['close'].rolling(window=self.bb_period, min_periods=self.bb_period).std()
        bb_upper = bb_middle + (self.bb_std_dev * bb_std)
        bb_lower = bb_middle - (self.bb_std_dev * bb_std)
        
        # 2. ADX (趋势强度)
        adx = self._calculate_adx(df, self.adx_period)
        
        # 3. EMA (趋势方向)
        ema_fast = self._calculate_ema(df['close'], self.ema_fast_period)
        ema_slow = self._calculate_ema(df['close'], self.ema_slow_period)
        
        # 当前值 - 使用 iloc[-1] 并检查 NaN
        current_price = df['close'].iloc[-1]
        
        # 检查布林带值是否有效
        current_bb_upper = bb_upper.iloc[-1] if not pd.isna(bb_upper.iloc[-1]) else current_price * 1.02
        current_bb_middle = bb_middle.iloc[-1] if not pd.isna(bb_middle.iloc[-1]) else current_price
        current_bb_lower = bb_lower.iloc[-1] if not pd.isna(bb_lower.iloc[-1]) else current_price * 0.98
        
        current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
        current_ema_fast = ema_fast.iloc[-1] if not pd.isna(ema_fast.iloc[-1]) else current_price
        current_ema_slow = ema_slow.iloc[-1] if not pd.isna(ema_slow.iloc[-1]) else current_price
        
        # 计算价格在布林带中的位置 (0-1)
        bb_range = current_bb_upper - current_bb_lower
        if bb_range > 0:
            bb_position = (current_price - current_bb_lower) / bb_range
        else:
            bb_position = 0.5
        
        # 判断市场状态
        if current_adx >= self.adx_trend_threshold:
            if current_ema_fast > current_ema_slow:
                market_state = 'UPTREND'
            else:
                market_state = 'DOWNTREND'
        elif current_adx <= self.adx_range_threshold:
            market_state = 'RANGING'
        else:
            market_state = 'UNCLEAR'
        
        return {
            'price': current_price,
            'bb_upper': current_bb_upper,
            'bb_middle': current_bb_middle,
            'bb_lower': current_bb_lower,
            'bb_position': bb_position,
            'bb_range': bb_range,
            'adx': current_adx,
            'ema_fast': current_ema_fast,
            'ema_slow': current_ema_slow,
            'market_state': market_state,
            'df_with_indicators': df  # 保留原始数据
        }
    
    def get_signal(self,
                   ticker: str,
                   new_data: pd.DataFrame,
                   current_position: float = 0.0,
                   avg_cost: float = 0.0,
                   verbose: bool = False,
                   is_market_close: bool = False,
                   current_time_et = None,
                   initial_capital: float = 1000.0,  # 新增：初始资金
                   current_equity: float = None) -> Tuple[Dict, pd.DataFrame]:  # 新增：当前权益
        """
        获取交易信号
        
        Args:
            initial_capital: 初始资金（用于计算总体亏损）
            current_equity: 当前账户权益（用于额外止损检查）
        
        Returns:
            (signal_data, updated_df)
        """
        
        # 更新历史数据
        if ticker not in self._history_data or self._history_data[ticker].empty:
            self._history_data[ticker] = new_data.copy()
        else:
            self._history_data[ticker] = pd.concat([
                self._history_data[ticker],
                new_data
            ]).drop_duplicates().tail(self.max_history_bars)
        
        df = self._history_data[ticker]
        
        # 计算指标
        indicators = self._calculate_indicators(df)
        
        price = indicators['price']
        bb_upper = indicators['bb_upper']
        bb_middle = indicators['bb_middle']
        bb_lower = indicators['bb_lower']
        bb_position = indicators['bb_position']
        adx = indicators['adx']
        ema_fast = indicators['ema_fast']
        ema_slow = indicators['ema_slow']
        market_state = indicators['market_state']
        
        # 默认信号
        signal = 'HOLD'
        confidence = 5
        reason = ""
        
        # === 市场收盘强制平仓 ===
        if is_market_close:
            if current_position > 0:
                signal = 'SELL'
                confidence = 10
                reason = "⏰ 市场收盘，强制平多仓"
            elif current_position < 0:
                signal = 'COVER'
                confidence = 10
                reason = "⏰ 市场收盘，强制平空仓"
            
            if verbose:
                print(f"⏰ [市场收盘] {ticker}: {reason}")
            
            return {
                'signal': signal,
                'confidence': confidence,
                'reason': reason,
                'price': price,
                'market_state': market_state,
                'adx': adx
            }, df
        
        # === 基于总权益的止损检查（优先级最高）===
        if current_equity is not None and initial_capital > 0 and current_position != 0:
            total_loss_pct = (current_equity - initial_capital) / initial_capital
            
            if verbose:
                print(f"   💰 账户检查: 初始=${initial_capital:.2f}, 当前=${current_equity:.2f}, 总亏损={total_loss_pct*100:.2f}%")
            
            # 如果总亏损超过阈值，立即平仓
            if total_loss_pct <= -self.stop_loss_threshold:
                if current_position > 0:
                    signal = 'SELL'
                    reason = f"🛑 账户止损！总亏损 {total_loss_pct*100:.2f}%（超过 {self.stop_loss_threshold*100:.1f}%）"
                elif current_position < 0:
                    signal = 'COVER'
                    reason = f"🛑 账户止损！总亏损 {total_loss_pct*100:.2f}%（超过 {self.stop_loss_threshold*100:.1f}%）"
                
                confidence = 10
                
                if verbose:
                    print(f"🛑 [账户止损] {ticker}: {reason}")
                    print(f"   ${initial_capital:.2f} → ${current_equity:.2f} (亏损 ${initial_capital - current_equity:.2f})")
                
                # 记录止损平仓时间
                current_time = df.index[-1] if len(df) > 0 else None
                self._last_exit_time[ticker] = current_time
                if verbose:
                    print(f"   ⏰ 账户止损触发，开始 {self.cooldown_minutes} 分钟冷却期")
                
                return {
                    'signal': signal,
                    'confidence': confidence,
                    'reason': reason,
                    'price': price,
                    'market_state': market_state,
                    'adx': adx
                }, df
        
        # === 基于持仓成本的止损检查 ===
        if current_position != 0 and avg_cost > 0:
            if verbose:
                print(f"   💰 持仓检查: 持仓={current_position}, 成本=${avg_cost:.2f}, 当前=${price:.2f}")
            
            if current_position > 0:
                # 多仓止损
                pnl_pct = (price - avg_cost) / avg_cost
                if verbose:
                    print(f"   📊 多仓盈亏: {pnl_pct*100:.2f}% (阈值: {-self.stop_loss_threshold*100:.1f}%)")
                
                if pnl_pct <= -self.stop_loss_threshold:
                    signal = 'SELL'
                    confidence = 10
                    reason = f"🛑 止损！多仓亏损 {pnl_pct*100:.2f}%（超过 {self.stop_loss_threshold*100:.1f}% 阈值）"
                    
                    if verbose:
                        print(f"🛑 [紧急止损] {ticker}: {reason}")
                        print(f"   买入成本: ${avg_cost:.2f} | 当前价格: ${price:.2f} | 亏损: ${price - avg_cost:.2f}")
                    
                    # 记录止损平仓时间（开启冷却期）
                    current_time = df.index[-1] if len(df) > 0 else None
                    self._last_exit_time[ticker] = current_time
                    if verbose:
                        print(f"   ⏰ 止损触发，记录平仓时间，开始 {self.cooldown_minutes} 分钟冷却期")
                    
                    return {
                        'signal': signal,
                        'confidence': confidence,
                        'reason': reason,
                        'price': price,
                        'market_state': market_state,
                        'adx': adx
                    }, df
                    
            elif current_position < 0:
                # 空仓止损
                pnl_pct = (avg_cost - price) / avg_cost
                if verbose:
                    print(f"   📊 空仓盈亏: {pnl_pct*100:.2f}% (阈值: {-self.stop_loss_threshold*100:.1f}%)")
                
                if pnl_pct <= -self.stop_loss_threshold:
                    signal = 'COVER'
                    confidence = 10
                    reason = f"🛑 止损！空仓亏损 {pnl_pct*100:.2f}%（超过 {self.stop_loss_threshold*100:.1f}% 阈值）"
                    
                    if verbose:
                        print(f"🛑 [紧急止损] {ticker}: {reason}")
                        print(f"   做空成本: ${avg_cost:.2f} | 当前价格: ${price:.2f} | 亏损: ${avg_cost - price:.2f}")
                    
                    # 记录止损平仓时间（开启冷却期）
                    current_time = df.index[-1] if len(df) > 0 else None
                    self._last_exit_time[ticker] = current_time
                    if verbose:
                        print(f"   ⏰ 止损触发，记录平仓时间，开始 {self.cooldown_minutes} 分钟冷却期")
                    
                    return {
                        'signal': signal,
                        'confidence': confidence,
                        'reason': reason,
                        'price': price,
                        'market_state': market_state,
                        'adx': adx
                    }, df
        
        # === 布林带宽度检查（防止在低波动期交易）===
        bb_width_pct = bb_range / price if price > 0 else 0
        
        if verbose:
            print(f"   📏 布林带宽度: {bb_width_pct*100:.2f}% (最小要求: {self.min_bb_width_pct*100:.1f}%)")
        
        # 只有在准备开新仓时才检查布林带宽度
        if current_position == 0 and bb_width_pct < self.min_bb_width_pct:
            signal = 'HOLD'
            confidence = 5
            reason = f"🔒 布林带收窄 ({bb_width_pct*100:.2f}% < {self.min_bb_width_pct*100:.1f}%)，观望"
            
            if verbose:
                print(f"   🔒 [低波动保护] {ticker}: {reason}")
                print(f"      当前BB宽度: ${bb_range:.2f} ({bb_width_pct*100:.2f}%)")
                print(f"      等待波动率恢复后再交易")
            
            return {
                'signal': signal,
                'confidence': confidence,
                'reason': reason,
                'price': price,
                'bb_position': bb_position,
                'market_state': market_state,
                'adx': adx,
                'ema_fast': ema_fast,
                'ema_slow': ema_slow
            }, df
        
        # === 根据市场状态选择策略 ===
        
        if market_state == 'UPTREND':
            # 🔵 上升趋势模式 - 只做多
            signal, confidence, reason = self._uptrend_logic(
                price, bb_position, bb_lower, bb_middle, 
                current_position, avg_cost
            )
        
        elif market_state == 'DOWNTREND':
            # 🔴 下降趋势模式 - 只做空
            signal, confidence, reason = self._downtrend_logic(
                price, bb_position, bb_upper, bb_middle,
                current_position, avg_cost
            )
        
        elif market_state == 'RANGING':
            # 🟡 震荡模式 - 均值回归
            signal, confidence, reason = self._ranging_logic(
                price, bb_position, bb_upper, bb_middle, bb_lower,
                current_position, avg_cost
            )
        
        else:
            # ⚪ 不明朗 - 保守观望
            signal = 'HOLD'
            confidence = 5
            reason = "市场状态不明朗，观望"
        
        # 输出调试信息
        if verbose:
            state_emoji = {
                'UPTREND': '🔵',
                'DOWNTREND': '🔴',
                'RANGING': '🟡',
                'UNCLEAR': '⚪'
            }
            
            pos_str = f"多{int(abs(current_position))}股" if current_position > 0 else \
                      f"空{int(abs(current_position))}股" if current_position < 0 else "无仓"
            
            print(f"\n{state_emoji.get(market_state, '⚪')} [{market_state}] {ticker}: {len(df)} 条K线 | {pos_str}")
            print(f"   价格: ${price:.2f} | BB位置: {bb_position*100:.1f}%")
            print(f"   ADX: {adx:.1f} | EMA快: ${ema_fast:.2f} | EMA慢: ${ema_slow:.2f}")
            print(f"   BB范围: [${bb_lower:.2f}, ${bb_middle:.2f}, ${bb_upper:.2f}]")
            
            signal_emoji = {
                'BUY': '🟢',
                'SELL': '🔴',
                'SHORT': '🟠',
                'COVER': '🟣',
                'HOLD': '⚪'
            }
            print(f"   {signal_emoji.get(signal, '⚪')} {signal} ({confidence}/10) - {reason}")
        
        # === 冷却期检查（在返回信号前）===
        current_time = df.index[-1] if len(df) > 0 else None
        
        # 情况1: 如果当前要开新仓（BUY或SHORT），检查是否在冷却期
        if signal in ['BUY', 'SHORT'] and current_position == 0:
            if ticker in self._last_exit_time and self._last_exit_time[ticker] is not None:
                time_since_exit = (current_time - self._last_exit_time[ticker]).total_seconds() / 60
                
                if time_since_exit < self.cooldown_minutes:
                    # 还在冷却期内，拒绝开新仓
                    if verbose:
                        print(f"   ⏰ [冷却期阻止] 距离上次平仓仅 {time_since_exit:.1f} 分钟")
                        print(f"      需要等待 {self.cooldown_minutes - time_since_exit:.1f} 分钟")
                    
                    # 改为 HOLD
                    signal = 'HOLD'
                    confidence = 5
                    reason = f"冷却期中（还需 {self.cooldown_minutes - time_since_exit:.1f} 分钟）"
        
        # 情况2: 如果是平仓信号，记录平仓时间（用于下次冷却期判断）
        if signal in ['SELL', 'COVER'] and current_position != 0:
            self._last_exit_time[ticker] = current_time
            if verbose:
                print(f"   ⏰ 记录平仓时间: {current_time}，开始 {self.cooldown_minutes} 分钟冷却期")
        
        # 情况3: 如果开仓成功，清除冷却时间
        if signal in ['BUY', 'SHORT'] and current_position == 0:
            # 只有通过冷却期检查才能走到这里
            if ticker in self._last_exit_time:
                self._last_exit_time[ticker] = None
                if verbose:
                    print(f"   ✅ 开仓成功，清除冷却期记录")
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'price': price,
            'bb_position': bb_position,
            'market_state': market_state,
            'adx': adx,
            'ema_fast': ema_fast,
            'ema_slow': ema_slow
        }, df
    
    def _uptrend_logic(self, price, bb_position, bb_lower, bb_middle, 
                       current_position, avg_cost):
        """
        上升趋势逻辑 - 只做多，不做空
        
        开仓：价格回调到50%左右（中轨附近）
        平仓：止盈3%或止损
        """
        signal = 'HOLD'
        confidence = 5
        reason = ""
        
        if current_position == 0:
            # 无仓 - 寻找回调买入机会
            # 要求：价格回调到中轨附近（40%-60%），且不能太接近下轨（避免假突破）
            if 0.40 <= bb_position <= 0.60:
                signal = 'BUY'
                confidence = 8
                reason = f"上升趋势回调买入（位置 {bb_position*100:.1f}%）"
            elif bb_position < 0.40:
                # 回调太深，可能趋势反转，观望
                reason = f"回调过深，观望（位置 {bb_position*100:.1f}%）"
        
        elif current_position > 0:
            # 持多仓 - 检查止盈
            if avg_cost > 0:
                pnl_pct = (price - avg_cost) / avg_cost
                if pnl_pct >= self.trend_exit_profit:
                    signal = 'SELL'
                    confidence = 9
                    reason = f"趋势跟踪止盈 {pnl_pct*100:.2f}%"
        
        return signal, confidence, reason
    
    def _downtrend_logic(self, price, bb_position, bb_upper, bb_middle,
                         current_position, avg_cost):
        """
        下降趋势逻辑 - 只做空，不做多
        
        开仓：价格反弹到50%左右（中轨附近）
        平仓：止盈3%或止损
        """
        signal = 'HOLD'
        confidence = 5
        reason = ""
        
        if current_position == 0:
            # 无仓 - 寻找反弹做空机会
            # 要求：价格反弹到中轨附近（40%-60%），且不能太接近上轨（避免假突破）
            if 0.40 <= bb_position <= 0.60:
                signal = 'SHORT'
                confidence = 8
                reason = f"下降趋势反弹做空（位置 {bb_position*100:.1f}%）"
            elif bb_position > 0.60:
                # 反弹太高，可能趋势反转，观望
                reason = f"反弹过高，观望（位置 {bb_position*100:.1f}%）"
        
        elif current_position < 0:
            # 持空仓 - 检查止盈
            if avg_cost > 0:
                pnl_pct = (avg_cost - price) / avg_cost
                if pnl_pct >= self.trend_exit_profit:
                    signal = 'COVER'
                    confidence = 9
                    reason = f"趋势跟踪止盈 {pnl_pct*100:.2f}%"
        
        return signal, confidence, reason
    
    def _ranging_logic(self, price, bb_position, bb_upper, bb_middle, bb_lower,
                       current_position, avg_cost):
        """
        震荡市场逻辑 - 均值回归
        
        做多：价格接近下轨 → 回到中轨平仓
        做空：价格接近上轨 → 回到中轨平仓
        """
        signal = 'HOLD'
        confidence = 5
        reason = ""
        
        if current_position == 0:
            # 无仓 - 寻找极值
            if bb_position <= (1 - self.mean_reversion_entry):
                signal = 'BUY'
                confidence = 7
                reason = f"震荡市做多（位置 {bb_position*100:.1f}%）"
            elif bb_position >= self.mean_reversion_entry:
                signal = 'SHORT'
                confidence = 7
                reason = f"震荡市做空（位置 {bb_position*100:.1f}%）"
        
        elif current_position > 0:
            # 持多仓 - 回到中轨以上平仓
            if bb_position >= self.mean_reversion_exit:
                signal = 'SELL'
                confidence = 8
                reason = f"震荡市平多（位置回到 {bb_position*100:.1f}%）"
        
        elif current_position < 0:
            # 持空仓 - 回到中轨以下平仓
            if bb_position <= (1 - self.mean_reversion_exit):
                signal = 'COVER'
                confidence = 8
                reason = f"震荡市平空（位置回到 {bb_position*100:.1f}%）"
        
        return signal, confidence, reason
    
    def get_history_data(self, ticker: str) -> pd.DataFrame:
        """
        获取历史数据（用于回测图表）
        
        Args:
            ticker: 股票代码
            
        Returns:
            包含所有技术指标的 DataFrame
        """
        if ticker not in self._history_data or self._history_data[ticker].empty:
            return pd.DataFrame()
        
        df = self._history_data[ticker].copy()
        
        # 计算所有指标（即使数据不足也要计算，只是前面会是 NaN）
        # 1. Bollinger Bands
        bb_middle = df['close'].rolling(window=self.bb_period, min_periods=1).mean()
        bb_std = df['close'].rolling(window=self.bb_period, min_periods=1).std()
        bb_upper = bb_middle + (self.bb_std_dev * bb_std)
        bb_lower = bb_middle - (self.bb_std_dev * bb_std)
        
        # 2. ADX
        adx = self._calculate_adx(df, self.adx_period)
        
        # 3. EMA
        ema_fast = self._calculate_ema(df['close'], self.ema_fast_period)
        ema_slow = self._calculate_ema(df['close'], self.ema_slow_period)
        
        # ✨ 添加到 DataFrame - 使用标准列名（大写 + 下划线）
        df['BB_UPPER'] = bb_upper  # 改为大写
        df['SMA'] = bb_middle      # 改为 SMA（标准中轨名称）
        df['BB_LOWER'] = bb_lower  # 改为大写
        df['ADX'] = adx
        df['EMA_FAST'] = ema_fast
        df['EMA_SLOW'] = ema_slow
        
        # 对于早期 NaN 值，用后续有效值填充（用于图表显示）
        df['BB_UPPER'] = df['BB_UPPER'].bfill()
        df['SMA'] = df['SMA'].bfill()
        df['BB_LOWER'] = df['BB_LOWER'].bfill()
        df['ADX'] = df['ADX'].fillna(0)
        df['EMA_FAST'] = df['EMA_FAST'].bfill()
        df['EMA_SLOW'] = df['EMA_SLOW'].bfill()
        
        # 检查数据充足性（只是警告，不影响返回）
        min_required = max(self.bb_period, self.adx_period, self.ema_slow_period)
        if len(df) < min_required:
            print(f"⚠️ 数据({len(df)}条)少于推荐值 {min_required}，前期指标可能不准确")
        
        return df