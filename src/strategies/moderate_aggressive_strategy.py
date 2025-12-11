# src/strategies/moderate_aggressive_strategy.py (IMPROVED VERSION)

"""
温和进取策略 - Moderate Aggressive Mean Reversion
改进版 - 强化收盘管理

核心改进：
1. 接近布林带边界就开仓（不必完全突破）
2. 回调到 60% 位置就平仓（不必回到中线）
3. 可调节的灵敏度参数
4. ✨ 强化的收盘时间管理：
   - 15:50后禁止开新仓（BUY/SHORT）
   - 15:55后强制平仓（SELL/COVER）
   - 多重安全检查确保不留隔夜仓

适合：18:20 这种接近但未突破的情况
"""

from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np


class ModerateAggressiveStrategy:
    """
    温和进取型均值回归策略（改进版）
    
    交易规则：
    - 价格 > 布林带宽度 85% → SHORT（例：接近上轨）
    - 空仓价格回落到 60% → COVER
    - 价格 < 布林带宽度 15% → BUY（例：接近下轨）
    - 多仓价格上涨到 40% → SELL
    
    收盘管理：
    - 15:50后：禁止开新仓（BUY/SHORT），只允许平仓（SELL/COVER）
    - 15:55后：强制平仓所有持仓
    - 16:00前：确保持仓为0
    
    示例（以你的 18:20 数据）：
    - BB Upper: $373.38, Middle: $370.89, Lower: $368.41
    - 布林带宽度: $4.97
    - 85% 线: $371.81（超过此价格就做空）
    - 15% 线: $369.16（低于此价格就做多）← 18:20 的 $369.04 会触发！
    """
    
    def __init__(self,
                 bb_period: int = 20,
                 bb_std_dev: float = 2.0,
                 entry_threshold: float = 0.85,    # 开仓阈值（0.85 = 接近 85%）
                 exit_threshold: float = 0.60,     # 平仓阈值（0.60 = 回到 60%）
                 stop_loss_threshold: float = 0.10,
                 monitor_interval_seconds: int = 60,
                 max_history_bars: int = 500,
                 # ✨ 新增：收盘时间控制
                 no_new_entry_time: int = 15 * 60 + 50,  # 15:50 (minutes from midnight)
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
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.monitor_interval_seconds = monitor_interval_seconds
        self.max_history_bars = max_history_bars
        
        # ✨ 收盘时间控制
        self.no_new_entry_time = no_new_entry_time
        self.force_close_time = force_close_time
        
        self._history_data: Dict[str, pd.DataFrame] = {}
        
        print(f"📊 温和进取策略初始化 (改进版):")
        print(f"   开仓阈值: {entry_threshold*100:.0f}%")
        print(f"   平仓阈值: {exit_threshold*100:.0f}%")
        print(f"   止损阈值: {stop_loss_threshold*100:.0f}%")
        print(f"   🔔 收盘管理:")
        print(f"      禁止新开仓: {no_new_entry_time//60:02d}:{no_new_entry_time%60:02d}")
        print(f"      强制平仓: {force_close_time//60:02d}:{force_close_time%60:02d}")
    
    # ==================== 数据管理 ====================
    
    def _merge_data(self, ticker: str, new_df: pd.DataFrame) -> pd.DataFrame:
        """合并新数据与历史数据"""
        if new_df.empty:
            return self._history_data.get(ticker, pd.DataFrame())
        
        if ticker not in self._history_data or self._history_data[ticker].empty:
            merged_df = new_df.copy()
        else:
            history_df = self._history_data[ticker]
            merged_df = pd.concat([history_df, new_df])
            merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
            merged_df = merged_df.sort_index()
        
        if len(merged_df) > self.max_history_bars:
            merged_df = merged_df.iloc[-self.max_history_bars:]
        
        return merged_df
    
    def get_history_data(self, ticker: str) -> pd.DataFrame:
        """获取历史数据"""
        if ticker in self._history_data:
            return self._history_data[ticker].copy()
        return pd.DataFrame()
    
    # ==================== 技术指标 ====================
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算布林带"""
        df = df.copy()
        df['SMA'] = df['close'].rolling(window=self.bb_period).mean()
        df['STD'] = df['close'].rolling(window=self.bb_period).std()
        df['BB_UPPER'] = df['SMA'] + (df['STD'] * self.bb_std_dev)
        df['BB_LOWER'] = df['SMA'] - (df['STD'] * self.bb_std_dev)
        
        # 🔥 新增：计算布林带内的位置（0-1）
        df['BB_WIDTH'] = df['BB_UPPER'] - df['BB_LOWER']
        df['BB_POSITION'] = (df['close'] - df['BB_LOWER']) / df['BB_WIDTH']
        # BB_POSITION = 0 → 在下轨
        # BB_POSITION = 0.5 → 在中线
        # BB_POSITION = 1 → 在上轨
        
        return df
    
    # ==================== 时间检查 ====================
    
    def _get_time_minutes(self, time_et: pd.Timestamp) -> int:
        """将东部时间转换为从午夜开始的分钟数"""
        if time_et is None:
            return 0
        return time_et.hour * 60 + time_et.minute
    
    def _is_in_no_new_entry_window(self, current_time_et: pd.Timestamp) -> bool:
        """检查是否在禁止开新仓窗口（15:50-16:00）"""
        if current_time_et is None:
            return False
        time_minutes = self._get_time_minutes(current_time_et)
        return time_minutes >= self.no_new_entry_time
    
    def _is_force_close_time(self, current_time_et: pd.Timestamp) -> bool:
        """检查是否到达强制平仓时间（15:55+）"""
        if current_time_et is None:
            return False
        time_minutes = self._get_time_minutes(current_time_et)
        return time_minutes >= self.force_close_time
    
    # ==================== 信号生成 ====================
    
    def _generate_signal(self,
                        price: float,
                        bb_upper: float,
                        bb_lower: float,
                        sma: float,
                        bb_position: float,
                        current_position: float = 0.0,
                        avg_cost: float = 0.0,
                        current_time_et: pd.Timestamp = None) -> Tuple[str, int, str]:
        """
        根据布林带位置生成信号（改进版）
        
        Args:
            bb_position: 价格在布林带中的位置（0-1）
                - 0 = 在下轨
                - 0.5 = 在中线
                - 1 = 在上轨
            current_time_et: 当前东部时间
        """
        if pd.isna([price, bb_upper, bb_lower, sma, bb_position]).any():
            return "HOLD", 0, "数据不足"
        
        # ===== 🔴 优先级1：止损检查（最高优先级）=====
        if current_position != 0 and avg_cost > 0:
            if current_position > 0:  # 多仓
                loss_pct = (avg_cost - price) / avg_cost
                if loss_pct >= self.stop_loss_threshold:
                    return "SELL", 10, f"⚠️ 止损！多仓亏损 {loss_pct*100:.2f}%"
            elif current_position < 0:  # 空仓
                loss_pct = (price - avg_cost) / avg_cost
                if loss_pct >= self.stop_loss_threshold:
                    return "COVER", 10, f"⚠️ 止损！空仓亏损 {loss_pct*100:.2f}%"
        
        # ===== 🔴 优先级2：时间窗口检查 =====
        in_no_entry_window = self._is_in_no_new_entry_window(current_time_et)
        
        # 如果在禁止开仓窗口且无持仓 → HOLD
        if in_no_entry_window and current_position == 0:
            time_str = f"{current_time_et.hour:02d}:{current_time_et.minute:02d}" if current_time_et else "N/A"
            return "HOLD", 0, f"⏰ {time_str} 禁止新开仓（15:50后）"
        
        # ===== 根据布林带位置交易 =====
        
        # 🔥 接近上轨 → 做空
        if bb_position > self.entry_threshold:
            if current_position <= 0:  # 空仓或无仓位
                # ✨ 检查时间窗口
                if in_no_entry_window:
                    return "HOLD", 0, f"⏰ 15:50后禁止新开空仓"
                
                return "SHORT", 8, (f"价格接近上轨！位置 {bb_position*100:.1f}% "
                                   f"(${price:.2f} vs 阈值 {self.entry_threshold*100:.0f}%)")
        
        # 🔥 空仓回调 → 平空
        if current_position < 0:
            if bb_position < self.exit_threshold:
                return "COVER", 7, (f"空仓获利平仓！位置回到 {bb_position*100:.1f}% "
                                   f"(目标 {self.exit_threshold*100:.0f}%)")
        
        # 🔥 接近下轨 → 做多
        if bb_position < (1 - self.entry_threshold):
            if current_position >= 0:  # 多仓或无仓位
                # ✨ 检查时间窗口
                if in_no_entry_window:
                    return "HOLD", 0, f"⏰ 15:50后禁止新开多仓"
                
                return "BUY", 8, (f"价格接近下轨！位置 {bb_position*100:.1f}% "
                                 f"(${price:.2f} vs 阈值 {(1-self.entry_threshold)*100:.0f}%)")
        
        # 🔥 多仓回调 → 平多
        if current_position > 0:
            if bb_position > (1 - self.exit_threshold):
                return "SELL", 7, (f"多仓获利平仓！位置回到 {bb_position*100:.1f}% "
                                  f"(目标 {(1-self.exit_threshold)*100:.0f}%)")
        
        # 持有
        return "HOLD", 3, f"价格在区间内 {bb_position*100:.1f}%"
    
    # ==================== 主接口 ====================
    
    def get_signal(self,
                   ticker: str,
                   new_data: pd.DataFrame,
                   current_position: float = 0.0,
                   avg_cost: float = 0.0,
                   verbose: bool = True,
                   is_market_close: bool = False,
                   current_time_et: pd.Timestamp = None) -> Tuple[Dict, float]:
        """
        获取交易信号（改进版）
        
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
        # ===== 🔴 最高优先级：强制平仓检查 =====
        
        # 检查1：is_market_close 标志（15:55+）
        if is_market_close and current_position != 0:
            close_signal = 'SELL' if current_position > 0 else 'COVER'
            time_str = f"{current_time_et.hour:02d}:{current_time_et.minute:02d}" if current_time_et else "15:55"
            reason = f"🔔 {time_str} 强制平仓！持仓: {current_position:.0f} 股"
            
            if verbose:
                print(f"⚠️ 收盘平仓: {close_signal} | {reason}")
            
            return {
                "signal": close_signal,
                "confidence_score": 10,
                "reason": reason
            }, 0.0
        
        # 检查2：时间判断（15:55+），双重保险
        if current_time_et is not None and current_position != 0:
            if self._is_force_close_time(current_time_et):
                close_signal = 'SELL' if current_position > 0 else 'COVER'
                time_str = f"{current_time_et.hour:02d}:{current_time_et.minute:02d}"
                reason = f"🔔 {time_str} 强制平仓（时间到）！持仓: {current_position:.0f} 股"
                
                if verbose:
                    print(f"⚠️ 收盘平仓: {close_signal} | {reason}")
                
                return {
                    "signal": close_signal,
                    "confidence_score": 10,
                    "reason": reason
                }, 0.0
        
        # ===== 正常交易逻辑 =====
        
        # 1. 合并数据
        df = self._merge_data(ticker, new_data)
        
        if verbose:
            pos_str = f"多{current_position:.0f}股" if current_position > 0 else \
                     f"空{abs(current_position):.0f}股" if current_position < 0 else "无仓"
            time_str = f"{current_time_et.hour:02d}:{current_time_et.minute:02d}" if current_time_et else "N/A"
            print(f"📊 [{time_str}] {ticker}: {len(df)} 条K线 | {pos_str}")
        
        if df.empty:
            return {"signal": "HOLD", "confidence_score": 0, "reason": "No data"}, 0.0
        
        # 2. 计算指标
        df = self._calculate_bollinger_bands(df)
        
        # 3. 更新历史数据（包含指标）
        self._history_data[ticker] = df.copy()
        
        # 4. 获取有效数据
        df_valid = df.dropna()
        
        if df_valid.empty or len(df_valid) < self.bb_period:
            if verbose:
                print(f"❌ 数据不足（需要 {self.bb_period} 条）")
            return {"signal": "HOLD", "confidence_score": 0, "reason": "数据不足"}, 0.0
        
        # 5. 获取最新数据
        latest = df_valid.iloc[-1]
        price = latest['close']
        
        # 6. 生成信号（传入时间）
        signal, confidence, reason = self._generate_signal(
            price,
            latest['BB_UPPER'],
            latest['BB_LOWER'],
            latest['SMA'],
            latest['BB_POSITION'],
            current_position,
            avg_cost,
            current_time_et  # ✨ 传入时间
        )
        
        # ===== 🔴 最终过滤：15:50后禁止BUY/SHORT =====
        if current_time_et is not None:
            if self._is_in_no_new_entry_window(current_time_et):
                if signal in ['BUY', 'SHORT']:
                    time_str = f"{current_time_et.hour:02d}:{current_time_et.minute:02d}"
                    if verbose:
                        print(f"⚠️ [{time_str}] 过滤信号 {signal} → HOLD（15:50后禁止新开仓）")
                    signal = "HOLD"
                    confidence = 0
                    reason = f"⏰ {time_str} 过滤{signal}信号（15:50后禁止新开仓）"
        
        # 7. 打印信息
        if verbose:
            print(f"   价格: ${price:.2f} | BB位置: {latest['BB_POSITION']*100:.1f}% | "
                  f"范围: [${latest['BB_LOWER']:.2f}, ${latest['SMA']:.2f}, ${latest['BB_UPPER']:.2f}]")
            
            if current_position != 0 and avg_cost > 0:
                pnl_pct = ((price - avg_cost) / avg_cost if current_position > 0 
                          else (avg_cost - price) / avg_cost) * 100
                print(f"   {'📈' if pnl_pct > 0 else '📉'} 持仓盈亏: {pnl_pct:+.2f}%")
            
            emoji = {"BUY": "🟢", "SELL": "🔴", "SHORT": "🔻", "COVER": "🔺", "HOLD": "⚪"}
            print(f"   {emoji.get(signal, '⚪')} {signal} ({confidence}/10) - {reason}")
        
        return {
            "signal": signal,
            "confidence_score": confidence,
            "reason": reason
        }, price