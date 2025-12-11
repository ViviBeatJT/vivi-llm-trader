# src/engine/live_engine.py

import time
import signal
import sys
import threading
from datetime import datetime, timezone, timedelta, time as dt_time
from typing import Optional, Callable, Tuple
import pandas as pd
import pytz

from src.cache.trading_cache import TradingCache
from src.manager.position_manager import PositionManager
from src.data_fetcher.alpaca_data_fetcher import AlpacaDataFetcher
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from src.strategies.base_strategy import BaseStrategy

# ✨ Import time utilities
from src.utils.market_time_utils import (
    US_EASTERN,
    MARKET_OPEN_TIME,
    MARKET_CLOSE_TIME,
    DEFAULT_FORCE_CLOSE_TIME,
    get_current_et_time,
    is_market_open,
    is_force_close_time,
    should_force_close_position,
    get_close_signal_for_position,
    format_time_et
)


class LiveEngine:
    """
    实盘/模拟盘交易引擎 - 实时运行策略并执行交易。
    
    特点：
    1. 实时获取市场数据
    2. 按设定间隔运行策略
    3. 自动检测美股交易时间
    4. 支持优雅退出 (Ctrl+C)
    5. 可选的交易时间限制
    6. ✨ 收盘强制平仓管理（15:55自动平仓）
    
    与 BacktestEngine 的区别：
    - BacktestEngine: 回放历史数据，快速模拟
    - LiveEngine: 实时运行，等待真实时间流逝
    """

    def __init__(self,
                 ticker: str,
                 strategy: BaseStrategy,
                 position_manager: PositionManager,
                 data_fetcher: AlpacaDataFetcher,
                 cache: Optional[TradingCache] = None,
                 interval_seconds: int = 300,
                 lookback_minutes: int = 120,
                 timeframe: Optional[TimeFrame] = None,
                 respect_market_hours: bool = True,
                 max_runtime_minutes: Optional[int] = None,
                 on_signal_callback: Optional[Callable] = None,
                 force_close_time: dt_time = DEFAULT_FORCE_CLOSE_TIME):  # ✨ 新增参数
        """
        初始化实盘交易引擎。

        Args:
            ticker: 股票代码
            strategy: 策略实例
            position_manager: 仓位管理器
            data_fetcher: 数据获取器
            cache: 缓存对象（可选）
            interval_seconds: 策略运行间隔（秒），默认 300 秒 (5分钟)
            lookback_minutes: 每次获取数据的回溯时间（分钟）
            timeframe: K线时间框架（默认为5分钟）
            respect_market_hours: 是否只在美股交易时间内运行
            max_runtime_minutes: 最大运行时间（分钟），None 表示无限制
            on_signal_callback: 信号回调函数，签名: (signal_dict, price, timestamp) -> None
            force_close_time: 强制平仓时间（默认15:55），设为None禁用强制平仓
        """
        self.ticker = ticker
        self.strategy = strategy
        self.position_manager = position_manager
        self.data_fetcher = data_fetcher
        self.cache = cache
        self.interval_seconds = interval_seconds
        self.lookback_minutes = lookback_minutes
        self.timeframe = timeframe or TimeFrame(5, TimeFrameUnit.Minute)
        self.respect_market_hours = respect_market_hours
        self.max_runtime_minutes = max_runtime_minutes
        self.on_signal_callback = on_signal_callback
        self.force_close_time = force_close_time  # ✨ 新增
        
        # 运行状态
        self._running = False
        self._start_time: Optional[datetime] = None
        self._iteration_count = 0
        self._signal_count = 0
        self._force_close_executed = False  # ✨ 新增：防止重复强制平仓
        
        # 用于中断 sleep 的事件
        self._stop_event = threading.Event()
    
    def _get_current_time_et(self) -> datetime:
        """获取当前 Eastern Time。"""
        return get_current_et_time()
    
    def _is_market_open(self) -> bool:
        """检查当前是否在美股交易时间内。"""
        return is_market_open()
    
    def _get_time_until_market_open(self) -> timedelta:
        """计算距离下次开盘的时间。"""
        now_et = self._get_current_time_et()
        
        # 计算今天的开盘时间
        today_open = now_et.replace(
            hour=MARKET_OPEN_TIME.hour,
            minute=MARKET_OPEN_TIME.minute,
            second=0,
            microsecond=0
        )
        
        # 如果今天已过开盘时间或是周末，计算下一个交易日
        if now_et.time() > MARKET_CLOSE_TIME or now_et.weekday() >= 5:
            # 找到下一个工作日
            days_ahead = 1
            next_day = now_et + timedelta(days=days_ahead)
            while next_day.weekday() >= 5:
                days_ahead += 1
                next_day = now_et + timedelta(days=days_ahead)
            
            today_open = next_day.replace(
                hour=MARKET_OPEN_TIME.hour,
                minute=MARKET_OPEN_TIME.minute,
                second=0,
                microsecond=0
            )
        elif now_et.time() < MARKET_OPEN_TIME:
            pass  # 使用今天的开盘时间
        
        return today_open - now_et
    
    def _fetch_data(self) -> Tuple[pd.DataFrame, float]:
        """
        获取最新市场数据和当前价格。
        
        Returns:
            Tuple[pd.DataFrame, float]: (OHLCV 数据, 最新价格)
        """
        now_utc = datetime.now(timezone.utc)
        
        df = self.data_fetcher.get_latest_bars(
            ticker=self.ticker,
            lookback_minutes=self.lookback_minutes,
            end_dt=now_utc,
            timeframe=self.timeframe
        )
        
        if not df.empty:
            current_price = df.iloc[-1]['close']
        else:
            current_price = 0.0
        
        return df, current_price
    
    def _format_duration(self, seconds: float) -> str:
        """格式化时长显示。"""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def _interruptible_sleep(self, seconds: float) -> bool:
        """
        可中断的睡眠。
        
        Args:
            seconds: 睡眠时间（秒）
            
        Returns:
            bool: True 如果正常完成，False 如果被中断
        """
        # 使用 Event.wait() 代替 time.sleep()，这样可以被中断
        interrupted = self._stop_event.wait(timeout=seconds)
        return not interrupted
    
    def _log_status(self, current_price: float):
        """打印当前状态。"""
        now_et = self._get_current_time_et()
        account_status = self.position_manager.get_account_status(current_price)
        
        print(f"\n{'='*60}")
        print(f"📊 [{now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}] 状态更新")
        print(f"   {self.ticker} 价格: ${current_price:.2f}")
        print(f"   账户权益: ${account_status['equity']:,.2f}")
        print(f"   现金: ${account_status['cash']:,.2f}")
        print(f"   持仓: {account_status['position']:.0f} 股")
        if account_status['position'] != 0:
            print(f"   持仓均价: ${account_status['avg_cost']:.2f}")
            if account_status['position'] > 0:
                unrealized_pnl = (current_price - account_status['avg_cost']) * account_status['position']
            else:
                unrealized_pnl = (account_status['avg_cost'] - current_price) * abs(account_status['position'])
            print(f"   未实现盈亏: ${unrealized_pnl:,.2f}")
        print(f"   运行迭代: {self._iteration_count} 次")
        print(f"   交易信号: {self._signal_count} 次")
        print(f"{'='*60}")
    
    def _execute_force_close(self, current_price: float, now_et: datetime, now_utc: datetime) -> bool:
        """
        执行强制平仓
        
        Args:
            current_price: 当前价格
            now_et: 当前东部时间
            now_utc: 当前UTC时间
            
        Returns:
            bool: 是否成功执行强制平仓
        """
        account_status = self.position_manager.get_account_status(current_price)
        current_position = account_status.get('position', 0.0)
        
        if current_position == 0:
            print(f"   ✅ 当前无持仓，无需强制平仓")
            return True
        
        close_signal = get_close_signal_for_position(current_position)
        
        print(f"\n🔔 [{format_time_et(now_et)}] 执行强制平仓！")
        print(f"   持仓: {current_position:.0f} 股")
        print(f"   价格: ${current_price:.2f}")
        print(f"   信号: {close_signal}")
        
        try:
            # 构造强制平仓信号
            force_close_signal = {
                'signal': close_signal,
                'confidence_score': 10,
                'reason': f'收盘强制平仓 ({format_time_et(now_et)})'
            }
            
            # 调用回调
            if self.on_signal_callback:
                try:
                    self.on_signal_callback(force_close_signal, current_price, now_utc)
                except Exception as e:
                    print(f"⚠️ 信号回调错误: {e}")
            
            # 执行交易
            trade_result = self.position_manager.execute_and_update(
                timestamp=now_utc,
                signal=close_signal,
                current_price=current_price,
                ticker=self.ticker
            )
            
            if trade_result:
                print(f"   ✅ 强制平仓成功")
                self._signal_count += 1
                return True
            else:
                print(f"   ❌ 强制平仓失败")
                return False
                
        except Exception as e:
            print(f"   ❌ 强制平仓错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_single_iteration(self) -> bool:
        """
        运行单次策略迭代。
        
        Returns:
            bool: 是否成功执行
        """
        now_utc = datetime.now(timezone.utc)
        now_et = self._get_current_time_et()
        
        self._iteration_count += 1
        
        # ✨ 优先检查：是否需要强制平仓
        if (self.force_close_time is not None and 
            not self._force_close_executed and
            is_force_close_time(now_et, self.force_close_time)):
            
            # 获取当前价格
            try:
                _, current_price = self._fetch_data()
            except Exception as e:
                print(f"⚠️ 获取价格失败: {e}")
                current_price = 0.0
            
            if current_price > 0:
                self._execute_force_close(current_price, now_et, now_utc)
                self._force_close_executed = True
                
                # 强制平仓后继续正常流程，但策略应该不会再产生新交易信号
        
        # ✨ 检查是否到达市场收盘时间
        if now_et.time() >= MARKET_CLOSE_TIME:
            print(f"\n🔴 [{format_time_et(now_et)}] 市场已收盘，停止运行")
            return False
        
        # 1. 获取数据
        market_data, current_price = self._fetch_data()
        
        if market_data.empty or current_price <= 0:
            print(f"⚠️ [{format_time_et(now_et)}] 无市场数据，跳过本次迭代")
            return True  # 继续运行，但跳过本次
        
        # 2. 获取当前持仓状态
        account_status = self.position_manager.get_account_status(current_price)
        current_position = account_status.get('position', 0.0)
        avg_cost = account_status.get('avg_cost', 0.0)
        
        # 3. 调用策略
        try:
            # ✨ 传递收盘时间信息给策略
            is_close_to_market_close = is_force_close_time(now_et, self.force_close_time) if self.force_close_time else False
            
            signal_data, strategy_price = self.strategy.get_signal(
                ticker=self.ticker,
                new_data=market_data,
                current_position=current_position,
                avg_cost=avg_cost,
                verbose=True,
                is_market_close=is_close_to_market_close,  # ✨ 传递强制平仓标志
                current_time_et=now_et  # ✨ 传递当前东部时间
            )
            
            signal = signal_data.get('signal', 'HOLD')
            confidence = signal_data.get('confidence_score', 0)
            reason = signal_data.get('reason', '')
            
            if strategy_price > 0:
                current_price = strategy_price
                
        except Exception as e:
            print(f"❌ [{format_time_et(now_et)}] 策略错误: {e}")
            import traceback
            traceback.print_exc()
            return True  # 继续运行
        
        # 4. 执行信号回调（如果有）
        if self.on_signal_callback:
            try:
                self.on_signal_callback(signal_data, current_price, now_utc)
            except Exception as e:
                print(f"⚠️ 信号回调错误: {e}")
        
        # 5. 执行交易
        if signal in ["BUY", "SELL", "SHORT", "COVER"]:
            self._signal_count += 1
            signal_emoji = {"BUY": "🟢", "SELL": "🔴", "SHORT": "🔻", "COVER": "🔺"}.get(signal, "⚪")
            print(f"\n{signal_emoji} [{format_time_et(now_et)}] 交易信号!")
            print(f"   信号: {signal} | 价格: ${current_price:.2f} | 置信度: {confidence}/10")
            print(f"   原因: {reason}")
            
            trade_result = self.position_manager.execute_and_update(
                timestamp=now_utc,
                signal=signal,
                current_price=current_price,
                ticker=self.ticker
            )
            
            if trade_result:
                print(f"   ✅ 交易执行成功")
            else:
                print(f"   ❌ 交易执行失败")
        
        # 6. 打印状态
        self._log_status(current_price)
        
        return True
    
    def run(self) -> dict:
        """
        启动实盘交易引擎。
        
        Returns:
            dict: 运行统计信息
        """
        self._running = True
        self._stop_event.clear()
        self._start_time = datetime.now(timezone.utc)
        self._iteration_count = 0
        self._signal_count = 0
        self._force_close_executed = False  # ✨ 重置强制平仓标志
        
        now_et = self._get_current_time_et()
        
        print("\n" + "="*60)
        print("🚀 实盘交易引擎启动")
        print("="*60)
        print(f"   股票代码: {self.ticker}")
        print(f"   策略: {self.strategy.__class__.__name__}")
        print(f"   运行间隔: {self.interval_seconds} 秒")
        print(f"   K线周期: {self.timeframe.amount} {self.timeframe.unit.name}")
        print(f"   遵守交易时间: {'是' if self.respect_market_hours else '否'}")
        if self.max_runtime_minutes:
            print(f"   最大运行时间: {self.max_runtime_minutes} 分钟")
        # ✨ 显示强制平仓时间
        if self.force_close_time:
            print(f"   强制平仓时间: {self.force_close_time.strftime('%H:%M')} ET")
        print(f"   启动时间: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"   按 Ctrl+C 停止运行")
        print("="*60)
        
        # 检查初始账户状态
        initial_status = self.position_manager.get_account_status(current_price=0.0)
        print(f"\n💰 初始账户状态:")
        print(f"   现金: ${initial_status['cash']:,.2f}")
        print(f"   持仓: {initial_status['position']:.0f} 股")
        
        try:
            while self._running:
                # 检查最大运行时间
                if self.max_runtime_minutes:
                    elapsed = (datetime.now(timezone.utc) - self._start_time).total_seconds() / 60
                    if elapsed >= self.max_runtime_minutes:
                        print(f"\n⏰ 达到最大运行时间 ({self.max_runtime_minutes} 分钟)，停止运行")
                        break
                
                # 检查交易时间
                if self.respect_market_hours and not self._is_market_open():
                    time_until_open = self._get_time_until_market_open()
                    now_et = self._get_current_time_et()
                    
                    print(f"\n😴 [{now_et.strftime('%H:%M:%S %Z')}] 当前非交易时间")
                    print(f"   距离开盘: {self._format_duration(time_until_open.total_seconds())}")
                    
                    # 如果距离开盘超过1小时，每小时检查一次；否则每分钟检查
                    if time_until_open.total_seconds() > 3600:
                        sleep_time = 3600  # 1小时
                    else:
                        sleep_time = 60  # 1分钟
                    
                    print(f"   {self._format_duration(sleep_time)} 后再次检查...")
                    
                    if not self._interruptible_sleep(sleep_time):
                        break  # 被中断
                    continue
                
                # 运行策略迭代
                continue_running = self._run_single_iteration()
                
                if not continue_running:
                    break  # 停止运行（例如市场收盘）
                
                # 等待下一次迭代
                if self._running:
                    print(f"\n⏳ 等待 {self.interval_seconds} 秒后进行下一次检查...")
                    
                    if not self._interruptible_sleep(self.interval_seconds):
                        break  # 被中断
                        
        except KeyboardInterrupt:
            print(f"\n\n⚠️ 收到 Ctrl+C，正在停止...")
        except Exception as e:
            print(f"\n❌ 运行时错误: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._running = False
            
            # ✨ 最终持仓检查
            self._final_position_check()
        
        # 生成运行报告
        return self._generate_report()
    
    def _final_position_check(self):
        """
        最终持仓检查 - 确保没有遗留持仓
        """
        print(f"\n{'='*60}")
        print("🔍 最终持仓检查")
        print("="*60)
        
        try:
            # 获取最终价格和持仓
            _, current_price = self._fetch_data()
            account_status = self.position_manager.get_account_status(current_price)
            final_position = account_status.get('position', 0.0)
            
            print(f"   最终持仓: {final_position:.0f} 股")
            
            if final_position != 0:
                print(f"\n⚠️  检测到未平仓位！")
                print(f"   执行最终强制平仓...")
                
                now_utc = datetime.now(timezone.utc)
                now_et = self._get_current_time_et()
                
                success = self._execute_force_close(current_price, now_et, now_utc)
                
                if success:
                    print(f"   ✅ 最终强制平仓完成")
                else:
                    print(f"   ❌ 最终强制平仓失败，请手动检查持仓！")
            else:
                print(f"   ✅ 持仓已归零")
                
        except Exception as e:
            print(f"⚠️ 最终检查失败: {e}")
            print(f"   请手动检查持仓状态！")
    
    def _generate_report(self) -> dict:
        """生成运行报告。"""
        end_time = datetime.now(timezone.utc)
        runtime_seconds = (end_time - self._start_time).total_seconds() if self._start_time else 0
        
        # 获取最终价格（尝试获取）
        try:
            _, final_price = self._fetch_data()
        except:
            final_price = 0.0
        
        final_status = self.position_manager.get_account_status(current_price=final_price)
        trade_log = self.position_manager.get_trade_log()
        
        report = {
            'ticker': self.ticker,
            'start_time': self._start_time,
            'end_time': end_time,
            'runtime_seconds': runtime_seconds,
            'iterations': self._iteration_count,
            'signals': self._signal_count,
            'trades_executed': len(trade_log) if trade_log is not None and not trade_log.empty else 0,
            'final_equity': final_status['equity'],
            'final_cash': final_status['cash'],
            'final_position': final_status['position'],
            'final_price': final_price,
            'force_close_executed': self._force_close_executed,  # ✨ 新增
        }
        
        # 打印报告
        print("\n" + "="*60)
        print("📋 运行报告")
        print("="*60)
        print(f"   运行时长: {self._format_duration(runtime_seconds)}")
        print(f"   迭代次数: {self._iteration_count}")
        print(f"   交易信号: {self._signal_count}")
        print(f"   执行交易: {report['trades_executed']}")
        print(f"   强制平仓: {'是' if self._force_close_executed else '否'}")  # ✨ 新增
        print(f"   最终价格: ${final_price:.2f}")
        print(f"   最终权益: ${final_status['equity']:,.2f}")
        print(f"   最终现金: ${final_status['cash']:,.2f}")
        print(f"   最终持仓: {final_status['position']:.0f} 股 {'✅' if final_status['position'] == 0 else '⚠️'}")  # ✨ 改进
        print("="*60)
        
        # 保存缓存
        if self.cache and len(self.cache.data) > 0:
            print(f"\n💾 保存缓存...")
            self.cache.save()
        
        return report
    
    def stop(self):
        """手动停止引擎。"""
        print("\n⏹️ 停止引擎...")
        self._running = False
        self._stop_event.set()  # 触发事件，中断 sleep
    
    @property
    def is_running(self) -> bool:
        """检查引擎是否在运行。"""
        return self._running


# ==================== 测试用例 ====================
if __name__ == '__main__':
    print("LiveEngine 模块 - 请通过 live_runner.py 运行")