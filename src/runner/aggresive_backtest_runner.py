# src/runner/aggressive_backtest_runner.py

"""
激进均值回归策略 - 回测运行器

特点：
- 使用历史数据快速回测
- 验证止损机制效果
- 评估高频交易策略表现
"""

from datetime import datetime, timezone, time as dt_time
import os
from dotenv import load_dotenv
import pytz

# --- Core Modules ---
from src.cache.trading_cache import TradingCache
from src.manager.position_manager import PositionManager
from src.data_fetcher.alpaca_data_fetcher import AlpacaDataFetcher
from src.engine.backtest_engine import BacktestEngine
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# --- Executors ---
from src.executor.simulation_executor import SimulationExecutor

# --- 激进策略 ---
from src.strategies.aggressive_mean_reversion_strategy import AggressiveMeanReversionStrategy

load_dotenv()

# ==========================================
# US Market Hours Validation
# ==========================================

US_EASTERN = pytz.timezone('America/New_York')
MARKET_OPEN_TIME = dt_time(9, 30)
MARKET_CLOSE_TIME = dt_time(16, 0)


def validate_market_hours(dt: datetime, label: str = "Time") -> datetime:
    """验证时间是否在美股开盘时间内"""
    if dt.tzinfo is None:
        dt = US_EASTERN.localize(dt)
    
    dt_eastern = dt.astimezone(US_EASTERN)
    market_time = dt_eastern.time()
    weekday = dt_eastern.weekday()
    
    if weekday >= 5:
        day_name = "Saturday" if weekday == 5 else "Sunday"
        raise ValueError(
            f"❌ {label} {dt_eastern.strftime('%Y-%m-%d %H:%M %Z')} 是 {day_name}，美股休市。\n"
            f"   请选择周一至周五的交易日。"
        )
    
    if market_time < MARKET_OPEN_TIME or market_time > MARKET_CLOSE_TIME:
        raise ValueError(
            f"❌ {label} {dt_eastern.strftime('%Y-%m-%d %H:%M %Z')} 不在美股交易时间内。\n"
            f"   美股交易时间: {MARKET_OPEN_TIME.strftime('%H:%M')} - {MARKET_CLOSE_TIME.strftime('%H:%M')} ET\n"
            f"   请调整时间到交易时段内。"
        )
    
    return dt.astimezone(timezone.utc)


def print_market_hours_info(start_dt: datetime, end_dt: datetime):
    """打印市场时间信息"""
    start_et = start_dt.astimezone(US_EASTERN)
    end_et = end_dt.astimezone(US_EASTERN)
    
    print(f"⏰ 回测时间范围:")
    print(f"   开始: {start_et.strftime('%Y-%m-%d %H:%M %Z')} ({start_dt.strftime('%H:%M UTC')})")
    print(f"   结束: {end_et.strftime('%Y-%m-%d %H:%M %Z')} ({end_dt.strftime('%H:%M UTC')})")
    print(f"   美股交易时间: {MARKET_OPEN_TIME.strftime('%H:%M')} - {MARKET_CLOSE_TIME.strftime('%H:%M')} ET")


# ==========================================
# 1. 策略参数配置
# ==========================================

# 💹 激进策略参数
BB_PERIOD = 20                      # 布林带周期
BB_STD_DEV = 2.0                    # 标准差倍数
STOP_LOSS_THRESHOLD = 0.10          # 止损阈值（10%）
MONITOR_INTERVAL_SECONDS = 60       # 监控间隔（用于标记，实际由 STEP_MINUTES 控制）

# 💰 财务参数
FINANCE_PARAMS = {
    'INITIAL_CAPITAL': 100000.0,
    'COMMISSION_RATE': 0.0003,
    'SLIPPAGE_RATE': 0.0001,
    'MIN_LOT_SIZE': 10,
    'MAX_ALLOCATION': 0.2,
    'STAMP_DUTY_RATE': 0.001,
}

# 🎯 回测设置
TICKER = "TSLA"

# 时间设置 (使用 Eastern Time 更直观)
# 📅 回测一整个交易日
START_TIME = US_EASTERN.localize(datetime(2025, 12, 8, 9, 30))   # 9:30 AM ET
END_TIME = US_EASTERN.localize(datetime(2025, 12, 8, 16, 0))     # 4:00 PM ET

# ⏱️ 回测步进设置
STEP_MINUTES = 1            # 每1分钟检查一次（模拟高频监控）
LOOKBACK_MINUTES = 120      # 数据回溯时间（需要足够计算布林带）

# 📊 K线周期
DATA_TIMEFRAME = TimeFrame(5, TimeFrameUnit.Minute)  # 5分钟K线用于计算指标


# ==========================================
# 2. 验证时间
# ==========================================

print(f"\n🚀 初始化激进策略回测 - {TICKER}")
print("="*60)

try:
    START_TIME = validate_market_hours(START_TIME, "Start time")
    END_TIME = validate_market_hours(END_TIME, "End time")
    
    if END_TIME <= START_TIME:
        raise ValueError("❌ End time must be after start time.")
    
    print_market_hours_info(START_TIME, END_TIME)
    
except ValueError as e:
    print(str(e))
    exit(1)


# ==========================================
# 3. 初始化组件
# ==========================================

# A. Data Fetcher
data_fetcher = AlpacaDataFetcher()

# B. Cache System
cache_path = os.path.join('cache', f'{TICKER}_aggressive_backtest_cache.json')
cache = TradingCache(cache_path)

# C. Executor & Position Manager
print("\n🔧 初始化模拟执行器...")
executor = SimulationExecutor(FINANCE_PARAMS)
position_manager = PositionManager(executor, FINANCE_PARAMS)

# D. 创建激进策略
print(f"\n💹 初始化激进均值回归策略...")
strategy = AggressiveMeanReversionStrategy(
    bb_period=BB_PERIOD,
    bb_std_dev=BB_STD_DEV,
    max_history_bars=500,
    stop_loss_threshold=STOP_LOSS_THRESHOLD,
    monitor_interval_seconds=MONITOR_INTERVAL_SECONDS
)

print(f"\n🎯 策略规则:")
print("="*60)
print("   📈 价格突破上轨 → 做空（SHORT）")
print("   📉 空仓时价格回到中线 → 平空（COVER）")
print("   📉 价格跌破下轨 → 做多（BUY）")
print("   📈 多仓时价格回到中线 → 平多（SELL）")
print(f"   ⚠️ 单笔持仓亏损 {STOP_LOSS_THRESHOLD*100:.0f}% → 强制止损")
print("="*60)


# ==========================================
# 4. 创建并运行回测引擎
# ==========================================

# 🔧 创建改进的回测引擎（需要传递持仓信息给策略）
class AggressiveBacktestEngine(BacktestEngine):
    """
    扩展的回测引擎，支持向策略传递持仓信息。
    
    这对于激进策略的止损机制至关重要。
    """
    
    def _run_single_iteration(self, current_time: datetime) -> bool:
        """运行单次策略迭代（重写以传递持仓信息）"""
        # 1. 获取数据
        market_data, current_price = self._fetch_data(current_time)
        
        if market_data.empty or current_price <= 0:
            if hasattr(current_time, 'strftime'):
                time_str = current_time.strftime('%m-%d %H:%M')
            else:
                time_str = str(current_time)
            print(f"⚠️ {time_str}: 无市场数据，跳过")
            return False
        
        # 2. 获取当前持仓状态
        account_status = self.position_manager.get_account_status(current_price)
        current_position = account_status['position']
        avg_cost = account_status['avg_cost']
        
        # 3. 调用策略获取信号（传入持仓信息）
        try:
            signal_data, strategy_price = self.strategy.get_signal(
                ticker=self.ticker,
                new_data=market_data,
                current_position=current_position,
                avg_cost=avg_cost,
                verbose=False
            )
            
            signal = signal_data.get('signal', 'HOLD')
            confidence = signal_data.get('confidence_score', 0)
            reason = signal_data.get('reason', '')
            
            # 优先使用策略返回的价格
            if strategy_price > 0:
                current_price = strategy_price
                
        except Exception as e:
            print(f"❌ 策略错误 @ {current_time}: {e}")
            return False
        
        # 4. 执行交易
        if signal in ["BUY", "SELL", "SHORT", "COVER"]:
            signal_emoji = {
                "BUY": "🟢", 
                "SELL": "🔴", 
                "SHORT": "🔻", 
                "COVER": "🔺"
            }.get(signal, "⚪")
            
            time_str = current_time.strftime('%m-%d %H:%M') if hasattr(current_time, 'strftime') else str(current_time)
            
            # 显示持仓状态
            if current_position > 0:
                pos_str = f"多仓{current_position:.0f}股@${avg_cost:.2f}"
            elif current_position < 0:
                pos_str = f"空仓{abs(current_position):.0f}股@${avg_cost:.2f}"
            else:
                pos_str = "无仓位"
            
            print(f"{signal_emoji} {time_str} | {signal} | ${current_price:.2f} | {pos_str}")
            print(f"   原因: {reason} (置信度: {confidence})")
            
            trade_result = self.position_manager.execute_and_update(
                timestamp=current_time,
                signal=signal,
                current_price=current_price,
                ticker=self.ticker
            )
            
            return trade_result
        
        return True


# 创建引擎
backtest_engine = AggressiveBacktestEngine(
    ticker=TICKER,
    start_dt=START_TIME,
    end_dt=END_TIME,
    strategy=strategy,
    position_manager=position_manager,
    data_fetcher=data_fetcher,
    cache=cache,
    step_minutes=STEP_MINUTES,
    lookback_minutes=LOOKBACK_MINUTES,
    timeframe=DATA_TIMEFRAME
)

# 运行回测
print(f"\n🏃 开始回测...")
print(f"   步进间隔: {STEP_MINUTES} 分钟（模拟 {MONITOR_INTERVAL_SECONDS}秒监控）")
print(f"   预计迭代次数: {(END_TIME - START_TIME).seconds // (STEP_MINUTES * 60)}")
print("="*60 + "\n")

initial_cache_size = len(cache.data)
final_equity, trade_log = backtest_engine.run()


# ==========================================
# 5. 结果分析
# ==========================================

# 保存缓存
if len(cache.data) > initial_cache_size:
    print(f"\n💾 保存 {len(cache.data) - initial_cache_size} 个新缓存条目...")
    cache.save()

# 计算收益
net_pnl = final_equity - FINANCE_PARAMS['INITIAL_CAPITAL']
return_pct = (net_pnl / FINANCE_PARAMS['INITIAL_CAPITAL']) * 100

print("\n" + "="*60)
print(f"💰 回测结果汇总 ({TICKER})")
print("="*60)
print(f"   策略:          {strategy}")
print(f"   初始资金:      ${FINANCE_PARAMS['INITIAL_CAPITAL']:,.2f}")
print(f"   最终权益:      ${final_equity:,.2f}")
print(f"   净盈亏:        ${net_pnl:,.2f} ({return_pct:+.2f}%)")
print(f"   布林带参数:    周期={BB_PERIOD}, 标准差={BB_STD_DEV}σ")
print(f"   止损阈值:      {STOP_LOSS_THRESHOLD*100:.1f}%")
print("="*60)

# 打印策略累积的历史数据信息
print(f"\n📊 策略累积了 {strategy.get_history_size(TICKER)} 条K线数据")

# 详细交易日志分析
if trade_log is not None and not trade_log.empty:
    print("\n📝 交易日志:")
    print("="*60)
    
    # 显示所有交易
    display_log = trade_log[['time', 'type', 'qty', 'price', 'fee', 'net_pnl']].copy()
    display_log['time'] = display_log['time'].dt.strftime('%Y-%m-%d %H:%M')
    print(display_log.to_markdown(index=False, floatfmt=".2f"))
    
    # 统计分析
    print("\n" + "="*60)
    print("📊 交易统计分析")
    print("="*60)
    
    # 基础统计
    total_trades = len(trade_log)
    buy_trades = len(trade_log[trade_log['type'] == 'BUY'])
    sell_trades = len(trade_log[trade_log['type'] == 'SELL'])
    short_trades = len(trade_log[trade_log['type'] == 'SHORT'])
    cover_trades = len(trade_log[trade_log['type'] == 'COVER'])
    
    print(f"   总交易次数:    {total_trades}")
    print(f"   做多交易:      BUY={buy_trades}, SELL={sell_trades}")
    print(f"   做空交易:      SHORT={short_trades}, COVER={cover_trades}")
    
    # 盈亏分析
    profitable_trades = trade_log[trade_log['net_pnl'] > 0]
    losing_trades = trade_log[trade_log['net_pnl'] < 0]
    
    if len(profitable_trades) > 0:
        avg_profit = profitable_trades['net_pnl'].mean()
        max_profit = profitable_trades['net_pnl'].max()
        print(f"\n   盈利交易:      {len(profitable_trades)} 笔")
        print(f"   平均盈利:      ${avg_profit:.2f}")
        print(f"   最大盈利:      ${max_profit:.2f}")
    
    if len(losing_trades) > 0:
        avg_loss = losing_trades['net_pnl'].mean()
        max_loss = losing_trades['net_pnl'].min()
        print(f"\n   亏损交易:      {len(losing_trades)} 笔")
        print(f"   平均亏损:      ${avg_loss:.2f}")
        print(f"   最大亏损:      ${max_loss:.2f}")
        
        # 检查是否有止损触发
        stop_loss_trades = losing_trades[
            losing_trades['net_pnl'] / (losing_trades['qty'] * losing_trades['price']) 
            <= -STOP_LOSS_THRESHOLD
        ]
        if len(stop_loss_trades) > 0:
            print(f"\n   ⚠️ 止损触发:   {len(stop_loss_trades)} 次")
    
    # 胜率
    completed_pairs = (sell_trades + cover_trades)
    if completed_pairs > 0:
        win_rate = len(profitable_trades) / completed_pairs * 100
        print(f"\n   完成交易对:    {completed_pairs} 对")
        print(f"   胜率:          {win_rate:.1f}%")
    
    # 总费用
    total_fees = trade_log['fee'].sum()
    print(f"\n   总手续费:      ${total_fees:.2f}")
    
    # 净盈亏（已包含手续费）
    total_pnl = trade_log['net_pnl'].sum()
    print(f"   净盈亏(含费):  ${total_pnl:.2f}")
    
    print("="*60)
    
else:
    print("\n🤷 无交易记录。")
    print("   可能原因:")
    print("   - 回测时间段太短")
    print("   - 价格未触发交易信号")
    print("   - 数据不足以计算布林带")

print("\n" + "="*60)
print("✅ 回测完成")
print("="*60)