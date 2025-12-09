# src/runner/backtest_with_dashboard.py

"""
带实时仪表板的回测运行器

特点：
- 回测时实时显示图表
- 浏览器自动刷新
- 可视化策略行为
- 帮助调试和优化

使用方法：
1. 运行脚本
2. 自动打开浏览器 http://localhost:8050
3. 观看实时回测过程
"""

from datetime import datetime, timezone
import os
import webbrowser
import time
from dotenv import load_dotenv
import pytz

# --- Core Modules ---
from src.cache.trading_cache import TradingCache
from src.manager.position_manager import PositionManager
from src.data_fetcher.alpaca_data_fetcher import AlpacaDataFetcher
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# --- Dashboard ---
from src.visualization.live_trading_dashboard import TradingDashboard

# --- Executors ---
from src.executor.simulation_executor import SimulationExecutor

# --- 策略 ---
from src.strategies.aggresive_mean_reversion_strategy import AggressiveMeanReversionStrategy

load_dotenv()

US_EASTERN = pytz.timezone('America/New_York')

# ==========================================
# 配置区
# ==========================================

# 🎯 基本设置
TICKER = "TSLA"
TRADING_DATE = "2024-12-05"

# 💹 策略参数
BB_PERIOD = 20
BB_STD_DEV = 2.0
STOP_LOSS_THRESHOLD = 0.10

# ⏱️ 回测设置
STEP_MINUTES = 1        # 每1分钟检查
LOOKBACK_MINUTES = 120

# 💰 初始资金
INITIAL_CAPITAL = 100000.0

# 🌐 仪表板设置
DASHBOARD_PORT = 8050
AUTO_OPEN_BROWSER = True  # 自动打开浏览器

# ==========================================
# 初始化
# ==========================================

print("\n" + "="*60)
print(f"🚀 带实时仪表板的回测 - {TICKER}")
print("="*60)

# 解析日期
date_parts = [int(x) for x in TRADING_DATE.split('-')]
START_TIME = US_EASTERN.localize(datetime(date_parts[0], date_parts[1], date_parts[2], 9, 30))
END_TIME = US_EASTERN.localize(datetime(date_parts[0], date_parts[1], date_parts[2], 16, 0))
START_TIME = START_TIME.astimezone(timezone.utc)
END_TIME = END_TIME.astimezone(timezone.utc)

print(f"\n📅 回测时间: {TRADING_DATE} 9:30-16:00 ET")
print(f"📊 仪表板: http://localhost:{DASHBOARD_PORT}")

# 财务参数
FINANCE_PARAMS = {
    'INITIAL_CAPITAL': INITIAL_CAPITAL,
    'COMMISSION_RATE': 0.0003,
    'SLIPPAGE_RATE': 0.0001,
    'MIN_LOT_SIZE': 10,
    'MAX_ALLOCATION': 0.2,
}

# 初始化组件
data_fetcher = AlpacaDataFetcher()
cache = TradingCache(os.path.join('cache', f'{TICKER}_dashboard_cache.json'))
executor = SimulationExecutor(FINANCE_PARAMS)
position_manager = PositionManager(executor, FINANCE_PARAMS)

# 创建策略
strategy = AggressiveMeanReversionStrategy(
    bb_period=BB_PERIOD,
    bb_std_dev=BB_STD_DEV,
    stop_loss_threshold=STOP_LOSS_THRESHOLD,
    monitor_interval_seconds=60
)

# ==========================================
# 启动仪表板
# ==========================================

print(f"\n🌐 启动实时仪表板...")
dashboard = TradingDashboard(
    ticker=TICKER,
    port=DASHBOARD_PORT,
    update_interval=500  # 500ms 刷新间隔
)

dashboard.set_initial_capital(INITIAL_CAPITAL)
dashboard.start()

# 自动打开浏览器
if AUTO_OPEN_BROWSER:
    time.sleep(1)
    webbrowser.open(f'http://localhost:{DASHBOARD_PORT}')
    print(f"✅ 已打开浏览器")

time.sleep(2)  # 等待服务器完全启动

# ==========================================
# 手动回测循环（带仪表板更新）
# ==========================================

print(f"\n🏃 开始回测（观察浏览器窗口）...")
print(f"   按 Ctrl+C 可提前停止\n")

from datetime import timedelta

current_time = START_TIME
iteration = 0

try:
    while current_time <= END_TIME:
        iteration += 1
        
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        
        # 1. 获取数据
        df = data_fetcher.get_latest_bars(
            ticker=TICKER,
            lookback_minutes=LOOKBACK_MINUTES,
            end_dt=current_time,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute)
        )
        
        if df.empty:
            current_time += timedelta(minutes=STEP_MINUTES)
            continue
        
        current_price = df.iloc[-1]['close']
        
        # 2. 获取持仓状态
        account_status = position_manager.get_account_status(current_price)
        current_position = account_status.get('position', 0.0)
        avg_cost = account_status.get('avg_cost', 0.0)
        
        # 3. 调用策略
        try:
            signal_data, strategy_price = strategy.get_signal(
                ticker=TICKER,
                new_data=df,
                current_position=current_position,
                avg_cost=avg_cost,
                verbose=False
            )
            
            signal = signal_data.get('signal', 'HOLD')
            confidence = signal_data.get('confidence_score', 0)
            reason = signal_data.get('reason', '')
            
            if strategy_price > 0:
                current_price = strategy_price
        
        except Exception as e:
            print(f"❌ 策略错误: {e}")
            current_time += timedelta(minutes=STEP_MINUTES)
            continue
        
        # 4. 获取策略维护的完整数据（包含所有指标）
        strategy_df = strategy.get_history_data(TICKER)
        if not strategy_df.empty:
            # 更新仪表板的市场数据
            dashboard.update_market_data(strategy_df)
        
        # 5. 执行交易
        if signal in ["BUY", "SELL", "SHORT", "COVER"]:
            print(f"{'🟢' if signal=='BUY' else '🔴' if signal=='SELL' else '🔻' if signal=='SHORT' else '🔺'} "
                  f"{current_time.strftime('%H:%M')} | {signal} | ${current_price:.2f}")
            
            trade_result = position_manager.execute_and_update(
                timestamp=current_time,
                signal=signal,
                current_price=current_price,
                ticker=TICKER
            )
            
            if trade_result:
                # 添加交易到仪表板
                dashboard.add_trade({
                    'time': current_time,
                    'type': signal,
                    'price': current_price,
                    'qty': abs(account_status['position'] - current_position)
                })
        
        # 6. 更新仪表板数据
        equity = account_status['equity']
        position = account_status['position']
        
        dashboard.update_equity(current_time, equity)
        dashboard.update_position(current_time, position)
        dashboard.update_stats(
            total_trades=len(position_manager.get_trade_log()),
            current_position=position,
            current_equity=equity,
            net_pnl=equity - INITIAL_CAPITAL
        )
        
        # 进度显示（每10次迭代）
        if iteration % 10 == 0:
            progress = (current_time - START_TIME) / (END_TIME - START_TIME) * 100
            print(f"   进度: {progress:.1f}% | 权益: ${equity:,.0f} | 持仓: {position:.0f}")
        
        current_time += timedelta(minutes=STEP_MINUTES)
    
    print("\n✅ 回测完成！")
    
except KeyboardInterrupt:
    print("\n⚠️ 用户中断回测")

# ==========================================
# 最终结果
# ==========================================

final_status = position_manager.get_account_status(current_price)
trade_log = position_manager.get_trade_log()

net_pnl = final_status['equity'] - INITIAL_CAPITAL
return_pct = (net_pnl / INITIAL_CAPITAL) * 100

print("\n" + "="*60)
print("💰 回测结果")
print("="*60)
print(f"   初始资金:  ${INITIAL_CAPITAL:,.0f}")
print(f"   最终权益:  ${final_status['equity']:,.0f}")
print(f"   净盈亏:    ${net_pnl:,.0f} ({return_pct:+.2f}%)")
print(f"   交易次数:  {len(trade_log)}")
print("="*60)

if not trade_log.empty:
    winning = trade_log[trade_log['net_pnl'] > 0]
    losing = trade_log[trade_log['net_pnl'] < 0]
    
    if len(winning) > 0:
        print(f"\n✅ 盈利: {len(winning)} 笔, 平均 ${winning['net_pnl'].mean():.2f}")
    if len(losing) > 0:
        print(f"❌ 亏损: {len(losing)} 笔, 平均 ${losing['net_pnl'].mean():.2f}")

print(f"\n📊 仪表板仍在运行: http://localhost:{DASHBOARD_PORT}")
print("   可以继续查看图表，按 Ctrl+C 退出\n")

# 保持运行，让用户查看最终图表
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n👋 再见！")
    dashboard.stop()