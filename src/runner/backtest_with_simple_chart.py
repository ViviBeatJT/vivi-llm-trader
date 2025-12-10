# src/runner/backtest_with_simple_chart.py

"""
带简单图表的回测运行器

特点：
- 无后台服务器，无线程问题
- 每次监控间隔更新一次图表
- 生成静态 HTML 文件
- 浏览器手动刷新查看
- 或使用浏览器自动刷新插件

使用方法：
1. 运行脚本
2. 自动打开浏览器
3. 每次策略运行后图表自动更新
4. 浏览器刷新查看（F5 或自动刷新插件）
"""

from datetime import datetime, timezone, timedelta
import os
from dotenv import load_dotenv
import pytz

# --- Core Modules ---
from src.cache.trading_cache import TradingCache
from src.manager.position_manager import PositionManager
from src.data_fetcher.alpaca_data_fetcher import AlpacaDataFetcher
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# --- Simple Chart Visualizer ---
from src.visualization.simple_chart_visualizer import SimpleChartVisualizer

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

# 📊 图表设置
CHART_OUTPUT_FILE = "backtest_chart.html"
AUTO_OPEN_BROWSER = True

# ==========================================
# 初始化
# ==========================================

print("\n" + "="*60)
print(f"🚀 带简单图表的回测 - {TICKER}")
print("="*60)

# 解析日期
date_parts = [int(x) for x in TRADING_DATE.split('-')]
START_TIME = US_EASTERN.localize(datetime(date_parts[0], date_parts[1], date_parts[2], 9, 30))
END_TIME = US_EASTERN.localize(datetime(date_parts[0], date_parts[1], date_parts[2], 16, 0))
START_TIME = START_TIME.astimezone(timezone.utc)
END_TIME = END_TIME.astimezone(timezone.utc)

print(f"\n📅 回测时间: {TRADING_DATE} 9:30-16:00 ET")
print(f"📊 图表文件: {CHART_OUTPUT_FILE}")
print(f"   提示: 使用浏览器自动刷新插件以查看实时更新")

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
cache = TradingCache(os.path.join('cache', f'{TICKER}_simple_chart_cache.json'))
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
# 创建图表可视化工具
# ==========================================

print(f"\n📊 初始化图表可视化...")
visualizer = SimpleChartVisualizer(
    ticker=TICKER,
    output_file=CHART_OUTPUT_FILE,
    auto_open=AUTO_OPEN_BROWSER
)

visualizer.set_initial_capital(INITIAL_CAPITAL)

# ==========================================
# 手动回测循环
# ==========================================

print(f"\n🏃 开始回测...")
print(f"   图表会按监控间隔（{STEP_MINUTES}分钟）更新")
print(f"   在浏览器中刷新页面查看最新状态")
print(f"   按 Ctrl+C 可提前停止\n")

current_time = START_TIME
iteration = 0
update_count = 0

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
        
        # 4. 执行交易
        if signal in ["BUY", "SELL", "SHORT", "COVER"]:
            print(f"{'🟢' if signal=='BUY' else '🔴' if signal=='SELL' else '🔻' if signal=='SHORT' else '🔺'} "
                  f"{current_time.strftime('%H:%M')} | {signal} | ${current_price:.2f}")
            print(f"   {reason}")
            
            position_manager.execute_and_update(
                timestamp=current_time,
                signal=signal,
                current_price=current_price,
                ticker=TICKER
            )
        
        # 5. 每次迭代都更新图表（按监控频率）
        strategy_df = strategy.get_history_data(TICKER)
        trade_log = position_manager.get_trade_log()
        
        if not strategy_df.empty:
            # 🔍 调试：检查数据列（仅第一次）
            if update_count == 0:
                print(f"\n🔍 策略数据诊断（第一次更新）:")
                print(f"   数据行数: {len(strategy_df)}")
                print(f"   数据列: {strategy_df.columns.tolist()}")
                
                # 检查布林带列
                bb_cols = ['SMA', 'BB_UPPER', 'BB_LOWER']
                for col in bb_cols:
                    if col in strategy_df.columns:
                        valid_count = strategy_df[col].notna().sum()
                        print(f"   ✅ {col}: {valid_count}/{len(strategy_df)} 有效值")
                        if valid_count > 0:
                            print(f"      范围: {strategy_df[col].min():.2f} - {strategy_df[col].max():.2f}")
                    else:
                        print(f"   ❌ {col}: 列不存在！")
                
                # 显示最后一行
                print(f"\n   最后一行数据:")
                if all(col in strategy_df.columns for col in bb_cols):
                    print(strategy_df[['close', 'SMA', 'BB_UPPER', 'BB_LOWER']].tail(1))
                else:
                    print(f"   ⚠️ 布林带列缺失，无法显示")
            
            visualizer.update_data(
                market_data=strategy_df,
                trade_log=trade_log,
                current_equity=account_status['equity'],
                current_position=current_position,
                timestamp=current_time
            )
            update_count += 1
        
        # 进度显示
        if iteration % 10 == 0:
            progress = (current_time - START_TIME) / (END_TIME - START_TIME) * 100
            print(f"   进度: {progress:.1f}% | 迭代: {iteration} | 图表更新: {update_count} 次")
        
        current_time += timedelta(minutes=STEP_MINUTES)
    
    print("\n✅ 回测完成！")
    
except KeyboardInterrupt:
    print("\n⚠️ 用户中断回测")

# ==========================================
# 最终更新和结果
# ==========================================

# 最后一次更新图表
strategy_df = strategy.get_history_data(TICKER)
trade_log = position_manager.get_trade_log()
final_status = position_manager.get_account_status(current_price)

if not strategy_df.empty:
    visualizer.update_data(
        market_data=strategy_df,
        trade_log=trade_log,
        current_equity=final_status['equity'],
        current_position=final_status['position'],
        timestamp=current_time
    )

net_pnl = final_status['equity'] - INITIAL_CAPITAL
return_pct = (net_pnl / INITIAL_CAPITAL) * 100

print("\n" + "="*60)
print("💰 回测结果")
print("="*60)
print(f"   初始资金:  ${INITIAL_CAPITAL:,.0f}")
print(f"   最终权益:  ${final_status['equity']:,.0f}")
print(f"   净盈亏:    ${net_pnl:,.0f} ({return_pct:+.2f}%)")
print(f"   交易次数:  {len(trade_log)}")
print(f"   图表更新:  {update_count} 次")
print("="*60)

if not trade_log.empty:
    winning = trade_log[trade_log['net_pnl'] > 0]
    losing = trade_log[trade_log['net_pnl'] < 0]
    
    if len(winning) > 0:
        print(f"\n✅ 盈利: {len(winning)} 笔, 平均 ${winning['net_pnl'].mean():.2f}")
    if len(losing) > 0:
        print(f"❌ 亏损: {len(losing)} 笔, 平均 ${losing['net_pnl'].mean():.2f}")

print(f"\n📊 查看最终图表: {os.path.abspath(CHART_OUTPUT_FILE)}")
print(f"   在浏览器中打开或刷新页面")
print("\n✅ 完成！\n")