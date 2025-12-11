# src/runner/bulk_backtest_runner.py

"""
批量回测运行器 - 多日期、多周期分析

功能：
1. 批量运行多日期回测
2. 每日盈亏统计
3. 月度汇总
4. 季度汇总
5. 年度汇总
6. 生成详细报告和图表

用法：
    # 单策略回测（2024年全年）
    python bulk_backtest_runner.py --strategy moderate --ticker TSLA --start 2024-12-01 --end 2025-12-01
    
    # 多策略对比
    python bulk_backtest_runner.py --strategies moderate,high_freq,ultra --ticker TSLA --start 2024-12-01 --end 2025-12-01
    
    # 只运行工作日
    python bulk_backtest_runner.py --strategy moderate --ticker TSLA --start 2024-12-01 --end 2025-12-01 --trading-days-only
"""

from datetime import datetime, timezone, timedelta
import os
from dotenv import load_dotenv
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from collections import defaultdict

# --- Core Modules ---
from src.cache.trading_cache import TradingCache
from src.manager.position_manager import PositionManager
from src.data_fetcher.alpaca_data_fetcher import AlpacaDataFetcher
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# --- Executors ---
from src.executor.simulation_executor import SimulationExecutor

# --- 所有策略 ---
from src.strategies.aggressive_mean_reversion_strategy import AggressiveMeanReversionStrategy
from src.strategies.moderate_aggressive_strategy import ModerateAggressiveStrategy
from src.strategies.high_frequency_strategy import HighFrequencyStrategy
from src.strategies.ultra_aggressive_strategy import UltraAggressiveStrategy
from src.strategies.moderate_aggressive_dynamic_strategy import ModerateAggressiveDynamicStrategy

load_dotenv()

# ==========================================
# 1. 策略配置
# ==========================================

STRATEGY_CONFIGS = {
    'conservative': {
        'class': AggressiveMeanReversionStrategy,
        'name': '原始保守策略',
        'params': {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'stop_loss_threshold': 0.10,
            'monitor_interval_seconds': 60,
        }
    },
    'moderate': {
        'class': ModerateAggressiveStrategy,
        'name': '温和进取策略',
        'params': {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'entry_threshold': 0.85,
            'exit_threshold': 0.60,
            'stop_loss_threshold': 0.10,
            'monitor_interval_seconds': 60,
        }
    },
    'moderate_dynamic': {
        'class': ModerateAggressiveDynamicStrategy,
        'name': '动态阈值温和进取策略',
        'params': {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'base_entry_threshold': 0.85,
            'aggressive_entry_threshold': 0.70,
            'exit_threshold': 0.60,
            'stop_loss_threshold': 0.10,
            'high_volatility_threshold': 0.02,
            'low_volatility_threshold': 0.01,
            'monitor_interval_seconds': 60,
        }
    },
    'high_freq': {
        'class': HighFrequencyStrategy,
        'name': '高频交易策略',
        'params': {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'strong_entry': 0.90,
            'mild_entry': 0.75,
            'exit_threshold': 0.65,
            'stop_loss_threshold': 0.08,
            'monitor_interval_seconds': 60,
        }
    },
    'ultra': {
        'class': UltraAggressiveStrategy,
        'name': '超激进动态策略',
        'params': {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'min_entry_threshold': 0.70,
            'max_entry_threshold': 0.90,
            'quick_exit_threshold': 0.55,
            'stop_loss_threshold': 0.06,
            'take_profit_threshold': 0.03,
            'monitor_interval_seconds': 60,
        }
    }
}

# ==========================================
# 2. 回测配置
# ==========================================

# 基本设置
INITIAL_CAPITAL = 1000.0
STEP_MINUTES = 1
LOOKBACK_MINUTES = 300

# 财务参数
FINANCE_PARAMS = {
    'INITIAL_CAPITAL': INITIAL_CAPITAL,
    'COMMISSION_RATE': 0.0003,
    'SLIPPAGE_RATE': 0.0001,
    'MIN_LOT_SIZE': 1,
    'MAX_ALLOCATION': 0.95,
}


# ==========================================
# 3. 单日回测函数
# ==========================================

def run_single_day_backtest(
    ticker: str,
    date_str: str,
    strategy_name: str,
    initial_capital: float = INITIAL_CAPITAL,  # ✨ 新增参数
    verbose: bool = False,
    log_dir: str = None
) -> Dict:
    """
    运行单日回测
    
    Args:
        ticker: 股票代码
        date_str: 日期字符串 'YYYY-MM-DD'
        strategy_name: 策略名称
        initial_capital: 初始资金（用于连续回测）
        verbose: 是否打印详细信息
        log_dir: 日志目录
        
    Returns:
        dict: 回测结果
    """
    import pytz
    import sys
    
    US_EASTERN = pytz.timezone('America/New_York')
    
    # 设置日志文件
    log_file = None
    original_stdout = sys.stdout
    
    if log_dir:
        log_file_path = Path(log_dir) / f"{date_str}_{strategy_name}.log"
        log_file = open(log_file_path, 'w', encoding='utf-8')
        sys.stdout = log_file
    
    try:
        print(f"{'='*80}")
        print(f"单日回测 - {date_str}")
        print(f"{'='*80}")
        print(f"股票: {ticker}")
        print(f"策略: {strategy_name} ({STRATEGY_CONFIGS[strategy_name]['name']})")
        print(f"初始资金: ${initial_capital:,.2f}")  # ✨ 使用传入的资金
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # 解析日期
        date_parts = [int(x) for x in date_str.split('-')]
        start_time = US_EASTERN.localize(datetime(date_parts[0], date_parts[1], date_parts[2], 9, 30))
        end_time = US_EASTERN.localize(datetime(date_parts[0], date_parts[1], date_parts[2], 16, 0))
        
        # 转换为 UTC
        start_time = start_time.astimezone(timezone.utc)
        end_time = end_time.astimezone(timezone.utc)
        
        # ✨ 使用传入的初始资金
        finance_params = FINANCE_PARAMS.copy()
        finance_params['INITIAL_CAPITAL'] = initial_capital
        
        # 初始化组件
        data_fetcher = AlpacaDataFetcher()
        executor = SimulationExecutor(finance_params)
        position_manager = PositionManager(executor, finance_params)
        
        # 创建策略
        strategy_config = STRATEGY_CONFIGS[strategy_name]
        strategy_class = strategy_config['class']
        params = strategy_config['params']
        strategy = strategy_class(**params)
        
        # 回测循环
        current_time = start_time
        iteration = 0
        
        while current_time <= end_time:
            iteration += 1
            
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=timezone.utc)
            
            # 获取数据
            df = data_fetcher.get_latest_bars(
                ticker=ticker,
                lookback_minutes=LOOKBACK_MINUTES,
                end_dt=current_time,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute)
            )
            
            if df.empty:
                current_time += timedelta(minutes=STEP_MINUTES)
                continue
            
            current_price = df.iloc[-1]['close']
            
            # 获取当前账户状态
            account_status = position_manager.get_account_status(current_price)
            current_position = account_status.get('position', 0.0)
            avg_cost = account_status.get('avg_cost', 0.0)
            
            # 获取信号
            current_et = current_time.astimezone(pytz.timezone('America/New_York'))
            is_close_to_market_close = current_et.hour == 15 and current_et.minute >= 55
            
            signal_data, _ = strategy.get_signal(
                ticker=ticker,
                new_data=df,
                current_position=current_position,
                avg_cost=avg_cost,
                verbose=True,  # ✨ 启用详细输出到日志
                is_market_close=is_close_to_market_close,
                current_time_et=current_et
            )
            
            signal = signal_data['signal']
            
            # 执行交易
            if signal in ['BUY', 'SELL', 'SHORT', 'COVER']:
                position_manager.execute_and_update(
                    timestamp=current_time,
                    signal=signal,
                    current_price=current_price,
                    ticker=ticker
                )
            
            # 前进1分钟
            current_time += timedelta(minutes=STEP_MINUTES)
    
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ 回测错误")
        print(f"{'='*80}")
        print(f"错误信息: {e}")
        print(f"错误位置: 迭代 {iteration}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # 恢复标准输出
        if log_file:
            sys.stdout = original_stdout
            log_file.close()
    
    # 获取最终结果
    try:
        df_final = data_fetcher.get_latest_bars(
            ticker=ticker,
            lookback_minutes=LOOKBACK_MINUTES,
            end_dt=end_time,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute)
        )
        
        final_price = df_final.iloc[-1]['close'] if not df_final.empty else current_price
        final_status = position_manager.get_account_status(final_price)
        trade_log = position_manager.get_trade_log()
        
        # ✨ 安全获取 PnL 数据
        total_pnl = final_status.get('total_pnl', 0.0)
        total_pnl_pct = final_status.get('total_pnl_pct', 0.0)
        
        # 如果 position_manager 没有返回 total_pnl，手动计算
        if total_pnl == 0.0 and 'equity' in final_status and 'cash' in final_status:
            final_equity = final_status['equity']
            total_pnl = final_equity - initial_capital
            total_pnl_pct = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0.0
        
        # 写入最终汇总到日志
        if log_dir:
            log_file_path = Path(log_dir) / f"{date_str}_{strategy_name}.log"
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"回测完成 - 最终结果\n")
                f.write(f"{'='*80}\n")
                f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总迭代数: {iteration}\n")
                f.write(f"最终价格: ${final_price:.2f}\n")
                f.write(f"初始资金: ${initial_capital:,.2f}\n")  # ✨ 新增
                f.write(f"最终权益: ${final_status.get('equity', 0.0):,.2f}\n")
                f.write(f"盈亏: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)\n")
                f.write(f"最终持仓: {final_status.get('position', 0.0):.0f} 股\n")
                
                if not trade_log.empty:
                    f.write(f"\n交易记录:\n")
                    f.write(f"-"*80 + "\n")
                    for idx, row in trade_log.iterrows():
                        f.write(f"{row['time'].strftime('%H:%M:%S')} | {row['type']:6s} | "
                               f"{row['qty']:3.0f} @ ${row['price']:7.2f} | "
                               f"PnL: ${row.get('net_pnl', 0):+7.2f}\n")
                
                f.write(f"{'='*80}\n")
        
        # 统计
        total_trades = len(trade_log) if not trade_log.empty else 0
        
        if not trade_log.empty and 'type' in trade_log.columns:
            completed_trades = trade_log[trade_log['type'].isin(['SELL', 'COVER'])]
            if not completed_trades.empty and 'net_pnl' in completed_trades.columns:
                winning_trades = len(completed_trades[completed_trades['net_pnl'] > 0])
                losing_trades = len(completed_trades[completed_trades['net_pnl'] < 0])
                win_rate = winning_trades / len(completed_trades) if len(completed_trades) > 0 else 0
            else:
                winning_trades = 0
                losing_trades = 0
                win_rate = 0
        else:
            completed_trades = pd.DataFrame()
            winning_trades = 0
            losing_trades = 0
            win_rate = 0
        
        return {
            'date': date_str,
            'ticker': ticker,
            'strategy': strategy_name,
            'initial_capital': initial_capital,  # ✨ 返回实际使用的初始资金
            'final_equity': final_status.get('equity', initial_capital),
            'pnl': total_pnl,
            'pnl_pct': total_pnl_pct,
            'total_trades': total_trades,
            'completed_trades': len(completed_trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'final_position': final_status.get('position', 0.0),
            'iterations': iteration
        }
    
    except Exception as e:
        print(f"❌ 获取最终结果失败: {e}", file=original_stdout)
        import traceback
        traceback.print_exc(file=original_stdout)
        return None


# ==========================================
# 4. 批量回测函数
# ==========================================

def get_trading_dates(start_date: str, end_date: str, trading_days_only: bool = True) -> List[str]:
    """
    获取交易日期列表
    
    Args:
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        trading_days_only: 是否只包含工作日
        
    Returns:
        List[str]: 日期列表
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    dates = []
    current = start
    
    while current <= end:
        # 跳过周末
        if trading_days_only and current.weekday() >= 5:
            current += timedelta(days=1)
            continue
        
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    return dates


def run_bulk_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    strategies: List[str],
    trading_days_only: bool = True,
    output_dir: str = 'bulk_backtest_results',
    consecutive_capital: bool = True  # ✨ 新增参数
) -> pd.DataFrame:
    """
    批量回测
    
    Args:
        ticker: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        strategies: 策略列表
        trading_days_only: 是否只包含工作日
        output_dir: 输出目录
        consecutive_capital: 是否使用连续资金（Day2使用Day1的结束资金）
        
    Returns:
        pd.DataFrame: 所有回测结果
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # ✨ 创建日志目录
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取交易日期
    dates = get_trading_dates(start_date, end_date, trading_days_only)
    
    print(f"\n{'='*60}")
    print(f"🚀 批量回测")
    print(f"{'='*60}")
    print(f"   股票: {ticker}")
    print(f"   日期范围: {start_date} 到 {end_date}")
    print(f"   交易日数: {len(dates)}")
    print(f"   策略: {', '.join(strategies)}")
    print(f"   输出目录: {output_dir}")
    print(f"   日志目录: {log_dir}")
    print(f"   连续资金: {'是' if consecutive_capital else '否'}")  # ✨ 新增
    print(f"{'='*60}\n")
    
    # 运行回测
    all_results = []
    
    total_runs = len(dates) * len(strategies)
    current_run = 0
    
    for strategy_name in strategies:
        print(f"\n📊 策略: {STRATEGY_CONFIGS[strategy_name]['name']}")
        print(f"{'='*60}")
        
        # ✨ 为每个策略维护独立的资金链
        current_capital = INITIAL_CAPITAL
        
        for date_str in dates:
            current_run += 1
            progress = current_run / total_runs * 100
            
            print(f"[{progress:5.1f}%] {date_str} - {strategy_name} (${current_capital:,.0f})...", 
                  end=' ', flush=True)
            
            result = run_single_day_backtest(
                ticker=ticker,
                date_str=date_str,
                strategy_name=strategy_name,
                initial_capital=current_capital,  # ✨ 传入当前资金
                verbose=False,
                log_dir=str(log_dir)
            )
            
            if result:
                all_results.append(result)
                status = "✅" if result['pnl'] >= 0 else "❌"
                print(f"{status} PnL: ${result['pnl']:+.2f} ({result['pnl_pct']:+.2f}%) → ${result['final_equity']:,.0f} | "
                      f"Log: logs/{date_str}_{strategy_name}.log")
                
                # ✨ 更新下一天的资金
                if consecutive_capital:
                    current_capital = result['final_equity']
            else:
                print("⚠️ 跳过（无数据或错误）")
                # 如果失败，保持当前资金不变
    
    # 转换为 DataFrame
    df = pd.DataFrame(all_results)
    
    # ✨ 添加累计权益列
    if consecutive_capital and not df.empty:
        for strategy in df['strategy'].unique():
            strategy_mask = df['strategy'] == strategy
            df.loc[strategy_mask, 'cumulative_equity'] = df.loc[strategy_mask, 'final_equity']
    
    # 保存原始结果
    df.to_csv(f"{output_dir}/daily_results.csv", index=False)
    print(f"\n✅ 每日结果已保存: {output_dir}/daily_results.csv")
    print(f"✅ 每日日志已保存: {log_dir}/ (共 {len(all_results)} 个文件)")
    
    # ✨ 打印最终资金汇总
    if consecutive_capital and not df.empty:
        print(f"\n💰 最终资金汇总:")
        for strategy in df['strategy'].unique():
            strategy_df = df[df['strategy'] == strategy]
            if not strategy_df.empty:
                initial = strategy_df.iloc[0]['initial_capital']
                final = strategy_df.iloc[-1]['final_equity']
                total_return = final - initial
                total_return_pct = (total_return / initial * 100) if initial > 0 else 0
                print(f"   {strategy:20s}: ${initial:,.2f} → ${final:,.2f} "
                      f"(${total_return:+,.2f}, {total_return_pct:+.2f}%)")
    
    return df


# ==========================================
# 5. 汇总分析函数
# ==========================================

def generate_summary_reports(df: pd.DataFrame, output_dir: str):
    """
    生成汇总报告
    
    Args:
        df: 每日回测结果
        output_dir: 输出目录
    """
    if df.empty:
        print("⚠️ 无数据，无法生成汇总报告")
        return
    
    # 确保日期列是 datetime 类型
    df['date'] = pd.to_datetime(df['date'])
    
    # 添加时间维度列
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year_month'] = df['date'].dt.to_period('M')
    df['year_quarter'] = df['date'].dt.to_period('Q')
    
    print(f"\n{'='*60}")
    print(f"📊 生成汇总报告")
    print(f"{'='*60}")
    
    # === 1. 每日汇总 ===
    daily_summary = df.groupby(['date', 'strategy']).agg({
        'pnl': 'sum',
        'pnl_pct': 'mean',
        'total_trades': 'sum',
        'completed_trades': 'sum',
        'winning_trades': 'sum',
        'losing_trades': 'sum',
        'win_rate': 'mean'
    }).reset_index()
    
    daily_summary.to_csv(f"{output_dir}/daily_summary.csv", index=False)
    print(f"✅ 每日汇总: {output_dir}/daily_summary.csv")
    
    # === 2. 月度汇总 ===
    monthly_summary = df.groupby(['year_month', 'strategy']).agg({
        'pnl': 'sum',
        'pnl_pct': 'mean',
        'total_trades': 'sum',
        'completed_trades': 'sum',
        'winning_trades': 'sum',
        'losing_trades': 'sum',
        'win_rate': 'mean'
    }).reset_index()
    
    monthly_summary['year_month'] = monthly_summary['year_month'].astype(str)
    monthly_summary.to_csv(f"{output_dir}/monthly_summary.csv", index=False)
    print(f"✅ 月度汇总: {output_dir}/monthly_summary.csv")
    
    # === 3. 季度汇总 ===
    quarterly_summary = df.groupby(['year_quarter', 'strategy']).agg({
        'pnl': 'sum',
        'pnl_pct': 'mean',
        'total_trades': 'sum',
        'completed_trades': 'sum',
        'winning_trades': 'sum',
        'losing_trades': 'sum',
        'win_rate': 'mean'
    }).reset_index()
    
    quarterly_summary['year_quarter'] = quarterly_summary['year_quarter'].astype(str)
    quarterly_summary.to_csv(f"{output_dir}/quarterly_summary.csv", index=False)
    print(f"✅ 季度汇总: {output_dir}/quarterly_summary.csv")
    
    # === 4. 年度汇总 ===
    yearly_summary = df.groupby(['year', 'strategy']).agg({
        'pnl': 'sum',
        'pnl_pct': 'mean',
        'total_trades': 'sum',
        'completed_trades': 'sum',
        'winning_trades': 'sum',
        'losing_trades': 'sum',
        'win_rate': 'mean'
    }).reset_index()
    
    yearly_summary.to_csv(f"{output_dir}/yearly_summary.csv", index=False)
    print(f"✅ 年度汇总: {output_dir}/yearly_summary.csv")
    
    # === 5. 策略对比汇总 ===
    strategy_comparison = df.groupby('strategy').agg({
        'pnl': ['sum', 'mean', 'std', 'min', 'max'],
        'pnl_pct': ['mean', 'std', 'min', 'max'],
        'total_trades': 'sum',
        'completed_trades': 'sum',
        'winning_trades': 'sum',
        'losing_trades': 'sum',
        'win_rate': 'mean'
    }).reset_index()
    
    strategy_comparison.columns = ['_'.join(col).strip('_') for col in strategy_comparison.columns.values]
    strategy_comparison.to_csv(f"{output_dir}/strategy_comparison.csv", index=False)
    print(f"✅ 策略对比: {output_dir}/strategy_comparison.csv")
    
    # === 6. 生成文本报告 ===
    generate_text_report(df, yearly_summary, monthly_summary, strategy_comparison, output_dir)


def generate_text_report(df: pd.DataFrame, yearly: pd.DataFrame, monthly: pd.DataFrame, 
                         strategy_comp: pd.DataFrame, output_dir: str):
    """生成文本格式的详细报告"""
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("批量回测详细报告")
    report_lines.append("="*80)
    report_lines.append("")
    
    # 基本信息
    report_lines.append("📋 基本信息")
    report_lines.append("-"*80)
    report_lines.append(f"   股票代码: {df['ticker'].iloc[0]}")
    report_lines.append(f"   日期范围: {df['date'].min()} 到 {df['date'].max()}")
    report_lines.append(f"   交易日数: {df['date'].nunique()}")
    report_lines.append(f"   策略数量: {df['strategy'].nunique()}")
    report_lines.append(f"   初始资金: ${df['initial_capital'].iloc[0]:,.2f}")
    report_lines.append("")
    
    # 年度汇总
    report_lines.append("📊 年度汇总")
    report_lines.append("-"*80)
    for _, row in yearly.iterrows():
        report_lines.append(f"\n{row['year']} - {row['strategy']}:")
        report_lines.append(f"   总盈亏: ${row['pnl']:,.2f}")
        report_lines.append(f"   平均日收益率: {row['pnl_pct']:.2f}%")
        report_lines.append(f"   总交易数: {row['total_trades']:.0f}")
        report_lines.append(f"   完成交易: {row['completed_trades']:.0f}")
        report_lines.append(f"   胜率: {row['win_rate']*100:.1f}%")
    report_lines.append("")
    
    # 策略对比
    report_lines.append("🏆 策略对比")
    report_lines.append("-"*80)
    for _, row in strategy_comp.iterrows():
        report_lines.append(f"\n{row['strategy']}:")
        report_lines.append(f"   累计盈亏: ${row['pnl_sum']:,.2f}")
        report_lines.append(f"   平均日盈亏: ${row['pnl_mean']:,.2f}")
        report_lines.append(f"   盈亏标准差: ${row['pnl_std']:,.2f}")
        report_lines.append(f"   最大单日盈利: ${row['pnl_max']:,.2f}")
        report_lines.append(f"   最大单日亏损: ${row['pnl_min']:,.2f}")
        report_lines.append(f"   平均胜率: {row['win_rate_mean']*100:.1f}%")
    report_lines.append("")
    
    # 最佳/最差交易日
    report_lines.append("📈 最佳/最差交易日")
    report_lines.append("-"*80)
    
    for strategy in df['strategy'].unique():
        strategy_df = df[df['strategy'] == strategy]
        best_day = strategy_df.loc[strategy_df['pnl'].idxmax()]
        worst_day = strategy_df.loc[strategy_df['pnl'].idxmin()]
        
        report_lines.append(f"\n{strategy}:")
        report_lines.append(f"   最佳: {best_day['date'].strftime('%Y-%m-%d')} - ${best_day['pnl']:+.2f} ({best_day['pnl_pct']:+.2f}%)")
        report_lines.append(f"   最差: {worst_day['date'].strftime('%Y-%m-%d')} - ${worst_day['pnl']:+.2f} ({worst_day['pnl_pct']:+.2f}%)")
    
    report_lines.append("")
    report_lines.append("="*80)
    
    # 保存报告
    report_text = "\n".join(report_lines)
    with open(f"{output_dir}/detailed_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✅ 详细报告: {output_dir}/detailed_report.txt")
    
    # 打印到控制台
    print("\n" + report_text)


# ==========================================
# 6. 主函数
# ==========================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量回测运行器')
    
    parser.add_argument('--ticker', type=str, default='TSLA',
                       help='股票代码 (默认: TSLA)')
    
    parser.add_argument('--start', type=str, required=True,
                       help='开始日期 (YYYY-MM-DD)')
    
    parser.add_argument('--end', type=str, required=True,
                       help='结束日期 (YYYY-MM-DD)')
    
    parser.add_argument('--strategies', type=str, default='moderate',
                       help='策略列表，逗号分隔 (默认: moderate)')
    
    parser.add_argument('--trading-days-only', action='store_true',
                       help='只包含工作日（跳过周末）')
    
    parser.add_argument('--output-dir', type=str, default='bulk_backtest_results',
                       help='输出目录 (默认: bulk_backtest_results)')
    
    parser.add_argument('--no-consecutive-capital', action='store_true',
                       help='禁用连续资金（每天都从初始资金开始）')
    
    args = parser.parse_args()
    
    # 解析策略列表
    strategies = [s.strip() for s in args.strategies.split(',')]
    
    # 验证策略
    for strategy in strategies:
        if strategy not in STRATEGY_CONFIGS:
            print(f"❌ 未知策略: {strategy}")
            print(f"   可选策略: {', '.join(STRATEGY_CONFIGS.keys())}")
            return
    
    # 运行批量回测
    df = run_bulk_backtest(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        strategies=strategies,
        trading_days_only=args.trading_days_only,
        output_dir=args.output_dir,
        consecutive_capital=not args.no_consecutive_capital  # ✨ 新增
    )
    
    # 生成汇总报告
    if not df.empty:
        generate_summary_reports(df, args.output_dir)
        
        print(f"\n{'='*60}")
        print(f"✅ 批量回测完成！")
        print(f"   结果保存在: {args.output_dir}/")
        print(f"{'='*60}\n")
    else:
        print("\n❌ 批量回测失败，无有效结果")


if __name__ == '__main__':
    main()