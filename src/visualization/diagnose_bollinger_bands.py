# diagnose_bollinger_bands.py

"""
诊断脚本 - 检查布林带数据

用于检查策略是否正确计算和存储布林带数据
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_chart_visualizer import SimpleChartVisualizer

print("\n" + "="*60)
print("🔍 布林带数据诊断")
print("="*60)

# 创建测试数据
print("\n📊 步骤 1: 创建测试数据...")
time_index = pd.date_range(start='2024-12-05 09:30', periods=100, freq='5min')
prices = 400 + np.cumsum(np.random.randn(100) * 2)

df = pd.DataFrame({
    'open': prices - 0.5,
    'high': prices + 2,
    'low': prices - 2,
    'close': prices,
    'volume': np.random.randint(10000, 50000, 100)
}, index=time_index)

print(f"✅ 创建了 {len(df)} 根K线")
print(f"   列: {df.columns.tolist()}")

# 计算布林带
print("\n📈 步骤 2: 计算布林带...")
window = 20
df['SMA'] = df['close'].rolling(window=window).mean()
df['BB_STD'] = df['close'].rolling(window=window).std()
df['BB_UPPER'] = df['SMA'] + (df['BB_STD'] * 2)
df['BB_LOWER'] = df['SMA'] - (df['BB_STD'] * 2)

print(f"✅ 布林带计算完成")
print(f"   更新后的列: {df.columns.tolist()}")

# 检查布林带数据
print("\n🔍 步骤 3: 检查布林带数据...")

has_sma = 'SMA' in df.columns
has_upper = 'BB_UPPER' in df.columns
has_lower = 'BB_LOWER' in df.columns

print(f"   SMA 列存在: {'✅' if has_sma else '❌'}")
print(f"   BB_UPPER 列存在: {'✅' if has_upper else '❌'}")
print(f"   BB_LOWER 列存在: {'✅' if has_lower else '❌'}")

if has_sma and has_upper and has_lower:
    # 检查数值
    print("\n📊 步骤 4: 检查数值范围...")
    
    # 去除 NaN
    valid_df = df.dropna()
    
    if len(valid_df) > 0:
        print(f"   有效数据: {len(valid_df)} 行")
        print(f"   价格范围: ${valid_df['close'].min():.2f} - ${valid_df['close'].max():.2f}")
        print(f"   SMA 范围: ${valid_df['SMA'].min():.2f} - ${valid_df['SMA'].max():.2f}")
        print(f"   BB_UPPER 范围: ${valid_df['BB_UPPER'].min():.2f} - ${valid_df['BB_UPPER'].max():.2f}")
        print(f"   BB_LOWER 范围: ${valid_df['BB_LOWER'].min():.2f} - ${valid_df['BB_LOWER'].max():.2f}")
        
        # 显示最后几行
        print("\n📋 最后 5 行数据:")
        print(valid_df[['close', 'SMA', 'BB_UPPER', 'BB_LOWER']].tail())
    else:
        print("   ⚠️ 警告: 所有数据都是 NaN！")
else:
    print("   ❌ 错误: 布林带列缺失！")
    exit(1)

# 创建图表
print("\n📈 步骤 5: 生成图表...")
visualizer = SimpleChartVisualizer(
    ticker='TEST',
    output_file='diagnose_bb.html',
    auto_open=True
)

visualizer.set_initial_capital(100000.0)

# 创建空的交易记录
trade_df = pd.DataFrame()

# 更新图表
visualizer.update_data(
    market_data=df,
    trade_log=trade_df,
    current_equity=100000.0,
    current_position=0,
    timestamp=df.index[-1]
)

print("\n" + "="*60)
print("✅ 诊断完成！")
print("="*60)
print("\n检查浏览器中的图表，你应该看到:")
print("   ✅ 清晰的蜡烛图")
print("   ✅ 红色虚线（上轨）")
print("   ✅ 蓝色实线（中线/SMA）")
print("   ✅ 绿色虚线（下轨）")
print("   ✅ 淡蓝色填充区域")
print("\n如果看不到布林带线，检查:")
print("   1. 浏览器控制台是否有错误")
print("   2. 图表右侧图例中布林带是否被隐藏（点击显示）")
print("   3. Y轴范围是否太大导致线重叠\n")