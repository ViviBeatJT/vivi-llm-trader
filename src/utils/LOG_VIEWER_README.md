# Bulk Backtest Logging System Guide

## Overview
The bulk backtest system now saves detailed logs to separate files for each trading day, making it easy to review individual day performance and debug issues.

---

## Log File Structure

### Directory Layout
```
bulk_backtest_results/
├── logs/                          # ✨ All log files here
│   ├── 2024-12-01_moderate.log
│   ├── 2024-12-02_moderate.log
│   ├── 2024-12-01_high_freq.log
│   └── ...
├── daily_results.csv
├── monthly_summary.csv
└── ...
```

### Log File Naming
Format: `YYYY-MM-DD_strategy.log`

Examples:
- `2024-12-05_moderate.log` - Moderate strategy on Dec 5, 2024
- `2024-12-05_high_freq.log` - High frequency strategy on Dec 5, 2024

---

## What's in a Log File

### Example Log Structure
```
================================================================================
单日回测 - 2024-12-05
================================================================================
股票: TSLA
策略: moderate (温和进取策略)
初始资金: $1,000.00
开始时间: 2024-12-11 10:30:45
================================================================================

📊 [09:30 AM] TSLA: 20 条K线 | 无仓
   价格: $369.50 | BB位置: 45.2% | 范围: [$365.20, $370.50, $375.80]
   ⚪ HOLD (3/10) - 价格在区间内 45.2%

📊 [09:31 AM] TSLA: 20 条K线 | 无仓
   价格: $368.80 | BB位置: 23.5% | 范围: [$365.20, $370.50, $375.80]
   🟢 BUY (8/10) - 价格接近下轨！位置 23.5%

[Trade execution details...]

📊 [15:55 PM] TSLA: 300 条K线 | 多1股
   价格: $372.30 | BB位置: 65.8% | 范围: [$365.20, $370.50, $375.80]
   🔴 SELL (7/10) - 多仓获利平仓！位置回到 65.8%

================================================================================
回测完成 - 最终结果
================================================================================
结束时间: 2024-12-11 10:35:12
总迭代数: 391
最终价格: $372.30
最终权益: $1,015.50
盈亏: $15.50 (+1.55%)
最终持仓: 0 股

交易记录:
--------------------------------------------------------------------------------
09:31:20 | BUY    |   1 @ $368.80 | PnL:   $0.00
15:55:45 | SELL   |   1 @ $372.30 | PnL:  $+3.20
================================================================================
```

### Key Sections

#### 1. Header
- Date and strategy
- Initial capital
- Start time

#### 2. Strategy Output (Every Minute)
- Timestamp
- Current position
- Price and BB position
- Signal with confidence
- Reasoning

#### 3. Final Summary
- Total iterations
- Final price and equity
- P&L ($ and %)
- Final position (should be 0)
- Trade log with timestamps

---

## Using the Log Viewer

### Installation
```bash
# Copy to your project
cp log_viewer.py src/utils/log_viewer.py

# Or run directly
python log_viewer.py --log-dir bulk_backtest_results/logs [options]
```

### Common Commands

#### 1. List All Logs
```bash
python log_viewer.py --log-dir bulk_backtest_results/logs --list
```

**Output:**
```
================================================================================
📋 日志文件列表 (共 252 个)
================================================================================

  1. 2024-12-01_moderate.log                    (  125.3 KB)
  2. 2024-12-02_moderate.log                    (  118.7 KB)
  3. 2024-12-03_moderate.log                    (  132.1 KB)
  ...
```

#### 2. View Specific Date
```bash
# Single strategy for that date
python log_viewer.py --log-dir bulk_backtest_results/logs --date 2024-12-05

# Specific strategy
python log_viewer.py --log-dir bulk_backtest_results/logs --date 2024-12-05 --strategy moderate
```

#### 3. View by Filename
```bash
python log_viewer.py --log-dir bulk_backtest_results/logs --filename 2024-12-05_moderate.log
```

#### 4. Find Best/Worst Days
```bash
# Top 5 most profitable days
python log_viewer.py --log-dir bulk_backtest_results/logs --top-profit 5

# Top 5 worst days
python log_viewer.py --log-dir bulk_backtest_results/logs --top-loss 5
```

**Output:**
```
================================================================================
🏆 盈利最多的 5 天
================================================================================

1. 2024-03-15_moderate.log                  | PnL:   $+89.30
2. 2024-07-22_moderate.log                  | PnL:   $+67.80
3. 2024-11-08_moderate.log                  | PnL:   $+54.20
4. 2024-02-14_moderate.log                  | PnL:   $+48.90
5. 2024-09-19_moderate.log                  | PnL:   $+45.60
```

#### 5. Search Logs
```bash
# Find logs with stop loss triggers
python log_viewer.py --log-dir bulk_backtest_results/logs --search "止损"

# Find logs with forced closure
python log_viewer.py --log-dir bulk_backtest_results/logs --search "强制平仓"

# Find logs with specific signal
python log_viewer.py --log-dir bulk_backtest_results/logs --search "SHORT"
```

**Output:**
```
================================================================================
🔍 搜索关键词: '止损'
================================================================================

📄 2024-08-22_moderate.log
--------------------------------------------------------------------------------
   ⚠️ 止损！多仓亏损 10.25%
   SELL (10/10) - ⚠️ 止损！多仓亏损 10.25%

📄 2024-11-15_moderate.log
--------------------------------------------------------------------------------
   ⚠️ 止损！空仓亏损 10.08%
   COVER (10/10) - ⚠️ 止损！空仓亏损 10.08%

✅ 找到 2 个包含 '止损' 的日志文件
```

#### 6. Generate Summary
```bash
python log_viewer.py --log-dir bulk_backtest_results/logs --summary
```

**Output:**
```
================================================================================
📊 日志汇总统计
================================================================================

日志文件总数: 252
交易日数: 252
策略数: 1
策略列表: moderate

盈亏统计:
  总盈亏: $2,345.60
  平均盈亏: $9.31
  中位数: $8.50
  标准差: $23.45
  最大盈利: $89.30
  最大亏损: $-45.20

盈亏分布:
  盈利天数: 147 (58.3%)
  持平天数: 5 (2.0%)
  亏损天数: 100 (39.7%)
```

#### 7. View Latest Logs
```bash
# Show last 10 processed logs
python log_viewer.py --log-dir bulk_backtest_results/logs --tail 10
```

---

## Console Output During Backtest

### Before (Old - Cluttered)
```
📊 [09:30] TSLA: 20 条K线 | 无仓
   价格: $369.50 | BB位置: 45.2%
   ⚪ HOLD (3/10) - 价格在区间内
📊 [09:31] TSLA: 20 条K线 | 无仓
   价格: $368.80 | BB位置: 23.5%
   🟢 BUY (8/10) - 价格接近下轨
[... 390 more lines per day ...]
```

### After (New - Clean)
```
================================================================================
🚀 批量回测
================================================================================
   股票: TSLA
   日期范围: 2024-12-01 到 2025-12-01
   交易日数: 252
   策略: moderate
   输出目录: bulk_backtest_results
   日志目录: bulk_backtest_results/logs
================================================================================

📊 策略: 温和进取策略
================================================================================
[  0.4%] 2024-12-01 - moderate... ✅ PnL: $+15.50 (+1.55%) | Log: logs/2024-12-01_moderate.log
[  0.8%] 2024-12-02 - moderate... ❌ PnL: $-5.30 (-0.53%) | Log: logs/2024-12-02_moderate.log
[  1.2%] 2024-12-03 - moderate... ✅ PnL: $+8.70 (+0.87%) | Log: logs/2024-12-03_moderate.log
...
[100.0%] 2025-12-01 - moderate... ✅ PnL: $+12.40 (+1.24%) | Log: logs/2025-12-01_moderate.log

✅ 每日结果已保存: bulk_backtest_results/daily_results.csv
✅ 每日日志已保存: bulk_backtest_results/logs/ (共 252 个文件)
```

---

## Use Cases

### Use Case 1: Debug a Bad Day
```bash
# Find worst day
python log_viewer.py --log-dir bulk_backtest_results/logs --top-loss 1

# Output: 2024-08-22_moderate.log | PnL: $-45.20

# View full log
python log_viewer.py --log-dir bulk_backtest_results/logs --date 2024-08-22 --strategy moderate
```

### Use Case 2: Analyze Stop Losses
```bash
# Find all stop loss events
python log_viewer.py --log-dir bulk_backtest_results/logs --search "止损"

# Review each log to understand why stop loss triggered
```

### Use Case 3: Check End-of-Day Closure
```bash
# Search for forced closures
python log_viewer.py --log-dir bulk_backtest_results/logs --search "强制平仓"

# Verify all positions closed
python log_viewer.py --log-dir bulk_backtest_results/logs --search "最终持仓: 0 股"
```

### Use Case 4: Compare Strategy Performance
```bash
# View same day, different strategies
python log_viewer.py --log-dir bulk_backtest_results/logs --date 2024-12-05 --strategy moderate
python log_viewer.py --log-dir bulk_backtest_results/logs --date 2024-12-05 --strategy high_freq
```

### Use Case 5: Find Patterns
```bash
# Days with many trades
python log_viewer.py --log-dir bulk_backtest_results/logs --search "总交易数"

# Days with high volatility
python log_viewer.py --log-dir bulk_backtest_results/logs --search "BB位置: 9"
```

---

## Advanced Tips

### Tip 1: Grep Through Logs (Linux/Mac)
```bash
# Find all BUY signals
grep -r "BUY" bulk_backtest_results/logs/

# Count stop losses
grep -r "止损" bulk_backtest_results/logs/ | wc -l

# Find days with > $50 profit
grep -r "盈亏: \$+[5-9][0-9]\." bulk_backtest_results/logs/
```

### Tip 2: Analyze in Python
```python
import pandas as pd
from pathlib import Path
import re

# Read all logs and extract PnL
log_dir = Path('bulk_backtest_results/logs')
results = []

for log_file in log_dir.glob('*.log'):
    with open(log_file) as f:
        content = f.read()
        match = re.search(r'盈亏: \$([+-]?\d+\.\d+)', content)
        if match:
            pnl = float(match.group(1))
            results.append({
                'date': log_file.stem.split('_')[0],
                'strategy': '_'.join(log_file.stem.split('_')[1:]),
                'pnl': pnl
            })

df = pd.DataFrame(results)
print(df.groupby('strategy')['pnl'].describe())
```

### Tip 3: Create Custom Reports
```bash
# Extract all final summaries
for log in bulk_backtest_results/logs/*.log; do
    echo "=== $(basename $log) ==="
    tail -20 "$log"
done > all_summaries.txt
```

### Tip 4: Watch Logs in Real-Time
```bash
# While backtest is running, watch latest log
watch -n 5 "ls -t bulk_backtest_results/logs/*.log | head -1 | xargs tail -20"
```

---

## Log File Size Management

### Typical Sizes
- **1 minute intervals**: ~100-150 KB per day
- **252 trading days**: ~25-38 MB total
- **Multiple strategies**: Multiply by number of strategies

### Compression (Optional)
```bash
# Compress old logs
gzip bulk_backtest_results/logs/*.log

# View compressed logs
zcat bulk_backtest_results/logs/2024-12-05_moderate.log.gz | less
```

### Cleanup Old Logs
```bash
# Delete logs older than 30 days
find bulk_backtest_results/logs -name "*.log" -mtime +30 -delete

# Archive by month
tar -czf logs_2024_12.tar.gz bulk_backtest_results/logs/2024-12-*.log
```

---

## Troubleshooting

### Problem: Log file is empty
**Cause**: Error occurred before any output

**Solution**: Check console output for errors

### Problem: Log file missing final summary
**Cause**: Backtest crashed before completion

**Solution**: Look for error messages at end of log

### Problem: Can't find log file
**Cause**: Backtest was skipped (no data)

**Solution**: Check console output for "⚠️ 跳过" message

### Problem: Too many log files
**Cause**: Running multiple strategies over long periods

**Solution**: Organize by subdirectories:
```bash
mkdir -p bulk_backtest_results/logs/moderate
mv bulk_backtest_results/logs/*_moderate.log bulk_backtest_results/logs/moderate/
```

---

## Best Practices

### 1. Regular Review
```bash
# Weekly review of worst days
python log_viewer.py --log-dir bulk_backtest_results/logs --top-loss 5
```

### 2. Quick Check After Backtest
```bash
# Verify all days completed
python log_viewer.py --log-dir bulk_backtest_results/logs --summary
```

### 3. Debug Systematically
1. Find problem day (top-loss)
2. View full log
3. Search for patterns across multiple days
4. Adjust strategy parameters
5. Re-run backtest

### 4. Archive Results
```bash
# After analysis, archive results
DATE=$(date +%Y%m%d)
tar -czf backtest_${DATE}.tar.gz bulk_backtest_results/
```

---

## Summary

### Key Benefits
✅ **Clean Console**: No clutter during batch processing
✅ **Detailed Logs**: Full strategy output for each day
✅ **Easy Debugging**: Quick access to problem days
✅ **Searchable**: Find patterns and issues
✅ **Organized**: One file per day/strategy

### Essential Commands
```bash
# List all logs
python log_viewer.py --log-dir logs --list

# View specific day
python log_viewer.py --log-dir logs --date 2024-12-05

# Find best/worst
python log_viewer.py --log-dir logs --top-profit 5
python log_viewer.py --log-dir logs --top-loss 5

# Search
python log_viewer.py --log-dir logs --search "关键词"

# Summary
python log_viewer.py --log-dir logs --summary
```

### Workflow
1. Run bulk backtest → Logs saved automatically
2. Review summary → Check overall performance
3. Find problem days → Use top-loss or search
4. View detailed logs → Understand what went wrong
5. Adjust strategy → Re-run backtest