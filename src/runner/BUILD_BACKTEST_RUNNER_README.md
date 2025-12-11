# Bulk Backtesting Guide

## Overview
The bulk backtest system allows you to run backtests across multiple days, analyze performance over different time periods, and compare strategies.

---

## Quick Start

### 1. Basic Usage (Single Strategy, One Year)
```bash
python bulk_backtest_runner.py \
    --ticker TSLA \
    --start 2024-12-01 \
    --end 2025-12-01 \
    --strategies moderate \
    --trading-days-only
```

### 2. Multiple Strategies Comparison
```bash
python bulk_backtest_runner.py \
    --ticker TSLA \
    --start 2024-12-01 \
    --end 2025-12-01 \
    --strategies moderate,high_freq,ultra \
    --trading-days-only \
    --output-dir results_2024_comparison
```

### 3. Generate Visualizations
```bash
# After running backtest
python bulk_backtest_visualizer.py --results-dir bulk_backtest_results
```

---

## Command Line Arguments

### Required Arguments
- `--start`: Start date in YYYY-MM-DD format
- `--end`: End date in YYYY-MM-DD format

### Optional Arguments
- `--ticker`: Stock symbol (default: TSLA)
- `--strategies`: Comma-separated strategy names (default: moderate)
  - Available: `conservative`, `moderate`, `moderate_dynamic`, `high_freq`, `ultra`
- `--trading-days-only`: Skip weekends (recommended)
- `--output-dir`: Output directory (default: bulk_backtest_results)

---

## Output Files

### Directory Structure
```
bulk_backtest_results/
├── daily_results.csv          # Raw daily results
├── daily_summary.csv          # Daily aggregated
├── monthly_summary.csv        # Monthly aggregated
├── quarterly_summary.csv      # Quarterly aggregated
├── yearly_summary.csv         # Yearly aggregated
├── strategy_comparison.csv    # Strategy statistics
├── detailed_report.txt        # Human-readable report
└── visualizations/
    ├── cumulative_returns.html
    ├── monthly_returns.html
    ├── strategy_comparison.html
    ├── drawdown_analysis.html
    └── win_loss_distribution.html
```

### File Descriptions

#### 1. daily_results.csv
Raw results for each trading day:
```csv
date,ticker,strategy,initial_capital,final_equity,pnl,pnl_pct,total_trades,completed_trades,winning_trades,losing_trades,win_rate,final_position,iterations
2024-12-01,TSLA,moderate,1000.0,1015.5,15.5,1.55,4,2,1,1,0.50,0,391
```

**Columns:**
- `date`: Trading date
- `pnl`: Profit/Loss in dollars
- `pnl_pct`: Return percentage
- `total_trades`: All trades (BUY/SELL/SHORT/COVER)
- `completed_trades`: Closed positions (SELL/COVER)
- `winning_trades`: Profitable closes
- `losing_trades`: Unprofitable closes
- `win_rate`: Winning trades / Completed trades
- `final_position`: End-of-day position (should be 0)

#### 2. monthly_summary.csv
Aggregated monthly performance:
```csv
year_month,strategy,pnl,pnl_pct,total_trades,completed_trades,winning_trades,losing_trades,win_rate
2024-12,moderate,234.50,1.23,89,45,28,17,0.62
```

#### 3. strategy_comparison.csv
Strategy performance comparison:
```csv
strategy,pnl_sum,pnl_mean,pnl_std,pnl_min,pnl_max,win_rate_mean
moderate,2345.60,11.73,25.40,-45.20,89.30,0.58
high_freq,1890.40,9.45,18.20,-32.10,67.80,0.62
```

**Key Metrics:**
- `pnl_sum`: Total profit/loss for entire period
- `pnl_mean`: Average daily P&L
- `pnl_std`: Standard deviation (volatility)
- `pnl_min/max`: Best/worst single day
- `win_rate_mean`: Average win rate

#### 4. detailed_report.txt
Human-readable summary with:
- Overall statistics
- Year/month/quarter summaries
- Strategy rankings
- Best/worst trading days
- Risk metrics

---

## Examples

### Example 1: Full Year Backtest
```bash
# Backtest entire 2024
python bulk_backtest_runner.py \
    --ticker TSLA \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --strategies moderate \
    --trading-days-only
```

**Expected Output:**
```
================================================================================
🚀 批量回测
================================================================================
   股票: TSLA
   日期范围: 2024-01-01 到 2024-12-31
   交易日数: 252
   策略: moderate
   输出目录: bulk_backtest_results
================================================================================

📊 策略: 温和进取策略
================================================================================
[  0.4%] 2024-01-02 - moderate... ✅ PnL: $+12.50 (+1.25%)
[  0.8%] 2024-01-03 - moderate... ❌ PnL: $-5.30 (-0.53%)
[  1.2%] 2024-01-04 - moderate... ✅ PnL: $+8.70 (+0.87%)
...
[100.0%] 2024-12-31 - moderate... ✅ PnL: $+15.20 (+1.52%)

✅ 每日结果已保存: bulk_backtest_results/daily_results.csv

================================================================================
📊 生成汇总报告
================================================================================
✅ 每日汇总: bulk_backtest_results/daily_summary.csv
✅ 月度汇总: bulk_backtest_results/monthly_summary.csv
✅ 季度汇总: bulk_backtest_results/quarterly_summary.csv
✅ 年度汇总: bulk_backtest_results/yearly_summary.csv
✅ 策略对比: bulk_backtest_results/strategy_comparison.csv
✅ 详细报告: bulk_backtest_results/detailed_report.txt

================================================================================
批量回测详细报告
================================================================================

📋 基本信息
--------------------------------------------------------------------------------
   股票代码: TSLA
   日期范围: 2024-01-02 到 2024-12-31
   交易日数: 252
   策略数量: 1
   初始资金: $1,000.00

📊 年度汇总
--------------------------------------------------------------------------------

2024 - moderate:
   总盈亏: $2,345.60
   平均日收益率: 1.23%
   总交易数: 1,234
   完成交易: 617
   胜率: 58.2%

🏆 策略对比
--------------------------------------------------------------------------------

moderate:
   累计盈亏: $2,345.60
   平均日盈亏: $9.31
   盈亏标准差: $23.45
   最大单日盈利: $89.30
   最大单日亏损: $-45.20
   平均胜率: 58.2%

📈 最佳/最差交易日
--------------------------------------------------------------------------------

moderate:
   最佳: 2024-03-15 - $+89.30 (+8.93%)
   最差: 2024-08-22 - $-45.20 (-4.52%)

================================================================================
```

### Example 2: Strategy Comparison
```bash
# Compare 3 strategies
python bulk_backtest_runner.py \
    --ticker TSLA \
    --start 2024-12-01 \
    --end 2025-12-01 \
    --strategies moderate,high_freq,ultra \
    --trading-days-only
```

**Output:**
```
🏆 策略对比
--------------------------------------------------------------------------------

moderate:
   累计盈亏: $2,345.60
   平均日盈亏: $9.31
   ...

high_freq:
   累计盈亏: $1,890.40
   平均日盈亏: $7.50
   ...

ultra:
   累计盈亏: $3,120.80
   平均日盈亏: $12.38
   ...
```

### Example 3: Quarterly Analysis
```bash
# Just Q4 2024
python bulk_backtest_runner.py \
    --ticker TSLA \
    --start 2024-10-01 \
    --end 2024-12-31 \
    --strategies moderate \
    --output-dir results_q4_2024
```

---

## Reading the Results

### Understanding Daily Results

#### Good Day Example:
```
2024-03-15 - moderate... ✅ PnL: $+89.30 (+8.93%)
```
- Made $89.30 profit
- 8.93% return on capital
- ✅ indicates positive P&L

#### Bad Day Example:
```
2024-08-22 - moderate... ❌ PnL: $-45.20 (-4.52%)
```
- Lost $45.20
- -4.52% return on capital
- ❌ indicates negative P&L

### Understanding Metrics

#### Win Rate
```
Win Rate: 58.2%
```
- 58.2% of completed trades were profitable
- **Good**: > 55%
- **Excellent**: > 60%
- **Poor**: < 50%

#### Sharpe Ratio (Implied from std dev)
```
Average Daily P&L: $9.31
Std Dev: $23.45
→ Sharpe ≈ 9.31 / 23.45 ≈ 0.40 daily
→ Annualized Sharpe ≈ 0.40 × √252 ≈ 6.35
```
- **Good**: > 1.0
- **Excellent**: > 2.0
- **Poor**: < 0.5

#### Maximum Drawdown
From drawdown_analysis.html chart:
- **Good**: < 10%
- **Acceptable**: 10-20%
- **High Risk**: > 20%

---

## Visualizations

### 1. Cumulative Returns Chart
**File**: `cumulative_returns.html`

Shows equity growth over time for each strategy.

**How to Read:**
- Y-axis: Cumulative profit/loss ($)
- X-axis: Date
- Multiple lines: One per strategy
- **Look for**: Smooth upward trend

### 2. Monthly Returns Bar Chart
**File**: `monthly_returns.html`

Compare monthly performance across strategies.

**How to Read:**
- Grouped bars by month
- Each color = one strategy
- **Look for**: Consistent positive months

### 3. Strategy Comparison Radar
**File**: `strategy_comparison.html`

Multi-dimensional strategy comparison.

**Dimensions:**
- Total Return
- Average Return
- Win Rate
- Sharpe Ratio
- Profitable Days %

**How to Read:**
- Larger area = better overall
- Look for balanced shape

### 4. Drawdown Analysis
**File**: `drawdown_analysis.html`

Shows equity curve and drawdown over time.

**Top Panel**: Equity curve
**Bottom Panel**: Drawdown (negative values)

**How to Read:**
- Deep valleys = large drawdowns
- Long valleys = slow recovery
- **Good**: Small, quick recoveries

### 5. Win/Loss Distribution
**File**: `win_loss_distribution.html`

Histogram of daily P&L distribution.

**How to Read:**
- Bell curve shape is ideal
- Center (mean) should be positive
- Fat right tail = large wins
- Thin left tail = limited losses

---

## Performance Benchmarks

### Excellent Performance
```
Annual Return: > 100%
Win Rate: > 60%
Sharpe Ratio: > 2.0
Max Drawdown: < 10%
Profitable Days: > 60%
```

### Good Performance
```
Annual Return: 50-100%
Win Rate: 55-60%
Sharpe Ratio: 1.0-2.0
Max Drawdown: 10-15%
Profitable Days: 55-60%
```

### Acceptable Performance
```
Annual Return: 20-50%
Win Rate: 50-55%
Sharpe Ratio: 0.5-1.0
Max Drawdown: 15-20%
Profitable Days: 50-55%
```

### Poor Performance
```
Annual Return: < 20%
Win Rate: < 50%
Sharpe Ratio: < 0.5
Max Drawdown: > 20%
Profitable Days: < 50%
```

---

## Tips & Best Practices

### 1. Start Small
```bash
# Test 1 month first
python bulk_backtest_runner.py --start 2024-12-01 --end 2024-12-31
```

### 2. Use Trading Days Only
```bash
# Skip weekends (market closed)
--trading-days-only
```

### 3. Compare Multiple Strategies
```bash
# Find best strategy
--strategies moderate,high_freq,ultra
```

### 4. Check Data Quality
Look for:
- "⚠️ 跳过（无数据或错误）" warnings
- If many days skipped → data issues

### 5. Analyze Risk-Adjusted Returns
Don't just look at total return:
- High return + high volatility = risky
- Moderate return + low volatility = better

### 6. Watch for Overfitting
If backtest results are "too good":
- Win rate > 80% → suspicious
- No losing months → suspicious
- Max drawdown < 2% → suspicious

### 7. Consider Transaction Costs
Results include:
- Commission: 0.03% per trade
- Slippage: 0.01% per trade
- Total: ~0.04% per trade

High-frequency strategies pay more fees!

---

## Troubleshooting

### Problem: No Data for Many Days
```
⚠️ 跳过（无数据或错误）
```

**Solutions:**
1. Check if dates are trading days (not holidays)
2. Verify Alpaca API access
3. Try a different date range
4. Check market hours (9:30-16:00 ET)

### Problem: All Strategies Losing Money
```
Total P&L: $-500.00 (all strategies)
```

**Possible Causes:**
1. Market trend doesn't match strategy
2. Wrong parameters for this stock
3. High volatility period
4. Data quality issues

**Solutions:**
- Try different time period
- Adjust strategy parameters
- Test on different stock

### Problem: Very Slow Performance
```
[5 minutes per day]
```

**Solutions:**
1. Reduce LOOKBACK_MINUTES (default: 300)
2. Increase STEP_MINUTES (default: 1)
3. Use fewer strategies at once
4. Run in parallel (advanced)

### Problem: Out of Memory
```
MemoryError
```

**Solutions:**
1. Reduce date range (do quarters instead of year)
2. Run strategies separately
3. Increase system RAM

---

## Advanced Usage

### Custom Date Ranges

**Month:**
```bash
--start 2024-12-01 --end 2024-12-31
```

**Quarter:**
```bash
--start 2024-10-01 --end 2024-12-31  # Q4
```

**Year:**
```bash
--start 2024-01-01 --end 2024-12-31
```

**Multi-Year:**
```bash
--start 2023-01-01 --end 2024-12-31
```

### Parallel Execution (Advanced)

Run multiple backtests in parallel:
```bash
# Terminal 1
python bulk_backtest_runner.py --strategies moderate --output-dir results_moderate &

# Terminal 2
python bulk_backtest_runner.py --strategies high_freq --output-dir results_high_freq &

# Terminal 3
python bulk_backtest_runner.py --strategies ultra --output-dir results_ultra &
```

### Export to Excel
```python
import pandas as pd

# Read CSV
df = pd.read_csv('bulk_backtest_results/daily_results.csv')

# Export to Excel with multiple sheets
with pd.ExcelWriter('results.xlsx') as writer:
    df.to_excel(writer, sheet_name='Daily')
    df.groupby('strategy').sum().to_excel(writer, sheet_name='Summary')
```

---

## Integration with Other Tools

### Export for Further Analysis
```python
import pandas as pd

# Load results
df = pd.read_csv('bulk_backtest_results/daily_results.csv')

# Calculate custom metrics
df['cumulative_return'] = (1 + df['pnl_pct']/100).cumprod() - 1
df['sharpe_ratio'] = df.groupby('strategy')['pnl'].transform(
    lambda x: x.mean() / x.std() * np.sqrt(252)
)

# Save
df.to_csv('enhanced_results.csv')
```

### Send Results via Email
```python
import smtplib
from email.mime.text import MIMEText

# Read report
with open('bulk_backtest_results/detailed_report.txt') as f:
    report = f.read()

# Send email
msg = MIMEText(report)
msg['Subject'] = 'Backtest Results'
msg['From'] = 'your@email.com'
msg['To'] = 'recipient@email.com'

# ... send via SMTP
```

---

## Summary

### Key Steps
1. Run bulk backtest: `python bulk_backtest_runner.py`
2. Generate charts: `python bulk_backtest_visualizer.py`
3. Review results in output directory
4. Compare strategies
5. Analyze risk metrics
6. Iterate and improve

### Most Important Files
- `detailed_report.txt` - Start here!
- `cumulative_returns.html` - Visual overview
- `strategy_comparison.csv` - Choose best strategy

### Next Steps
After bulk backtest:
1. Choose best strategy
2. Optimize parameters
3. Test on different stocks
4. Paper trade before live