source llm-trade/bin/activate


目前用的比较好的runner:
backtest_with_chart_multi_strategy

python -m src.runner.backtest_with_chart_multi_strategy --strategy moderate --ticker TSLA --date 2025-11-05


python -m src.runner.live_runner_with_chart_multi_strategy --strategy moderate --ticker TSLA --mode paper



# 1-year backtest with moderate strategy
python -m src.runner.bulk_backtest_runner \
    --ticker TSLA \
    --start 2025-11-01 \
    --end 2025-12-01 \
    --strategies moderate \
    --trading-days-only


# 🚀 多进程交易系统使用指南

## 📋 目录结构

运行多进程交易后，会自动创建以下目录结构：

```
live_trading/
├── pids/                    # 进程 ID 文件
│   ├── TSLA_moderate_paper.pid
│   ├── AAPL_moderate_paper.pid
│   └── NVDA_moderate_paper.pid
├── logs/                    # 独立日志文件
│   ├── TSLA_moderate_paper.log
│   ├── AAPL_moderate_paper.log
│   └── NVDA_moderate_paper.log
├── cache/                   # 独立缓存文件
│   ├── TSLA_moderate_paper_cache.json
│   ├── AAPL_moderate_paper_cache.json
│   └── NVDA_moderate_paper_cache.json
└── charts/                  # 独立图表文件
    ├── TSLA_moderate_paper.html
    ├── AAPL_moderate_paper.html
    └── NVDA_moderate_paper.html
```

## 🎯 快速开始

### 方法 1: 使用管理脚本（推荐）

```bash
# 1. 同时启动多个股票的交易
python multi_process_manager.py start \
    --tickers TSLA AAPL NVDA GOOGL MSFT \
    --strategy moderate \
    --mode paper

# 2. 查看运行状态
python multi_process_manager.py status

# 3. 查看特定股票的实时日志
python multi_process_manager.py logs --ticker TSLA --follow

# 4. 打开图表
python multi_process_manager.py chart --ticker TSLA

# 5. 停止特定股票
python multi_process_manager.py stop --ticker TSLA

# 6. 停止所有进程
python multi_process_manager.py stop --all
```

### 方法 2: 手动启动（更灵活）

```bash
# 启动 TSLA - 温和进取策略
python live_runner.py --strategy moderate --ticker TSLA --mode paper &

# 启动 AAPL - 动态策略
python live_runner.py --strategy moderate_dynamic --ticker AAPL --mode paper &

# 启动 NVDA - 高频策略
python live_runner.py --strategy high_freq --ticker NVDA --mode paper &

# 启动 GOOGL - 保守策略（禁用图表以节省资源）
python live_runner.py --strategy conservative --ticker GOOGL --mode paper --no-chart &

# 查看所有后台进程
jobs

# 查看运行状态
python multi_process_manager.py status
```

## 📊 使用示例

### 示例 1: 同时交易 5 支科技股

```bash
# 启动
python multi_process_manager.py start \
    --tickers TSLA AAPL NVDA GOOGL MSFT \
    --strategy moderate \
    --mode paper \
    --interval 60

# 输出:
# 🚀 启动 5 个交易进程...
#    策略: moderate
#    模式: paper
#    间隔: 60 秒
#
# 🚀 启动 TSLA (moderate, paper)...
#    ✅ TSLA 启动成功 (PID: 12345)
# 🚀 启动 AAPL (moderate, paper)...
#    ✅ AAPL 启动成功 (PID: 12346)
# ...
# ✅ 成功启动 5/5 个进程
```

### 示例 2: 查看运行状态

```bash
python multi_process_manager.py status

# 输出:
# 📊 运行中的交易进程 (共 5 个):
# ================================================================================
# Ticker   Strategy             Mode         PID      日志文件
# --------------------------------------------------------------------------------
# TSLA     moderate             paper        12345    ✅
# AAPL     moderate             paper        12346    ✅
# NVDA     moderate             paper        12347    ✅
# GOOGL    moderate             paper        12348    ✅
# MSFT     moderate             paper        12349    ✅
# ================================================================================
```

### 示例 3: 实时监控日志

```bash
# 实时查看 TSLA 的交易日志
python multi_process_manager.py logs --ticker TSLA --follow

# 输出示例:
# [TSLA] 2024-12-10 09:35:00 - INFO - 📊 [2024-12-10 09:35:00 EST] 状态更新
# [TSLA] 2024-12-10 09:35:00 - INFO -    TSLA 价格: $369.04
# [TSLA] 2024-12-10 09:35:00 - INFO -    账户权益: $100,000.00
# [TSLA] 2024-12-10 09:35:30 - INFO - 🟢 交易信号!
# [TSLA] 2024-12-10 09:35:30 - INFO -    信号: BUY | 价格: $369.04 | 置信度: 8/10
# [TSLA] 2024-12-10 09:35:31 - INFO -    💱 [09:35:31] 🟢 买入开多 TSLA: 250 股
# [TSLA] 2024-12-10 09:35:31 - INFO -       订单ID: 12345678-abcd-...
# [TSLA] 2024-12-10 09:35:31 - INFO -    ✅ 交易执行成功
```

### 示例 4: 停止特定进程

```bash
# 只停止 TSLA
python multi_process_manager.py stop --ticker TSLA

# 输出:
# ⏹️  停止 TSLA (PID: 12345)...
#    ✅ TSLA 已停止
# ✅ 已停止 1 个进程
```

### 示例 5: 不同策略同时运行

```bash
# TSLA 使用温和进取策略
python live_runner.py --strategy moderate --ticker TSLA --mode paper &

# AAPL 使用动态策略
python live_runner.py --strategy moderate_dynamic --ticker AAPL --mode paper &

# NVDA 使用高频策略（更快的更新间隔）
python live_runner.py --strategy high_freq --ticker NVDA --mode paper --interval 30 &
```

## 🔒 多进程安全特性

### 1. 独立的资源文件

每个进程使用完全独立的文件：

```
进程标识: {TICKER}_{STRATEGY}_{MODE}

示例:
- TSLA_moderate_paper
- AAPL_moderate_paper
- NVDA_high_freq_paper
```

### 2. 防止重复启动

```bash
# 尝试重复启动会被阻止
python live_runner.py --strategy moderate --ticker TSLA --mode paper

# 输出:
# ⚠️ 错误: TSLA (策略: moderate, 模式: paper) 已经在运行！
#    PID 文件: live_trading/pids/TSLA_moderate_paper.pid
#    如需强制启动，请使用 --force 参数

# 强制启动（会终止旧进程）
python live_runner.py --strategy moderate --ticker TSLA --mode paper --force
```

### 3. 独立的日志输出

每个进程的日志都带有 ticker 标识，不会混淆：

```
[TSLA] 2024-12-10 09:35:00 - INFO - 交易信号: BUY @ $369.04
[AAPL] 2024-12-10 09:35:05 - INFO - 交易信号: SHORT @ $195.21
[NVDA] 2024-12-10 09:35:10 - INFO - 交易信号: SELL @ $485.76
```

### 4. 独立的图表文件

每个 ticker 生成独立的图表，可以同时在多个浏览器标签页中打开：

```
- TSLA_moderate_paper.html
- AAPL_moderate_paper.html
- NVDA_moderate_paper.html
```

## 🎛️ 高级用法

### 1. 混合交易模式

```bash
# TSLA 使用实盘（谨慎！）
python live_runner.py --strategy moderate --ticker TSLA --mode live &

# AAPL 使用模拟盘
python live_runner.py --strategy moderate --ticker AAPL --mode paper &

# NVDA 使用本地模拟
python live_runner.py --strategy moderate --ticker NVDA --mode simulation &
```

### 2. 自定义更新频率

```bash
# 高频交易（30秒）
python live_runner.py --strategy high_freq --ticker TSLA --interval 30 &

# 日内交易（5分钟）
python live_runner.py --strategy moderate --ticker AAPL --interval 300 &
```

### 3. 禁用图表以节省资源

```bash
# 交易 20 支股票，只为前 5 支生成图表
python multi_process_manager.py start \
    --tickers TSLA AAPL NVDA GOOGL MSFT \
    --strategy moderate \
    --mode paper

# 剩余 15 支禁用图表
python multi_process_manager.py start \
    --tickers AMD INTC QCOM AVGO NFLX DIS BABA AMZN META NFLX PFE JNJ UNH CVS WMT \
    --strategy moderate \
    --mode paper \
    --no-chart
```

## 📈 监控和管理

### 查看系统资源

```bash
# 查看所有 Python 进程
ps aux | grep live_runner.py

# 查看特定 ticker 的进程
ps aux | grep "ticker TSLA"

# 查看资源占用（CPU, 内存）
top -p $(cat live_trading/pids/*.pid | tr '\n' ',')
```

### 批量操作

```bash
# 停止所有进程
python multi_process_manager.py stop --all

# 重启所有进程（先停止再启动）
python multi_process_manager.py stop --all
sleep 2
python multi_process_manager.py start --tickers TSLA AAPL NVDA --strategy moderate --mode paper
```

### 日志聚合查看

```bash
# 查看所有日志的最新交易信号
grep "交易信号" live_trading/logs/*.log | tail -20

# 查看所有成功的交易
grep "交易执行成功" live_trading/logs/*.log

# 统计每个 ticker 的交易次数
grep "交易执行成功" live_trading/logs/*.log | cut -d'[' -f2 | cut -d']' -f1 | sort | uniq -c
```

## 🛠️ 故障排查

### 问题 1: 进程启动失败

```bash
# 检查日志
python multi_process_manager.py logs --ticker TSLA --lines 100

# 手动清理 PID 文件
rm live_trading/pids/TSLA_moderate_paper.pid

# 重新启动
python live_runner.py --strategy moderate --ticker TSLA --mode paper
```

### 问题 2: 进程僵尸

```bash
# 查找僵尸进程
ps aux | grep live_runner.py | grep defunct

# 强制杀死
kill -9 $(cat live_trading/pids/*.pid)
rm live_trading/pids/*.pid
```

### 问题 3: 日志文件过大

```bash
# 压缩旧日志
gzip live_trading/logs/*.log

# 或删除旧日志
rm live_trading/logs/*.log
```

## 🎯 最佳实践

### 1. 分批启动

```bash
# 避免同时启动过多进程，分批启动
python multi_process_manager.py start --tickers TSLA AAPL NVDA --strategy moderate --mode paper
sleep 10
python multi_process_manager.py start --tickers GOOGL MSFT AMD --strategy moderate --mode paper
```

### 2. 使用 nohup 保持后台运行

```bash
# 即使关闭终端也继续运行
nohup python live_runner.py --strategy moderate --ticker TSLA --mode paper > /dev/null 2>&1 &
```

### 3. 使用 screen 或 tmux

```bash
# 创建独立的会话
screen -S trading_tsla
python live_runner.py --strategy moderate --ticker TSLA --mode paper
# Ctrl+A, D 分离会话

# 重新连接
screen -r trading_tsla
```

### 4. 定期监控

```bash
# 创建监控脚本
cat > monitor.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    python multi_process_manager.py status
    sleep 60
done
EOF

chmod +x monitor.sh
./monitor.sh
```

## ⚠️ 注意事项

1. **资源限制**: 每个进程占用一定的 CPU 和内存，建议不要同时运行超过 20 个进程
2. **API 限流**: Alpaca API 有速率限制，过多进程可能触发限流
3. **实盘交易**: 使用 `--mode live` 前请务必充分测试
4. **数据延迟**: IEX 数据有 15 分钟延迟（免费版）
5. **市场时间**: 默认只在美股交易时间运行（9:30-16:00 ET）

## 📚 相关命令参考

```bash
# live_runner.py 参数
--strategy      # 策略: moderate, moderate_dynamic, high_freq, ultra, conservative
--ticker        # 股票代码（必填）
--mode          # 模式: paper, live, simulation
--interval      # 更新间隔（秒）
--no-chart      # 禁用图表
--force         # 强制启动

# multi_process_manager.py 命令
start           # 启动进程
status          # 查看状态
stop            # 停止进程
logs            # 查看日志
chart           # 打开图表
```