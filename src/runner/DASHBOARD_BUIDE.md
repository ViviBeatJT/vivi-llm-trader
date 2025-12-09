# 🌐 Real-Time Trading Dashboard - 完整指南

## 🎉 你现在拥有的功能

✅ **Web 实时仪表板**（浏览器访问）  
✅ **回测时实时可视化**（观看策略运行）  
✅ **实盘时实时监控**（观察真实交易）  
✅ **自动刷新图表**（无需手动刷新）  
✅ **4个关键图表**：价格+布林带、成交量、权益曲线、持仓状态  

---

## 📦 需要安装的包

### 1. 安装依赖

```bash
pip install plotly dash pandas numpy
```

**包说明**：
- `plotly` - 交互式图表库
- `dash` - Web 仪表板框架
- `pandas`, `numpy` - 已有的依赖

### 2. 验证安装

```bash
python -c "import plotly, dash; print('✅ 安装成功!')"
```

---

## 🚀 快速开始

### 方式 1: 回测 + 仪表板

```bash
# 1. 复制文件
cp live_trading_dashboard.py src/visualization/
cp backtest_with_dashboard.py src/runner/

# 2. 运行
cd src/runner
python backtest_with_dashboard.py
```

**会发生什么**：
1. 自动打开浏览器 → `http://localhost:8050`
2. 回测开始运行
3. 图表实时更新
4. 看到每笔交易出现在图上
5. 权益曲线实时绘制

### 方式 2: 实盘 + 仪表板

```bash
# 1. 复制文件
cp live_trading_dashboard.py src/visualization/
cp live_with_dashboard.py src/runner/

# 2. 配置
# 编辑 live_with_dashboard.py
TRADING_MODE = 'paper'  # 先用模拟盘

# 3. 运行
python live_with_dashboard.py
```

---

## 📊 仪表板界面预览

```
┌─────────────────────────────────────────────────────────┐
│  🚀 Real-Time Trading Dashboard - TSLA                  │
├─────────────────────────────────────────────────────────┤
│  Total Trades │ Position  │ Equity    │ P&L    │ Time  │
│      12       │ 50 (Long) │ $102,450  │ +$2,450│ 14:32 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  📈 Price & Bollinger Bands                             │
│  ┌──────────────────────────────────────────┐          │
│  │  🟢 Buy/Short    🔴 Sell/Cover           │          │
│  │  ━━━ Close Price  ┄┄┄ BB Upper/Lower    │          │
│  │                                           │          │
│  │       🟢                   🔴             │          │
│  │      ╱                    ╱╲              │          │
│  │    ━━━━━━━━━━━━━━━━━━━━━━━━━             │          │
│  │  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄                │          │
│  └──────────────────────────────────────────┘          │
│                                                          │
│  📊 Volume                                              │
│  ┌──────────────────────────────────────────┐          │
│  │  ▌▌ ▌ ▌▌▌  ▌ ▌▌▌▌                       │          │
│  └──────────────────────────────────────────┘          │
│                                                          │
│  💰 Equity Curve                                        │
│  ┌──────────────────────────────────────────┐          │
│  │      ╱╲    ╱╲                            │          │
│  │    ╱    ╲╱  ╲                            │          │
│  │  ━━━━━━━━━━━━━━━━  (Initial: $100k)     │          │
│  └──────────────────────────────────────────┘          │
│                                                          │
│  📍 Position Status                                     │
│  ┌──────────────────────────────────────────┐          │
│  │  ▌ Long  ▌ Flat ▌ Short ▌ Long          │          │
│  │  ─────────────────────────────────────   │          │
│  └──────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────┘
```

---

## ⚙️ 配置选项

### 回测仪表板配置

在 `backtest_with_dashboard.py` 中：

```python
# 股票和日期
TICKER = "TSLA"
TRADING_DATE = "2024-12-05"

# 回测速度
STEP_MINUTES = 1  # 每1分钟检查一次

# 仪表板
DASHBOARD_PORT = 8050         # 端口号
AUTO_OPEN_BROWSER = True      # 自动打开浏览器
```

### 实盘仪表板配置

在 `live_with_dashboard.py` 中：

```python
# 交易模式
TRADING_MODE = 'paper'  # 'paper' / 'live' / 'simulation'

# 监控频率
MONITOR_INTERVAL_SECONDS = 60  # 每60秒检查

# 仪表板刷新
update_interval=1000  # 1秒刷新一次图表
```

---

## 🎯 使用场景

### 场景 1: 回测参数调优

**目标**: 找到最优的止损阈值

```python
# 运行多次，每次调整参数
STOP_LOSS_THRESHOLD = 0.08  # 第1次
STOP_LOSS_THRESHOLD = 0.10  # 第2次
STOP_LOSS_THRESHOLD = 0.12  # 第3次

# 观察图表，看哪个表现最好
```

**看什么**：
- 止损触发频率
- 权益曲线平滑度
- 最大回撤

### 场景 2: 策略行为验证

**目标**: 确认策略按预期运行

```python
# 观察图表
✅ 价格突破上轨时，出现 🔻 SHORT 标记
✅ 价格回到中线时，出现 🔺 COVER 标记
✅ 权益曲线符合预期
```

### 场景 3: 实盘监控

**目标**: 实时监控交易状态

```python
# 运行实盘
python live_with_dashboard.py

# 在另一个屏幕打开浏览器
# 一边工作，一边看图表
# 发现异常可立即 Ctrl+C 停止
```

---

## 🔍 调试技巧

### 问题 1: 图表不更新

**检查**：
```python
# 确保策略正在累积数据
strategy_df = strategy.get_history_data(TICKER)
print(f"历史数据: {len(strategy_df)} 条")

# 应该看到数据逐渐增加
```

**解决**：
```python
# 确保策略被正确调用
signal_data, price = strategy.get_signal(...)

# 确保数据被传递给仪表板
dashboard.update_market_data(strategy_df)
```

### 问题 2: 浏览器打不开

**手动打开**：
```
http://localhost:8050
```

**换端口**：
```python
DASHBOARD_PORT = 8051  # 改成其他端口
```

### 问题 3: 图表显示不全

**原因**: 数据不足

**解决**：
```python
# 增加回溯时间
LOOKBACK_MINUTES = 200  # 从 120 改为 200

# 或等待更多数据累积
```

---

## 📈 高级功能

### 功能 1: 自定义刷新间隔

```python
# 更快刷新（适合回测）
dashboard = TradingDashboard(
    update_interval=500  # 500ms = 0.5秒
)

# 更慢刷新（适合实盘，节省资源）
dashboard = TradingDashboard(
    update_interval=2000  # 2秒
)
```

### 功能 2: 多股票监控

```python
# 为每个股票创建仪表板
dashboard_tsla = TradingDashboard(ticker='TSLA', port=8050)
dashboard_aapl = TradingDashboard(ticker='AAPL', port=8051)

# 在浏览器打开多个标签页
```

### 功能 3: 保存图表快照

```python
# 在 TradingDashboard 中添加方法
def save_snapshot(self, filename):
    """保存当前图表为 PNG"""
    fig = self._create_figure()
    fig.write_image(f"{filename}.png")
```

---

## ⚠️ 注意事项

### 性能考虑

```python
# ❌ 不推荐：太快的刷新
update_interval=100  # 0.1秒，浏览器可能卡顿

# ✅ 推荐：合理的刷新间隔
update_interval=1000  # 1秒，平滑流畅
```

### 内存管理

```python
# 回测时数据会累积
# 如果回测很长时间（多天），可能占用大量内存

# 解决：分段回测
for date in date_list:
    run_backtest(date)
    dashboard.reset()  # 清空历史数据
```

### 浏览器兼容性

**推荐浏览器**：
- ✅ Chrome / Edge（最佳）
- ✅ Firefox（良好）
- ⚠️ Safari（基本可用）

---

## 🎓 学习路径

### 第1天: 基础使用
1. 安装依赖
2. 运行示例回测
3. 观察图表变化

### 第2天: 调试策略
1. 修改策略参数
2. 观察行为变化
3. 找到最优配置

### 第3天: 实盘准备
1. 用仪表板回测
2. 验证策略逻辑
3. 切换到模拟盘

### 第4天: 实盘运行
1. 启动实盘 + 仪表板
2. 实时监控
3. 根据表现调整

---

## 🆘 常见问题

### Q1: 端口被占用

**错误**: `Address already in use`

**解决**:
```bash
# 杀死占用端口的进程
# Mac/Linux:
lsof -ti:8050 | xargs kill -9

# Windows:
netstat -ano | findstr :8050
taskkill /PID <PID> /F

# 或改用其他端口
DASHBOARD_PORT = 8051
```

### Q2: 图表很慢

**原因**: 数据太多或刷新太快

**解决**:
```python
# 降低刷新频率
update_interval=2000  # 从 1秒 改为 2秒

# 限制显示的数据点
# 只显示最近 100 条K线
dashboard.update_market_data(strategy_df.tail(100))
```

### Q3: Ctrl+C 无法停止

**原因**: Dash 服务器在后台运行

**解决**:
```bash
# 强制退出
# Mac/Linux:
pkill -f "python.*dashboard"

# Windows:
# 关闭终端窗口
```

---

## 📚 进一步定制

想要添加更多功能？例如：

### 添加 RSI 图表

```python
# 在 _create_figure 中添加
if 'RSI' in df.columns:
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name='RSI'),
        row=5, col=1  # 新增第5个子图
    )
```

### 添加警报通知

```python
# 在 update_position 中添加
if abs(position) > 100:
    send_notification("持仓超过100股！")
```

### 导出数据

```python
# 添加按钮导出 CSV
dashboard.trade_log_to_csv('trades.csv')
```

---

## 🎊 总结

你现在有：

✅ **完整的可视化系统**  
✅ **回测 + 实盘都支持**  
✅ **Web 界面，专业美观**  
✅ **实时更新，无需刷新**  
✅ **帮助调试和决策**  

**下一步**: 运行你的第一个带仪表板的回测！ 🚀

```bash
python backtest_with_dashboard.py
```

---

有问题随时问我！📊💡