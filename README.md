量化交易 LLM 辅助系统 (LLM-Trader)

本项目旨在构建一个利用大型语言模型（LLM，特别是 Gemini）进行辅助决策的量化交易系统。系统专注于通过技术分析（如布林带）结合 LLM 的模式识别能力，自动生成交易信号，并在模拟或实盘环境中执行交易。

🏛️ 当前系统架构概览 (MVP)

当前的系统主要围绕数据获取、策略决策和回测执行三个核心部分构建：

组件名称

路径

核心功能

Data Fetcher

src/data/alpaca_data_fetcher.py

从 Alpaca 获取 OHLCV 历史 K 线数据，并计算布林带 (BB) 和 RSI 等技术指标。

Strategy

src/strategies/mean_reversion_strategy.py

负责策略逻辑。目前主要通过调用 Gemini LLM 分析指标数据，输出 BUY/SELL/HOLD 信号。

Backtest Runner

backtest_runner.py

驱动回测或实时运行，循环调用策略并使用 SimulationExecutor 或 AlpacaExecutor 模拟或执行交易。

💡 后续改进与架构升级规划

为了使系统更加健壮、功能更完善，您的后续改进想法将围绕职责分离、模块化和决策智能化进行。



1. data fetcher, 加入一个基本类，然后实现不同api的data fetch，比如可以用moomoo api 来data fetch.
data fetcher 要能获取仓位信息。live runner 的position manager要通过data fetcher来进行仓位信息获取，而不是自定义。

2. exectuor也要加入moomoo api的executor. 

3. 更激进的交易规则：突破上轨做空，回到中线平空，跌破下轨做多，回到中线平多
高频监控：每1分钟检查一次（可配置）
5分钟K线：技术指标计算仍用5分钟线
止损机制：单笔持仓亏损10%时自动平仓（可配置）

4. add a list of ticker to monitor on. 

5. moderate strategy looks better than trend aware strategy with MSFT. optimize the moderate strategy by : 1. add stop loss, when loss greater than 1%, sell/cover, reduce 仓位to 50%. 2. when equity loss greater than 2%,  stop a side action (e.g. buy or short).
当布林带宽度很窄的时候不交易。

potential stock:
F, NVDA, TSLA, MSCI, SPLV, F, MSFT

5. 长期规划：结合 LLM 深度分析 (Wall Street News)

目标： 利用 Gemini 模型的搜索和总结能力，纳入基本面和市场情绪分析，实现多模态决策。

功能实现：

在 Strategy 层引入新的情绪分析模块。

利用 Gemini 的 Google Search Grounding 能力，实时抓取“华尔街新闻”等金融资讯。

LLM 对新闻内容进行**利好（Bullish）/利空（Bearish）**判断和总结。

将情绪得分作为权重或乘数，调整最终的交易信号，增强策略对突发事件的抗风险能力。



```
python -m src.runner.backtest_runner
python -m src.runner.live_runner                                                    
```