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

1. 核心模块化：引入仓位管理器（Position Manager）

目标： 实现账户资金和持仓的集中管理，将复杂的财务和风险计算从执行器中分离出来。

新增模块： PositionManager 类。

职责：

维护当前账户的净值 (Equity)、可用资金 (Cash)、总仓位 (Total Position)。

记录和计算每笔交易的利润/亏损 (PnL)。

封装账户状态和风险控制相关的逻辑。

交互升级： Executor (执行器) 将不再直接维护仓位状态，而是调用 PositionManager 来更新和查询账户信息。

2. 策略升级：独立策略层与指标计算解耦

目标： 建立一个独立的 Strategy 基类，允许更容易地添加新策略，并将技术指标的计算逻辑从数据获取器中迁移出来。

策略基类： Strategy。

具体策略： MeanReversionStrategy。

职责迁移：

将 布林带 (Bollinger Bands) 和 RSI 的计算逻辑从 alpaca_data_fetcher.py 迁移到 MeanReversionStrategy 中。

DataFetcher 的职责将简化为仅获取原始 K 线数据。

Strategy 接收原始数据，计算所需指标，然后将指标结果传递给 Decision Maker。

3. 决策层分离：引入决策制定器（Decision Maker）

目标： 将策略的“决策”环节抽象成一个可插拔的模块，以支持不同的决策模型（如 LLM、传统数学模型）。

新增模块： DecisionMaker 基类。

具体实现：

GeminiDecisionMaker： 当前使用的 LLM 决策逻辑，接收指标数据，调用 Gemini API 输出交易信号。

TraditionalDecisionMaker： 基于纯数学或规则的决策器（例如：收盘价低于下轨线，RSI < 30 即 BUY）。

交互： Strategy 将调用 DecisionMaker.decide(df_with_indicators) 来获取最终的 BUY/SELL/HOLD 信号。

4. 长期规划：结合 LLM 深度分析 (Wall Street News)

目标： 利用 Gemini 模型的搜索和总结能力，纳入基本面和市场情绪分析，实现多模态决策。

功能实现：

在 Strategy 层引入新的情绪分析模块。

利用 Gemini 的 Google Search Grounding 能力，实时抓取“华尔街新闻”等金融资讯。

LLM 对新闻内容进行**利好（Bullish）/利空（Bearish）**判断和总结。

将情绪得分作为权重或乘数，调整最终的交易信号，增强策略对突发事件的抗风险能力。