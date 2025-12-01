# src/executor/alpaca_trade_executor.py

import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# 导入策略和数据模块
from src.strategies.mean_reversion_strategy import get_mean_reversion_signal
from src.data.alpaca_data_fetcher import get_latest_bars # 用于获取最新价格

# --- 配置 ---
load_dotenv()
# 自动从环境变量中读取 KEY_ID 和 SECRET_KEY
# paper=True 表示使用模拟交易账户
trading_client = TradingClient(os.getenv('ALPACA_API_KEY_ID'), os.getenv('ALPACA_SECRET_KEY'), paper=True)

# --- 交易参数 ---
# 我们在每次交易中冒的风险金额 (用于确定购买数量)
RISK_AMOUNT_USD = 100 
# Limit Order 的价格容差：在市场价基础上便宜 0.05 USD 来挂单
LIMIT_TOLERANCE_USD = 0.05 

def get_current_price(ticker: str) -> float:
    """获取标的物的最新收盘价。"""
    # 临时获取最新的 K 线数据，用于确定当前价格
    kline_data_text = get_latest_bars(ticker=ticker, lookback_minutes=5)
    
    if "没有找到可用的" in kline_data_text or kline_data_text == "没有找到可用的 K 线数据。":
        raise ValueError(f"无法获取 {ticker} 的最新价格。")
        
    # 由于 get_latest_bars 返回 Markdown 文本，我们在这里重新获取 DataFrame
    # 这是一个简化处理，实际应用中应该优化数据流
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    
    data_client = StockHistoricalDataClient(os.getenv('ALPACA_API_KEY_ID'), os.getenv('ALPACA_SECRET_KEY'))
    request_params = StockBarsRequest(
        symbol_or_symbols=[ticker],
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        limit=1
    )
    bar_set = data_client.get_stock_bars(request_params)
    
    if bar_set.df.empty:
        raise ValueError(f"无法获取 {ticker} 的最新 K 线数据。")

    latest_bar = bar_set.df.loc[ticker].iloc[-1]
    return latest_bar['close']

def calculate_order_qty(latest_price: float, usd_amount: int) -> int:
    """根据风险金额计算购买数量 (向下取整)。"""
    if latest_price <= 0:
        return 0
    return int(usd_amount / latest_price)

def execute_trading_signal(ticker: str = "TSLA"):
    """
    运行策略，根据 LLM 信号执行模拟交易订单。
    """
    print(f"\n--- 1. 运行 LLM 策略获取信号 ({ticker}) ---")
    signal_result = get_mean_reversion_signal(ticker=ticker)
    
    signal = signal_result.get('signal')
    confidence = signal_result.get('confidence_score', 5)
    reason = signal_result.get('reason', 'N/A')

    print(f"🧠 LLM 信号: {signal} | 置信度: {confidence}/10")
    print(f"   原因: {reason}")
    print("-" * 40)

    if signal == "HOLD" or confidence < 6:
        print("⏸️ 信号为 HOLD 或置信度过低 (低于 6)，跳过交易。")
        return

    try:
        # 2. 获取最新价格
        latest_price = get_current_price(ticker)
        print(f"💰 最新市场价: ${latest_price:.2f}")

        # 3. 计算订单数量
        qty = calculate_order_qty(latest_price, RISK_AMOUNT_USD)
        
        if qty == 0:
            print(f"🛑 风险金额 {RISK_AMOUNT_USD} USD 不足以购买至少 1 股 {ticker}。")
            return

        # 4. 确定订单方向和价格
        if signal == "BUY":
            order_side = OrderSide.BUY
            # Mean Reversion 买入：价格会低于或接近市场价
            limit_price = round(latest_price - LIMIT_TOLERANCE_USD, 2)
            print(f"⬆️ 准备买入 {qty} 股，挂单价格 (Limit Price): ${limit_price:.2f}")

        elif signal == "SELL":
            order_side = OrderSide.SELL
            # Mean Reversion 卖出（平仓）：使用 Market Order 快速成交
            limit_price = None # 使用市价单 (Market Order)
            print(f"⬇️ 准备卖出 {qty} 股，使用市价单 (Market Order)。")
        
        else:
            print(f"无效信号: {signal}，跳过。")
            return
        
        # 5. 提交订单 (Limit Order)
        if order_side == OrderSide.BUY:
            order_request = LimitOrderRequest(
                symbol=ticker,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.GTC, # Good Til Canceled (订单有效直到被取消)
                limit_price=limit_price 
            )
            trading_client.submit_order(order_request)
            print(f"✅ 成功提交 限价买入订单 (Limit Order)！数量: {qty} @ ${limit_price:.2f}")
            
        # 5. 提交订单 (Market Order for Selling/Exiting)
        elif order_side == OrderSide.SELL:
            order_request = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY 
            )
            trading_client.submit_order(order_request)
            print(f"✅ 成功提交 市价卖出订单 (Market Order)！数量: {qty}")
            
    except Exception as e:
        print(f"❌ 交易执行失败: {e}")
        # 如果是 SELL 信号，可能需要检查持仓是否足够
        if "insufficient quantity" in str(e).lower():
            print("注意：卖出失败，可能没有足够的持仓。")


if __name__ == '__main__':
    # 在 T-2 市场开放时间 (通常是工作日 9:30 AM ET 到 4:00 PM ET) 运行此代码
    # 如果是非市场开放时间，可能会出现数据/交易错误。
    
    print("--- 启动 Alpaca 模拟交易执行器 ---")
    
    # 检查账户是否可用
    account = trading_client.get_account()
    if account.status != 'ACTIVE':
        print(f"🔴 账户状态不可用: {account.status}")
    else:
        print(f"🟢 账户状态活跃。当前可用现金: ${float(account.cash):.2f}")
        
        # 执行 TSLA 的交易逻辑
        execute_trading_signal(ticker="TSLA")
        
        print("\n--- 执行完成 ---")