# test_alpaca_connection.py
import os
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

# 加载 .env 文件中的环境变量
load_dotenv()

# 从环境变量获取 API Key
# 注意：若未设置，程序会报错，保证安全
API_KEY_ID = os.getenv('ALPACA_API_KEY_ID')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

# 设置连接参数：使用 Paper Trading (模拟交易) 的 URL
# data_feed='iex' 是默认的，用于获取报价数据
client = StockHistoricalDataClient(API_KEY_ID, SECRET_KEY)

def test_alpaca_data_connection():
    """
    测试 Alpaca 数据 API 连接，并获取 AAPL 的最新报价。
    """
    if not API_KEY_ID or not SECRET_KEY:
        print("❌ 错误：请检查 .env 文件中是否正确设置了 ALPACA_API_KEY_ID 和 ALPACA_SECRET_KEY。")
        return

    print("--- 尝试连接 Alpaca Data API ---")
    
    try:
        # 创建请求对象，请求 AAPL 的最新报价
        request_params = StockLatestQuoteRequest(symbol_or_symbols=["AAPL"])
        
        # 发送请求
        latest_quote = client.get_stock_latest_quote(request_params)
        
        # 结果是一个字典，键是股票代码
        aapl_quote = latest_quote.get("AAPL")
        
        if aapl_quote:
            print("\n✅ 连接成功！")
            print("--- 获取 AAPL 最新报价 ---")
            print(f"股票代码: AAPL")
            print(f"最新买入价 (Bid): {aapl_quote.bid_price}")
            print(f"最新卖出价 (Ask): {aapl_quote.ask_price}")
            print(f"交易所时间戳: {aapl_quote.timestamp}")
            
        else:
            print("\n❌ 连接失败或获取数据失败。请检查股票代码或网络连接。")

    except Exception as e:
        print(f"\n❌ 发生连接异常。请检查 API Key 是否正确。错误信息: {e}")

if __name__ == '__main__':
    test_alpaca_data_connection()