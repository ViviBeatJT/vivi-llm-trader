from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

# --- Configuration ---
TICKER = 'SPLV'
INTERVAL = '5Min'
FILENAME = f'data/{TICKER}_{INTERVAL}_1year_data.csv' # Define the desired output filename

# 1. Initialize the Client
# Replace with your actual API Key ID and Secret Key
# Be mindful of pagination for a full year of 5-minute data!
api_key = os.getenv('ALPACA_API_KEY_ID')
secret_key = os.getenv('ALPACA_SECRET_KEY')
stock_client = StockHistoricalDataClient(api_key, secret_key)

# 2. Define the Request Parameters
end_date = datetime.now().date()
start_date = end_date - timedelta(days=365) # Approx 1 year

request_params = StockBarsRequest(
    symbol_or_symbols=[TICKER],
    timeframe=TimeFrame(5,TimeFrameUnit.Minute), 
    start=start_date,
    end=end_date,
    adjustment='split', 
)

# 3. Fetch the Data 
# The .df accessor converts the Alpaca response into a clean Pandas DataFrame.
# Alpaca's SDK handles pagination (multiple requests) automatically if needed.
tsla_bars = stock_client.get_stock_bars(request_params).df

# 4. Save the DataFrame to a CSV file (The new crucial step!)
# index=True is the default, but we specify it to be explicit.
# Pandas uses the 'timestamp' (which is the bar's open time) as the index.
tsla_bars.to_csv(FILENAME, index=True) 

print(f"Successfully downloaded {len(tsla_bars)} bars.")
print(f"Data saved to: {FILENAME}")