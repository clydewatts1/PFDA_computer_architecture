import yfinance as yf
# Ticker List
# Facebook (META)
# Apple (AAPL)
# Amazon (AMZN)
# Netflix (NFLX)
# Google (GOOG)
tickers = ["META", "AAPL", "AMZN", "NFLX", "GOOG"]

# Create a ticker object for Apple
aapl = yf.Ticker(tickers[0])

# Get historical market data for the last year
hist = aapl.history(period="1y")
    
# Display the last 5 rows of the data
print(hist.tail())