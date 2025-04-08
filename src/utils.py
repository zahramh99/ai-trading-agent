import numpy as np
import pandas as pd

# Define action space constants
ACTIONS = {
    0: "HOLD",
    1: "BUY", 
    2: "SELL"
}

def get_state(data: pd.DataFrame, index: int) -> np.ndarray:
    """
    Extracts the state representation from the dataset at a given time index.
    
    The state consists of:
    - Closing price
    - 5-day Simple Moving Average (SMA)
    - 20-day Simple Moving Average (SMA)
    - Daily return percentage
    
    Args:
        data: DataFrame containing the stock market data
        index: Current time index in the data
        
    Returns:
        np.ndarray: Array containing the state features
    """
    return np.array([
        float(data.loc[index, 'Close']),
        float(data.loc[index, 'SMA_5']),
        float(data.loc[index, 'SMA_20']), 
        float(data.loc[index, 'Returns'])
    ])

def prepare_stock_data(symbol: str = "AAPL", 
                      start_date: str = "2020-01-01", 
                      end_date: str = "2025-02-14") -> pd.DataFrame:
    """
    Downloads and prepares stock market data with technical indicators.
    
    Args:
        symbol: Stock ticker symbol (default: 'AAPL')
        start_date: Start date for data (YYYY-MM-DD format)
        end_date: End date for data (YYYY-MM-DD format)
        
    Returns:
        pd.DataFrame: Processed DataFrame with technical indicators
    """
    # Download historical data
    data = yf.download(symbol, start=start_date, end=end_date)
    
    # Calculate technical indicators
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Returns'] = data['Close'].pct_change()
    
    # Clean data
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    return data