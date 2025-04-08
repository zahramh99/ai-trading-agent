from environment import TradingEnvironment
from agent import DQNAgent
from utils import get_state, ACTIONS
import yfinance as yf
import pandas as pd

def main():
    # Data collection and preprocessing
    symbol = "AAPL"
    start_date = "2020-01-01"
    end_date = "2025-02-14"
    data = yf.download(symbol, start=start_date, end=end_date)
    
    # Feature engineering
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Returns'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    # Training
    env = TradingEnvironment(data)
    agent = DQNAgent(state_size=4, action_size=3)
    
    # ... (rest of your training code)
    
    # Testing
    test_env = TradingEnvironment(data)
    state = test_env.reset()
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = test_env.step(action)
        state = next_state if next_state is not None else state
    
    final_balance = test_env.balance
    profit = final_balance - test_env.initial_balance
    print(f"Final Balance after testing: ${final_balance:.2f}")
    print(f"Total Profit: ${profit:.2f}")

if __name__ == "__main__":
    main()
    