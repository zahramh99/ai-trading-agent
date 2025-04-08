# AI Trading Agent using Agentic AI

This project implements a Deep Q-Network (DQN) agent for stock trading using Agentic AI principles.
Agentic AI involves several key components.I have built an AI Agent for trading. The key components of Agentic AI with  example:
The Agent: The agent is the decision-making entity in the AI system. In our case, the DQN trading agent will be responsible for making trading decisions based on market data.
The Environment: The environment is the external system in which the agent operates. Our trading environment will consist of stock market data, where the agent will interact with price movements and execute trades.
The State: The state represents the information available to the agent at any given time. Our trading agent’s state includes the stock’s closing price, moving averages, and daily returns.
The Action Space: The action space defines what actions the agent can take. Our trading agent has three possible actions: Buy, Sell, and Hold.
The Reward Function: The reward function determines the agent’s performance by assigning a numerical value to its actions. The goal of our trading agent will be to maximize total profit by the end of the trading session.

## Features
- DQN-based trading agent
- Custom trading environment
- Technical indicators integration
- Experience replay

## Installation
```bash
pip install -r requirements.txt

## Usage
python src/main.py

## Project Structure
ai-trading-agent/
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── environment.py
│   ├── agent.py
│   ├── dqn.py
│   └── utils.py
├── data/
│   └── aapl_2020-2024.csv
├── notebooks/
│   └── exploration.ipynb
└── results/
    └── training_results.txt

    
I will be happy to answer any questions about the code or development process so if you have any questions please fell free to contac me. Your thoughts and suggestions are always appreciated! :))

