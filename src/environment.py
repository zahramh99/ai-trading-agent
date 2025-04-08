# trading environment
class TradingEnvironment:
    def __init__(self, data):
        self.data = data
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.holdings = 0
        self.index = 0

    def reset(self):
        self.balance = self.initial_balance
        self.holdings = 0
        self.index = 0
        return get_state(self.data, self.index)

    def step(self, action):
        price = float(self.data.loc[self.index, 'Close'])
        reward = 0

        if action == 1 and self.balance >= price:  # BUY
            self.holdings = self.balance // price
            self.balance -= self.holdings * price
        elif action == 2 and self.holdings > 0:  # SELL
            self.balance += self.holdings * price
            self.holdings = 0

        self.index += 1
        done = self.index >= len(self.data) - 1

        if done:
            reward = self.balance - self.initial_balance

        next_state = get_state(self.data, self.index) if not done else None
        return next_state, reward, done, {}