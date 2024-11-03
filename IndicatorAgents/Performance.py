import numpy as np
from datetime import datetime


class PerformanceEvaluator:
    def __init__(self, df, initial_capital, commission=0.001, slippage_percent=0.001):
        self.df = df
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage_percent = slippage_percent
        self.trading_state = {
            'capital': initial_capital,
            'shares': 0,
            'profits': [],
            'pct_changes': [],
            'wins': 0,
            'losses': 0,
            'trades': []
        }

    def slippage(self, highlows):
        """
        Efficiently calculate the average of high and low values plus 1 standard deviation for slippage estimation.
        """
        ohlc_array = np.asarray(highlows)
        high_low_avg = (ohlc_array[:, 0] + ohlc_array[:, 1]) / 2
        avg = np.mean(high_low_avg)
        std_dev = np.std(high_low_avg)
        return avg + std_dev

    def calculate_slippage(self, price):
        """
        Apply slippage to a given price using the slippage_percent defined at class initialization.
        """
        return price * (1 + self.slippage_percent)

    def calculate_commission(self, shares, price):
        """
        Calculate the commission fees based on the number of shares and the price.
        """
        return shares * price * self.commission

    def record_trade(self, ticker, entry_price, exit_price, shares, buy_time, sell_time, slippage, fees, trade_profit, trade_pct_change):
        """
        Records details of each trade, including entry/exit timestamps, prices, slippage, and fees.
        """
        trade_duration = (
            sell_time - buy_time).total_seconds() / 3600  # in hours
        trade_record = {
            'Ticker': ticker,
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'Shares': shares,
            'Buy Time': buy_time,
            'Sell Time': sell_time,
            'Trade Duration (hours)': trade_duration,
            'Slippage': slippage,
            'Fees': fees,
            'Trade Profit': trade_profit,
            'Trade Pct Change': trade_pct_change
        }
        self.trading_state['trades'].append(trade_record)

    def get_buy_sell_coordinates(self, signal_column, ticker):
        df = self.df
        trading_state = self.trading_state

        buy_signals_x = []
        buy_signals_y = []
        sell_signals_x = []
        sell_signals_y = []

        buy_at_ohlc = 'Low'  # Buy at low to account for market slippage
        sell_at_ohlc = 'High'  # Sell at high to account for market slippage

        for i in range(len(df) - 1):
            execution_date = i + 1
            buy_time = df.index[execution_date]

            # Buy
            if df[signal_column].iloc[i] == 1:
                num_shares_to_buy = 1
                buy_price = df[buy_at_ohlc].iloc[execution_date]
                slippage_price = self.calculate_slippage(buy_price)
                commission_fee = self.calculate_commission(
                    num_shares_to_buy, slippage_price)

                # Update trading state
                trading_state['shares'] += num_shares_to_buy
                trading_state['capital'] -= (num_shares_to_buy *
                                             slippage_price) + commission_fee
                buy_signals_x.append(buy_time)
                buy_signals_y.append(buy_price)

            # Sell
            elif df[signal_column].iloc[i] == -1 and buy_signals_y and trading_state['shares'] > 0:
                sell_price = df[sell_at_ohlc].iloc[execution_date]
                sell_time = df.index[execution_date]
                slippage_price = self.calculate_slippage(sell_price)
                commission_fee = self.calculate_commission(
                    trading_state['shares'], slippage_price)

                # Update capital and record the trade
                sell_revenue = trading_state['shares'] * slippage_price
                trading_state['capital'] += sell_revenue - commission_fee
                trade_profit = sell_revenue - \
                    (buy_signals_y[-1] *
                     trading_state['shares']) - commission_fee
                trade_pct_change = (
                    (sell_price - buy_signals_y[-1]) / buy_signals_y[-1]) * 100

                # Track buy/sell signals
                sell_signals_x.append(sell_time)
                sell_signals_y.append(sell_price)

                # Record the trade
                self.record_trade(
                    ticker=ticker,
                    entry_price=buy_signals_y[-1],
                    exit_price=sell_price,
                    shares=trading_state['shares'],
                    buy_time=buy_signals_x[-1],
                    sell_time=sell_time,
                    slippage=slippage_price,
                    fees=commission_fee,
                    trade_profit=trade_profit,
                    trade_pct_change=trade_pct_change
                )

                # Update profit and percent change tracking
                trading_state['profits'].append(trade_profit)
                trading_state['pct_changes'].append(trade_pct_change)

                # Track wins and losses
                if trade_profit > 0:
                    trading_state['wins'] += 1
                else:
                    trading_state['losses'] += 1

                # Reset shares after selling
                trading_state['shares'] = 0

        return buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y, trading_state

    def print_trade_summary(self):
        """
        Prints a summary of all recorded trades.
        """
        for trade in self.trading_state['trades']:
            print(f"Ticker: {trade['Ticker']}, Entry: {trade['Entry Price']}, Exit: {trade['Exit Price']}, "
                  f"Shares: {trade['Shares']}, Profit: {trade['Trade Profit']:.2f}, "
                  f"Pct Change: {trade['Trade Pct Change']:.2f}%")
