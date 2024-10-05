import yfinance as yf
import numpy as np
import pandas as pd

'''
Risk and Performance Metrics:
Cumulative Profit/Loss: Tracks overall profitability.
Win/Loss Ratio: Tracks how often you win compared to how often you lose.
Average Win Size vs. Average Loss Size: Measures the profitability of wins relative to losses.
Drawdown (Max and Rolling): Monitors your largest losses.
Volatility: Tracks how volatile your portfolio is and how well you are managing risk.
Trade Frequency: How many trades are executed in a certain time period. This can impact your transaction costs and performance.
Risk-Adjusted Returns: Use the Sharpe, Sortino, or other ratios to monitor risk-adjusted performance.

'''
class MainAgent:
    def __init__(self, df, trend_agent, momentum_agent, volume_agent, volatility_agent):
        self.df = df
        self.trend_agent = trend_agent
        self.momentum_agent = momentum_agent
        self.volume_agent = volume_agent
        self.volatility_agent = volatility_agent

    def aggregate_signals(self, window_size):
        """
        Aggregates signals from all sub-agents over a sliding window.
        
        (TODO) dynamically adjust window size based on a heuristic
        """
        aggregated_signals = []
        for day in range(window_size, len(self.df)):
            window_data = self.df.iloc[day - window_size:day]

            trend_signal = self.trend_agent.generate_signal(window_data)
            momentum_signal = self.momentum_agent.generate_signal(window_data)
            volume_signal = self.volume_agent.generate_signal(window_data)
            volatility_signal = self.volatility_agent.generate_signal(
                window_data)

            # Aggregating signals (e.g., summing signals or using a weighted voting system)
            total_signal = trend_signal + momentum_signal + volume_signal + volatility_signal
            aggregated_signals.append(self.make_trading_decision(total_signal))

        return aggregated_signals

    def make_trading_decision(self, total_signal):
        """
        Based on the combined signal, make a buy/sell/hold decision.
        """
        if total_signal > 1:
            return "buy"
        elif total_signal < -1:
            return "sell"
        else:
            return "hold"
