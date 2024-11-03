import numpy as np
from TechnicalIndicators import *
from Performance import *
from Visualize import *
import yfinance as yf
from tabulate import tabulate


class StockTradingBacktest:
    def __init__(self, df, initial_capital):
        """
        Initializes the Backtest class.

        Parameters:
        df (pd.DataFrame): Dataframe of historical stock data from yfinance.
        initial_capital (float): Initial capital to begin the backtest.
        """
        self.df = df
        self.initial_capital = initial_capital
        self.performance_evaluator = PerformanceEvaluator(
            df=df,
            initial_capital=initial_capital,
        )
        self.capital = initial_capital
        self.shares = 0
        self.profits = []

        # Create a dictionary that maps indicator names to plot functions and signal column names
        self.indicators = {
            'bollinger_bands': (Visualize.plot_bollinger_bands, 'bollinger bands signals'),
            'heikin_ashi': (Visualize.plot_heikin_ashi, 'HA signals'),
            'macd': (Visualize.plot_macd, 'macd signals'),
            'sar': (Visualize.plot_sar, 'sar signals'),
            'rsi': (Visualize.plot_rsi, 'rsi signals'),
            'golden_death_cross': (Visualize.plot_golden_death_cross, 'golden death cross signal'),
            'stochastic_oscillator': (Visualize.plot_stochastic_oscillator, 'stochastic signals'),
            'roc': (Visualize.plot_roc, 'roc signals'),
            'williams_r': (Visualize.plot_williams_r, 'williams_r signals'),
            'cci': (Visualize.plot_cci, 'cci signals'),
            'hv': (Visualize.plot_hv, 'hv signals'),
            'sd': (Visualize.plot_sd, 'sd signals'),
            'vol_osc': (Visualize.plot_vol_osc, 'vol_osc_signals'),
            'vroc': (Visualize.plot_vroc, 'VROC signals'),
            'mfi': (Visualize.plot_mfi, 'MFI signals'),
        }

    def reset_state(self):
        self.capital = self.initial_capital
        self.shares = 0
        self.profits = []

    # Generalized plotting function
    def plot_indicator(self, indicator_name):
        if indicator_name in self.indicators:
            plot_function, signal_column = self.indicators[indicator_name]
            self.execute_plot(plot_function, signal_column)
        else:
            print(f"Indicator {indicator_name} not found.")

    def execute_plot(self, plot_function, signal_column):
        buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y, updated_state = self.performance_evaluator.get_buy_sell_coordinates(
            signal_column)
        self.capital = updated_state['capital']
        self.shares = updated_state['shares']
        self.profits = updated_state['profits']

        self.print_performance_summary()
        self.reset_state()

    def print_performance_summary(self):
        overall_percent_change = (
            (self.capital - self.initial_capital) / self.initial_capital) * 100

        wins = self.performance_evaluator.trading_state['wins']
        losses = self.performance_evaluator.trading_state['losses']

        avg_win_percent = np.mean(
            [p for p in self.performance_evaluator.trading_state['pct_changes'] if p > 0]) if wins > 0 else 0
        avg_loss_percent = np.mean(
            [p for p in self.performance_evaluator.trading_state['pct_changes'] if p < 0]) if losses > 0 else 0

        table = [
            ["Initial Capital", f"${self.initial_capital:,}"],
            ["Ending Capital", f"${self.capital:,}"],
            ["Profit/Loss", f"${sum(self.profits):,.2f}"],
            ["Total Trades", len(self.profits)],
            ["Winning Trades", wins],
            ["Losing Trades", losses],
            ["Overall Percent Change", f"{overall_percent_change:.2f}%"],
            ["Average Win %", f"{avg_win_percent:.2f}%"],
            ["Average Loss %", f"{avg_loss_percent:.2f}%"],
        ]
        print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))


def main():
    df = yf.download("AAPL", "2020-01-04", "2024-08-04")
    df = TechnicalIndicators(df).generate_all_signals()

    backtest = StockTradingBacktest(df, initial_capital=10000)

    # Now you can call the plot method with the indicator name
    indicators_to_test = [
        'bollinger_bands', 'heikin_ashi', 'macd', 'sar', 'rsi',
        'golden_death_cross', 'stochastic_oscillator', 'roc',
        'williams_r', 'cci', 'hv', 'sd', 'vol_osc', 'vroc', 'mfi'
    ]

    for indicator in indicators_to_test:
        print(f"\n{indicator.capitalize()} Performance:")
        backtest.plot_indicator(indicator)


if __name__ == '__main__':
    main()
