
import numpy as np
from ClaculateTechnicals import *
from Optimize import *
from PerformanceMetrics import *
from Visualize import *

class StockTradingBacktest:
    def __init__(self, df, initial_capital, position_size=None, position_strategy='uniform'):
        """
        Initializes the Backtest class.

        Parameters:
        df (pd.DataFrame): Dataframe of historical stock data from yfinance.
        initial_capital (float): Initial capital to begin the backtest.
        position_size (float, optional): Total capital to allocate to each trading position.
                                        Required if position_strategy is 'uniform'.
        position_strategy (str): Strategy for position sizing. Options are:
            - 'uniform': Allocate the same amount of capital to each position.
            - 'fixed_fractional': Allocate a fixed percentage of capital to each position.
            - 'volatility_adjusted': Adjust position size based on the volatility of the asset.
            - 'equity_curve_based': Adjust position size based on recent trading performance.
        """
        self.df = df
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.position_strategy = position_strategy
        
        # Ensure position_size is provided for uniform strategy
        if self.position_strategy == 'uniform' and self.position_size is None:
            raise ValueError(
                "Position size must be provided for uniform strategy")

        self.position_sizer = PositionSizer(initial_capital, position_size=position_size)

        self.performance_evaluator = PerformanceEvaluator(
            df=df,
            position_sizer=self.position_sizer,
            initial_capital=initial_capital,
            position_size=position_size,
            position_strategy=position_strategy
        )

        self.capital = initial_capital
        self.shares = 0
        self.profits = []

    def reset_state(self):
        self.capital = self.initial_capital
        self.shares = 0
        self.profits = []
        self.position_sizer.update_capital(self.initial_capital)


    # Plotting #
    def execute_plot(self, plot_function, signal_column):
        buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y, updated_state = self.performance_evaluator.get_buy_sell_coordinates(signal_column)
        self.capital = updated_state['capital']
        self.shares = updated_state['shares']
        self.profits = updated_state['profits']
        plot_function(self.df, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y, self.position_strategy, self.profits)
        self.reset_state()
    
    def plot_bollinger_bands(self):
        self.execute_plot(Visualize.plot_bollinger_bands, 'bollinger bands signals')

    def plot_heikin_ashi(self):
        self.execute_plot(Visualize.plot_heikin_ashi, 'HA signals')

    def plot_macd(self):
        self.execute_plot(Visualize.plot_macd, 'macd signals')
    
    def plot_sar(self):
        self.execute_plot(Visualize.plot_sar, 'sar signals')

    def plot_rsi(self):
        self.execute_plot(Visualize.plot_rsi, 'rsi signals')
    
    def plot_golden_death_cross(self):
        self.execute_plot(Visualize.plot_golden_death_cross, 'golden death cross signal')
    
    def plot_stochastic_oscillator(self):
        self.execute_plot(Visualize.plot_stochastic_oscillator, 'stochastic signals')
    
    def plot_roc(self):
        self.execute_plot(Visualize.plot_roc, 'roc signals')
    
    def plot_williams_r(self):
        self.execute_plot(Visualize.plot_williams_r, 'williams_r signals')
    
    def plot_cci(self):
        self.execute_plot(Visualize.plot_cci, 'cci signals')
    
    def plot_hv(self):
        self.execute_plot(Visualize.plot_hv, 'hv signals')
    
    def plot_sd(self):
        self.execute_plot(Visualize.plot_sd, 'sd signals')

    def plot_vol_osc(self):
        self.execute_plot(Visualize.plot_vol_osc, 'vol_osc_signals')
    
    def plot_vroc(self):
        self.execute_plot(Visualize.plot_vroc, 'VROC signals')

    def plot_mfi(self):
        self.execute_plot(Visualize.plot_mfi, 'MFI signals')

    def plot_ridge_lasso(self):
        self.execute_plot(Visualize.plot_ridge_lasso, 'ridge/lasso signals')

    



def main():

    #df = yf.download('AAPL', start='2006-01-01', end='2010-12-31', progress=False)
    df = yf.download("AAPL", "2020-01-04", "2024-08-04")
    df = TechnicalIndicators(df).generate_all_signals()
    
    for strategy in ['uniform', 'fixed_fractional', 'volatility_adjusted', 'equity_curve_based']:
        if strategy == 'uniform':
            backtest = StockTradingBacktest(df, initial_capital=10000, position_size=1000, position_strategy=strategy)
            # Plot technical indicators
            backtest.plot_bollinger_bands()
            backtest.plot_heikin_ashi()
            backtest.plot_macd()
            backtest.plot_sar()
            backtest.plot_rsi()
            backtest.plot_golden_death_cross()
            backtest.plot_stochastic_oscillator()
            backtest.plot_roc()
            backtest.plot_williams_r()
            backtest.plot_cci()
            backtest.plot_hv()
            backtest.plot_sd()
            backtest.plot_vol_osc()
            backtest.plot_vroc()
            backtest.plot_mfi()


if __name__=='__main__':
    main()