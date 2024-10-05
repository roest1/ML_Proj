'''
Optimize class
===============

- Position sizing 

Future Goals:

Include transaction costs, slippage, market impact, liquidity, and scenario analysis
'''

from TechnicalIndicators import *


class PositionSizer:
    def __init__(self, capital, risk_fraction=0.02, risk_per_share=100, lookback=10, risk_increase_factor=2, position_size=None):
        self.capital = capital
        self.risk_fraction = risk_fraction
        self.risk_per_share = risk_per_share
        self.lookback = lookback
        self.risk_increase_factor = risk_increase_factor
        self.position_size = position_size

    def update_capital(self, new_capital):
        self.capital = new_capital

    def calculate_position_size(self, strategy, atr=None, profits=None):
        if strategy == 'fixed_fractional':
            return self.fixed_fractional()
        elif strategy == 'volatility_adjusted':
            if atr is None:
                raise ValueError(
                    "ATR value is required for volatility_adjusted strategy")
            return self.volatility_adjusted(atr)
        elif strategy == 'equity_curve_based':
            if profits is None:
                raise ValueError(
                    "Profits data is required for equity_curve_based strategy")
            return self.equity_curve_based(profits)
        elif strategy == 'uniform':
            return self.uniform()
        else:
            raise ValueError("Unknown position strategy")

    def fixed_fractional(self):
        return self.capital * self.risk_fraction

    def volatility_adjusted(self, atr):
        return self.risk_per_share / atr

    def equity_curve_based(self, profits):
        recent_performance = np.mean(profits[-self.lookback:])
        return self.capital * (self.risk_increase_factor if recent_performance > 0 else 1)
    
    def uniform(self):
        return self.position_size

'''
Hyperparameter grid search for technical indicator signal generators

parameter_config = {
    'heikin_ashi_signal_generation': {'stop_loss': [2, 3, 4, 5]},
    'parabolic_sar_signal_generation': {
        'initial_af': [0.01, 0.02, 0.03, 0.04, 0.05],
        'step_af': [0.01, 0.02],
        'end_af': [0.1, 0.2, 0.3]
    },
    'macd_signal_generation': {
        'ma1': [8, 10, 12, 14, 16],
        'ma2': [20, 22, 24, 26, 28, 30],
        'signal': [7, 8, 9, 10, 11, 12],
        'ma_type': ['sma', 'ema']
    },
    'golden_death_cross_signal_generation': {
        'short_window': [40, 45, 50, 55, 60],
        'long_window': [180, 190, 200, 210, 220],
        'ma_type': ['sma', 'ema']
    },
    'rsi_signal_generation': {'lag_days': [10, 12, 14, 16, 18, 20]},
    'stochastic_oscillator_signal_generation': {
        'k_period': [10, 12, 14, 16, 18, 20],
        'd_period': [2, 3, 4, 5]
    },
    'roc_signal_generation': {'n_days': [10, 12, 14, 16, 18, 20]},
    'williams_r_signal_generation': {'n_days': [10, 12, 14, 16, 18, 20]},
    'cci_signal_generation': {'n_days': [15, 17, 19, 21, 23, 25]},
    'bollinger_bands_signal_generation': {
        'window': [15, 17, 19, 21, 23, 25],
        'num_of_std': [1.5, 1.75, 2, 2.25, 2.5]
    },
    'historical_volatility_signal_generation': {
        'window': [15, 17, 19, 21, 23, 25],
        'volatility_threshold': [0.05, 0.075, 0.1, 0.125, 0.15]
    },
    'standard_deviation_signal_generation': {'window': [5, 7, 10, 12, 15]},
    'volume_oscillator_signal_generation': {
        'short_period': [10, 12, 14, 15],
        'long_period': [20, 22, 25, 28, 30]
    },
    'vroc_signal_generation': {'period': [20, 25, 30, 35, 40]},
    'mfi_signal_generation': {'period': [10, 12, 14, 16, 18, 20]}
}
'''
