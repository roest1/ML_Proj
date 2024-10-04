import numpy as np
import warnings
warnings.filterwarnings('ignore')

'''
All Indicators:
================
- Heikin-Ashi
- Parabolic SAR
- Moving Average Convergence Divergence (MACD)
- Golden/Death Cross
- Relative Strength Index (RSI)
- Stochastic Oscillator
- Rate of Change (ROC)
- Williams %R
- Commodity Channel Index (CCI)
- Bollinger Bands
- Historical Volatility
- Standard Deviation
- Volume Oscillator
- Volume Rate of Change (VROC)
- Money Flow Index (MFI)
'''


class TrendIndicators:
    '''
    - Heikin-Ashi
    - Parabolic SAR
    - Moving Average Convergence Divergence (MACD)
    - Golden/Death Cross
    '''

    def __init__(self, df):
        self.df = df

    def heikin_ashi_signal_generation(self, stop_loss=3):
        '''
        Heikin-Ashi Parameters:
        =======================
        stop_loss (float):
        - The number of consecutive buy/sell signals to generate before switching to the opposite signal
        '''
        df = self.df
        df['HA close'] = (df['Open'] + df['Close'] +
                          df['High'] + df['Low']) / 4
        df['HA open'] = df['Open'].copy()
        df['HA open'][1:] = 0.5 * (df['Open'][:-1] + df['Close'][:-1])
        df['HA high'] = df[['HA open', 'HA close', 'High']].max(axis=1)
        df['HA low'] = df[['HA open', 'HA close', 'Low']].min(axis=1)
        df['HA signals'] = 0
        df['cumsum'] = 0  # For stop_loss
        for n in range(1, len(df)):
            if (df['HA open'].iloc[n] > df['HA close'].iloc[n]
                and df['HA open'].iloc[n] == df['HA high'].iloc[n]
                and np.abs(df['HA open'].iloc[n] - df['HA close'].iloc[n]) > np.abs(df['HA open'].iloc[n-1] - df['HA close'].iloc[n-1])
                    and df['HA open'].iloc[n-1] > df['HA close'].iloc[n-1]):

                df['HA signals'].iloc[n] = 1 if df['cumsum'].iloc[n -
                                                                  1] < stop_loss else 0

            elif (df['HA open'].iloc[n] < df['HA close'].iloc[n]
                  and df['HA open'].iloc[n] == df['HA low'].iloc[n]
                  and df['HA open'].iloc[n-1] < df['HA close'].iloc[n-1]):

                df['HA signals'].iloc[n] = - \
                    1 if df['cumsum'].iloc[n-1] > 0 else 0

            df['cumsum'].iloc[n] = df['HA signals'].iloc[:n+1].sum()

        df.drop(['cumsum'], axis=1, inplace=True)
        return df

    def parabolic_sar_signal_generation(self, initial_af=0.02, step_af=0.02, end_af=0.2):
        '''
        Parabolic SAR Parameters:
        ========================
        initial_af (float):
        - Initial acceleration factor for the SAR calculation
        step_af (float):
        - Step value by which the acceleration factor is increased
        end_af (float):
        - Maximum value that the acceleration factor can reach
        '''
        df = self.df

        df['trend'] = 0
        df['sar'] = 0.0
        df['ep'] = df['High'] if df['Close'].iloc[1] > df['Close'].iloc[0] else df['Low']
        df['af'] = initial_af

        for i in range(1, len(df)):
            reverse = False
            prev_sar = df.iloc[i - 1, df.columns.get_loc('sar')]
            prev_ep = df.iloc[i - 1, df.columns.get_loc('ep')]
            prev_af = df.iloc[i - 1, df.columns.get_loc('af')]
            new_sar = prev_sar + prev_af * (prev_ep - prev_sar)

            # uptrend
            if df.iloc[i - 1, df.columns.get_loc('trend')] == 1:
                new_sar = min(new_sar, df.iloc[i - 1, df.columns.get_loc(
                    'Low')], df.iloc[i - 2, df.columns.get_loc('Low')])
                if df.iloc[i, df.columns.get_loc('Low')] < new_sar:
                    reverse = True
                    df.iloc[i, df.columns.get_loc('sar')] = prev_ep
                    df.iloc[i, df.columns.get_loc('trend')] = -1
                    df.iloc[i, df.columns.get_loc(
                        'ep')] = df.iloc[i, df.columns.get_loc('Low')]
                    df.iloc[i, df.columns.get_loc('af')] = initial_af
                else:
                    df.iloc[i, df.columns.get_loc('sar')] = new_sar
                    df.iloc[i, df.columns.get_loc('trend')] = 1
                    df.iloc[i, df.columns.get_loc('ep')] = max(
                        df.iloc[i, df.columns.get_loc('High')], prev_ep)
                    df.iloc[i, df.columns.get_loc('af')] = min(
                        end_af, prev_af + step_af) if df.iloc[i, df.columns.get_loc('High')] > prev_ep else prev_af
            # downtrend
            else:
                new_sar = max(new_sar, df.iloc[i - 1, df.columns.get_loc(
                    'High')], df.iloc[i - 2, df.columns.get_loc('High')])
                if df.iloc[i, df.columns.get_loc('High')] > new_sar:
                    reverse = True
                    df.iloc[i, df.columns.get_loc('sar')] = prev_ep
                    df.iloc[i, df.columns.get_loc('trend')] = 1
                    df.iloc[i, df.columns.get_loc(
                        'ep')] = df.iloc[i, df.columns.get_loc('High')]
                    df.iloc[i, df.columns.get_loc('af')] = initial_af
                else:
                    df.iloc[i, df.columns.get_loc('sar')] = new_sar
                    df.iloc[i, df.columns.get_loc('trend')] = -1
                    df.iloc[i, df.columns.get_loc('ep')] = min(
                        df.iloc[i, df.columns.get_loc('Low')], prev_ep)
                    df.iloc[i, df.columns.get_loc('af')] = min(
                        end_af, prev_af + step_af) if df.iloc[i, df.columns.get_loc('Low')] < prev_ep else prev_af

            if i == 1 and reverse:
                df.iloc[i, df.columns.get_loc(
                    'sar')] = df.iloc[0, df.columns.get_loc('Close')]

        df['positions'] = np.where(df['sar'] < df['Close'], 1, 0)
        df['sar signals'] = df['positions'].diff()
        # df.drop(['trend', 'sar', 'ep', 'af', 'positions'], axis=1, inplace=True)
        df.drop(['positions'], axis=1, inplace=True)
        return df

    def macd_signal_generation(self, ma_type='ema', ma1=12, ma2=26, signal=9):
        '''
        MACD Parameters:
        ================
        ma_type ('sma' or 'ema'):
        - Type of moving average to use (simple or exponential)
        ma1 (int):
        - Number of periods for short moving average
        ma2 (int):
        - Number of periods for long moving average
        signal (int):
        - Number of periods for signal line average
        '''
        df = self.df
        if ma_type == 'ema':
            df['macd short ma'] = df['Close'].ewm(
                span=ma1, adjust=False).mean()
            df['macd long ma'] = df['Close'].ewm(span=ma2, adjust=False).mean()
            df['macd ma1'] = df['macd short ma'] - df['macd long ma']
            df['macd ma2'] = df['macd ma1'].ewm(
                span=signal, adjust=False).mean()

        elif ma_type == 'sma':
            df['macd short ma'] = df['Close'].rolling(window=ma1).mean()
            df['macd long ma'] = df['Close'].rolling(window=ma2).mean()
            df['macd ma1'] = df['macd short ma'] - df['macd long ma']
            df['macd ma2'] = df['macd ma1'].rolling(window=signal).mean()

        df['macd divergence'] = df['macd ma1'] - df['macd ma2']
        df['macd positions'] = np.where(df['macd ma1'] > df['macd ma2'], 1, 0)
        df['macd signals'] = df['macd positions'].diff()
        df.drop(['macd positions'], axis=1, inplace=True)

        return df

    def golden_death_cross_signal_generation(self, short_window=50, long_window=200, ma_type='sma'):
        '''
        Golden Cross: 
        - This is a bullish signal that occurs when a short-term moving average crosses above a long-term moving average. 
          It's considered a confirmation of an upward trend.

        Death Cross: 
        - This is a bearish signal that occurs when a short-term moving average crosses below a long-term moving average,
          suggesting a potential downward trend.

        Golden Cross Parameters:
        ========================
        short_window (int):
        - Number of periods for short moving average
        long_window (int):
        - Number of periods for long moving average
        ma_type ('sma' or 'ema'):
        - Type of moving average to use (simple or exponential)
        '''
        df = self.df
        if ma_type == 'sma':
            df['gd ma_short'] = df['Close'].rolling(window=short_window).mean()
            df['gd ma_long'] = df['Close'].rolling(window=long_window).mean()
        elif ma_type == 'ema':
            df['gd ma_short'] = df['Close'].ewm(
                span=short_window, adjust=False).mean()
            df['gd ma_long'] = df['Close'].ewm(
                span=long_window, adjust=False).mean()

        df['golden death cross signal'] = 0

        df['golden death cross signal'][short_window:] = np.where(df['gd ma_short'][short_window:] > df['gd ma_long'][short_window:], 1,
                                                                  np.where(df['gd ma_short'][short_window:] < df['gd ma_long'][short_window:], -1, 0))

        df['golden_cross'] = (df['gd ma_short'] >
                              df['gd ma_long']).fillna(False)
        df['death_cross'] = (df['gd ma_short'] <
                             df['gd ma_long']).fillna(False)

        # Identify the crossover points
        df['golden_cross_points'] = df['golden_cross'] & (
            ~df['golden_cross'].shift().fillna(False))
        df['death_cross_points'] = df['death_cross'] & (
            ~df['death_cross'].shift().fillna(False))

        df.drop(['golden_cross', 'death_cross'], axis=1, inplace=True)
        return df


class MomentumIndicators:
    '''
    - Relative Strength Index (RSI)
    - Stochastic Oscillator
    - Rate of Change (ROC)
    - Williams %R
    - Commodity Channel Index (CCI)
    '''

    def __init__(self, df):
        self.df = df

    def rsi_signal_generation(self, lag_days=14, oversold=30, overbought=70):
        '''
        RSI Parameters:
        ===============
        lag_days (int):
        - The number of periods used for calculating the average gains and losses
        oversold (int):
        - Limit to generate buy signal
        overbought (int):
        - Limit to generate sell signal
        '''
        df = self.df
        df['rsi oversold'] = oversold
        df['rsi overbought'] = overbought
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.ewm(span=lag_days, adjust=False).mean()
        avg_loss = loss.ewm(span=lag_days, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['positions'] = np.select(
            [df['rsi'] < oversold, df['rsi'] > overbought], [1, -1], default=0)
        df['rsi signals'] = df['positions'].diff()
        df.drop(['positions'], axis=1, inplace=True)
        return df

    def stochastic_oscillator_signal_generation(self, k_period=14, d_period=3, overbought=80, oversold=20):
        '''
        Stochastic Oscillator Parameters:
        =================================
        k_period (int):
        - The number of periods used to calculate the %K line
        d_period (int):
        - The number of periods used to calculate the %D line (moving average of %K)
        '''
        df = self.df
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        df['%K'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100
        df['%D'] = df['%K'].rolling(window=d_period).mean()
        df['stochastic signals'] = 0

        # Trade signals based on %K and %D crossover
        # df.loc[(df['%K'].shift(1) < df['%D'].shift(1)) & (df['%K'] > df['%D']), 'stochastic signals'] = 1
        # df.loc[(df['%K'].shift(1) > df['%D'].shift(1)) & (df['%K'] < df['%D']), 'stochastic signals'] = -1

        # Trade signals based on %K crossing over overbought/oversold levels
        df.loc[(df['%K'] > overbought) & (df['%K'].shift(1) <=
                                          overbought), 'stochastic signals'] = -1  # Sell signal
        df.loc[(df['%K'] < oversold) & (df['%K'].shift(1) >= oversold),
               'stochastic signals'] = 1  # Buy signal

        df['stochastic oscillator overbought'] = overbought
        df['stochastic oscillator oversold'] = oversold
        return df

    def roc_signal_generation(self, n_days=14, oversold=-10, overbought=10):
        '''
        ROC Parameters:
        ===============
        n_days (int):
        - The number of periods used for the Rate of Change calculation
        overbought (int):
        - Limit to generate sell signal
        oversold (int):
        - Limit to generate buy signal
        '''
        df = self.df
        df['roc'] = ((df['Close'] - df['Close'].shift(n_days)) /
                     df['Close'].shift(n_days)) * 100
        df['roc signals'] = 0
        df.loc[df['roc'] > overbought, 'roc signals'] = -1
        df.loc[df['roc'] < oversold, 'roc signals'] = 1
        df['roc overbought'] = overbought
        df['roc oversold'] = oversold
        return df

    def williams_r_signal_generation(self, n_days=14, oversold=-80, overbought=-20):
        '''
        Williams %R Parameters:
        =======================
        n_days (int):
        - The number of periods used for the Williams %R calculation
        oversold (int):
        - Limit to generate buy signal
        overbought (int):
        - Limit to generate sell signal
        '''
        df = self.df
        highest_high = df['High'].rolling(window=n_days).max()
        lowest_low = df['Low'].rolling(window=n_days).min()
        df['Williams %R'] = ((highest_high - df['Close']) /
                             (highest_high - lowest_low)) * -100
        df['williams_r signals'] = 0
        df.loc[df['Williams %R'] > overbought, 'williams_r signals'] = -1
        df.loc[df['Williams %R'] < oversold, 'williams_r signals'] = 1
        df['Williams %R oversold'] = oversold
        df['Williams %R overbought'] = overbought
        return df

    def cci_signal_generation(self, n_days=20, oversold=-100, overbought=100):
        '''
        CCI Parameters:
        ===============
        n_days (int):
        - The number of periods used for the Commodity Channel Index calculation
        oversold (int):
        - Limit to generate buy signal
        overbought (int):
        - Limit to generate sell signal
        '''
        df = self.df
        TP = (df['High'] + df['Low'] + df['Close']) / 3
        SMA_TP = TP.rolling(window=n_days).mean()
        def mean_deviation(x): return np.mean(np.abs(x - np.mean(x)))
        MD = TP.rolling(window=n_days).apply(mean_deviation, raw=True)
        df['CCI'] = (TP - SMA_TP) / (0.015 * MD)
        df['cci signals'] = 0
        df.loc[df['CCI'] > overbought, 'cci signals'] = -1
        df.loc[df['CCI'] < oversold, 'cci signals'] = 1
        df['CCI oversold'] = oversold
        df['CCI overbought'] = overbought
        return df


class VolatilityIndicators:
    '''
    - Bollinger Bands
    - Historical Volatility
    - Standard Deviation
    '''

    def __init__(self, df):
        self.df = df

    def bollinger_bands_signal_generation(self, window=20, num_of_std=2):
        '''
        Bollinger Bands Parameters:
        ===========================
        window (int):
        - The number of periods used to calculate the moving average and standard deviation
        num_of_std (int):
        - The number of standard deviations for the upper and lower bands
        '''
        df = self.df
        rolling_stats = df['Close'].rolling(window=window, min_periods=window)
        df['bollinger bands mid band'] = rolling_stats.mean()
        std = rolling_stats.std()
        df['bollinger bands upper band'] = df['bollinger bands mid band'] + \
            num_of_std * std
        df['bollinger bands lower band'] = df['bollinger bands mid band'] - \
            num_of_std * std
        df['bollinger bands signals'] = 0
        df.loc[df['Close'] < df['bollinger bands lower band'],
               'bollinger bands signals'] = 1
        df.loc[df['Close'] > df['bollinger bands upper band'],
               'bollinger bands signals'] = -1
        return df

    def historical_volatility_signal_generation(self, window=20, volatility_threshold=.06):
        '''
        Historical Volatility Parameters:
        =================================
        window (int):
        - The number of periods used to calculate the historical volatility
        volatility_threshold (float):
        - The threshold above which the volatility is considered significant       
        '''
        df = self.df
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['historical_volatility'] = df['log_return'].rolling(
            window=window).std() * np.sqrt(window)
        df['hv signals'] = np.where(
            df['historical_volatility'] > volatility_threshold, 1, -1)
        df['hv threshold'] = volatility_threshold
        df.drop(['log_return'], axis=1, inplace=True)
        return df

    def standard_deviation_signal_generation(self, window=10):
        '''
        Standard Deviation Parmaters:
        window (int):
        - The number of periods used to calculate the standard deviation
        '''
        df = self.df
        df['standard_deviation'] = df['Close'].rolling(window=window).std()
        df['sd threshold'] = df['standard_deviation'].mean()
        df['sd signals'] = np.where(
            df['standard_deviation'] > df['sd threshold'], 1, -1)
        return df


class VolumeIndicators:
    '''
    - Volume Oscillator
    - Volume Rate of Change (VROC)
    - Money Flow Index (MFI)
    '''

    def __init__(self, df):
        self.df = df

    def volume_oscillator_signal_generation(self, short_period=12, long_period=26):
        '''
        Volume Oscillator Parameters:
        =============================
        short_period (int):
        - The number of periods for the short-term moving average of volume 
        long_period (int):
        - The number of periods for the long-term moving average of volume
        '''
        df = self.df
        short_vol = df['Volume'].rolling(window=short_period).mean()
        long_vol = df['Volume'].rolling(window=long_period).mean()
        df['vol_osc'] = short_vol - long_vol
        df['vol_osc_signals'] = np.where(df['vol_osc'] > 0, 1, -1)
        return df

    def vroc_signal_generation(self, period=34):
        '''
        VROC Parameters:
        ================
        period (int):
        - The number of periods used for the Volume Rate of Change calculation
        '''
        df = self.df
        volume_shifted = df['Volume'].shift(period)

        # Avoid division by zero by replacing zeros in shifted volume with NaN
        volume_shifted = volume_shifted.replace(0, np.nan)

        # Calculate VROC
        df['VROC'] = ((df['Volume'] - volume_shifted) / volume_shifted) * 100

        # Generate signals
        df['VROC signals'] = 0
        df['VROC signals'][df['VROC'] > 0] = 1
        df['VROC signals'][df['VROC'] < 0] = -1

        # Replace any remaining NaNs with 0 (or handle them as you see fit)
        df['VROC'].fillna(0, inplace=True)

        return df

    def mfi_signal_generation(self, period=14):
        '''
        MFI Parameters:
        ===============
        period (int):
        - The number of periods used for the Money Flow Index calculation
        '''
        df = self.df
        df['Typical Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['Money Flow'] = df['Typical Price'] * df['Volume']
        df['Positive Money Flow'] = df['Money Flow'].where(
            df['Typical Price'] > df['Typical Price'].shift(1), 0)
        df['Negative Money Flow'] = df['Money Flow'].where(
            df['Typical Price'] < df['Typical Price'].shift(1), 0)
        df['MFI'] = 100 - (100 / (1 + (df['Positive Money Flow'].rolling(
            window=period).sum() / df['Negative Money Flow'].rolling(window=period).sum())))
        df['MFI signals'] = 0
        df['MFI signals'][df['MFI'] > 80] = -1
        df['MFI signals'][df['MFI'] < 20] = 1
        df.drop(['Typical Price', 'Money Flow', 'Positive Money Flow',
                'Negative Money Flow'], axis=1, inplace=True)
        return df


class TechnicalIndicators:
    def __init__(self, df):
        self.df = df.copy()
        self.trend = TrendIndicators(self.df)
        self.momentum = MomentumIndicators(self.df)
        self.volatility = VolatilityIndicators(self.df)
        self.volume = VolumeIndicators(self.df)

    def generate_trend_signals(self):
        self.df = self.trend.heikin_ashi_signal_generation()
        self.df = self.trend.parabolic_sar_signal_generation()
        self.df = self.trend.macd_signal_generation()
        self.df = self.trend.golden_death_cross_signal_generation()
        return self.df

    def generate_momentum_signals(self):
        self.df = self.momentum.rsi_signal_generation()
        self.df = self.momentum.stochastic_oscillator_signal_generation()
        self.df = self.momentum.roc_signal_generation()
        self.df = self.momentum.williams_r_signal_generation()
        self.df = self.momentum.cci_signal_generation()
        return self.df

    def generate_volatility_signals(self):
        self.df = self.volatility.bollinger_bands_signal_generation()
        self.df = self.volatility.historical_volatility_signal_generation()
        self.df = self.volatility.standard_deviation_signal_generation()
        return self.df

    def generate_volume_signals(self):
        self.df = self.volume.volume_oscillator_signal_generation()
        self.df = self.volume.vroc_signal_generation()
        self.df = self.volume.mfi_signal_generation()
        return self.df

    def generate_all_signals(self):
        self.generate_trend_signals()
        self.generate_momentum_signals()
        self.generate_volatility_signals()
        self.generate_volume_signals()
        self.calculate_atr()  # used for equity based position sizing
        self.df.dropna(inplace=True)
        return self.df

    def calculate_atr(self, period=14):
        df = self.df
        df['Previous Close'] = df['Close'].shift(1)
        df['High-Low'] = df['High'] - df['Low']
        df['High-PrevClose'] = abs(df['High'] - df['Previous Close'])
        df['Low-PrevClose'] = abs(df['Low'] - df['Previous Close'])
        df['TR'] = df[['High-Low', 'High-PrevClose',
                       'Low-PrevClose']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=period).mean()
        df.drop(['Previous Close', 'High-Low', 'High-PrevClose',
                 'Low-PrevClose', 'TR'], axis=1, inplace=True)
        df.dropna(inplace=True)
        return df
