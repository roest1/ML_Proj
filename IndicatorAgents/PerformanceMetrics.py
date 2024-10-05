'''
Performance Evaluator Class
================================

Calculate buy and sell signals (finds profit/loss)

Future:

Sharpe Ratio
Sortino Ratio
Maximum Drawdown
Beta
Win Rate
Profit Factor
profit/loss


4 | 4 == 0100 V 0100

4 or 4  0x100 ?= 0x100
'''

class PerformanceEvaluator:
    def __init__(self, df, position_sizer, initial_capital, position_size, position_strategy):
        self.df = df
        self.position_sizer = position_sizer
        self.position_strategy = position_strategy
        self.trading_state = {
            'capital': initial_capital,
            'shares': 0,
            'profits': [],
            'position_size': position_size
        }
    
    def get_buy_sell_coordinates(self, signal_column):
        df = self.df

        trading_state = self.trading_state

        buy_signals_x = []
        buy_signals_y = []
        sell_signals_x = []
        sell_signals_y = []

        # using buy at low and sell at high to account for market slippage 
        # being called on a daily basis so there is multiple days of data between buy and sell signals
        buy_at_ohlc = 'High' 
        sell_at_ohlc = 'Low'
        #buy_at_ohlc = 'Open'
        #sell_at_ohlc = 'Close'


        for i in range(len(df) - 1):
            self.position_sizer.update_capital(trading_state['capital'])
            execution_date = i + 1
            # Buy
            if df[signal_column].iloc[i] == 1:
                # position_size = self.position_sizer.calculate_position_size(
                #     self.position_strategy, df['ATR'].iloc[i], trading_state['profits'])
                num_shares_to_buy = 1
                trading_state['shares'] += num_shares_to_buy
                trading_state['capital'] -= num_shares_to_buy * df[buy_at_ohlc].iloc[execution_date]
                buy_signals_x.append(df.index[execution_date])
                buy_signals_y.append(df[buy_at_ohlc].iloc[execution_date])
            # Sell
            elif df[signal_column].iloc[i] == -1 and buy_signals_y and trading_state['shares'] > 0:
                sell_revenue = trading_state['shares'] * df[sell_at_ohlc].iloc[execution_date]
                trading_state['capital'] += sell_revenue
                sell_signals_x.append(df.index[execution_date])
                sell_signals_y.append(df[sell_at_ohlc].iloc[execution_date])
                trading_state['profits'].append(sell_revenue - (trading_state['shares'] * buy_signals_y[-1]))
                trading_state['shares'] = 0

          
            
            # elif df[signal_column].iloc[i] == 2 
            # elif df[signal_column].iloc[i] == 3: # and so on. maybe we can aggregate signals 

        return buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y, trading_state
    
    # For forecasted data
    def calculate_color_prediction_accuracy(data, predictions):
        def get_color(open_price, close_price):
            return 'green' if open_price <= close_price else 'red'

        actual_colors = [get_color(row['Open'], row['Close'])
                        for _, row in data.iterrows()]

        predicted_colors = [get_color(row['Open'], row['Close'])
                            for _, row in predictions.iterrows()]

        correct_predictions = sum(a == p for a, p in zip(
            actual_colors, predicted_colors))
        total_predictions = len(actual_colors)
        accuracy = correct_predictions / total_predictions

        return accuracy
