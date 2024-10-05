import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

class Visualize:

    @staticmethod
    def create_stock_chart(df, open='Open', close='Close', high='High', low='Low', subplot=False):
        if subplot:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3, 0.3])
        else:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df[open],
                high=df[high],
                low=df[low],
                close=df[close],
                name='Stock Chart',
                whiskerwidth=1,
                increasing=dict(line=dict(color='rgba(0, 255, 0, 0.8)'), fillcolor='rgba(0, 255, 0, 0.6)'),  # Green with alpha for increasing
                decreasing=dict(line=dict(color='rgba(255, 0, 0, 0.8)'), fillcolor='rgba(255, 0, 0, 0.6)')   # Red with alpha for decreasing
            ), 
            row=1, col=1
        )
        
        return fig
    
    @staticmethod
    def add_buy_sell_signals(fig, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y):
        arrow_size = 15

        fig.add_trace(go.Scatter(
            x=buy_signals_x, y=buy_signals_y,
            mode='markers', marker_symbol='triangle-up',
            marker_line_color='rgba(0, 255, 0, 1)', marker_color='rgba(0, 255, 0, 1)',
            marker_size=arrow_size, name='Buy Signal'
        ))

        fig.add_trace(go.Scatter(
            x=sell_signals_x, y=sell_signals_y,
            mode='markers', marker_symbol='triangle-down',
            marker_line_color='rgba(255, 0, 0, 1)', marker_color='rgba(255, 0, 0, 1)',
            marker_size=arrow_size, name='Sell Signal'
        ))
    
    @staticmethod
    def add_volume_subplot(fig, df, row, col):
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color='rgba(70, 130, 180, 0.7)'  # Steel blue with alpha
            ),
            row=row, col=col
        )
        fig.update_yaxes(title_text='Volume', showgrid=False, row=row, col=col)
        fig.update_xaxes(showgrid=False, row=row, col=col)

    
    @staticmethod
    def finalize_plot(df, fig, title, position_strategy, profits, subplot_title=None):
        
        fig.update_layout(
            title=f"{title} (Final Profit/Loss with {position_strategy} position sizing = ${format(round(np.sum(profits), 2), ',')})")

        
        if subplot_title:
            fig.update_yaxes(title_text='Price $', row=1, col=1)
            fig.update_yaxes(title_text=subplot_title, showgrid=False, row=2, col=1)
            fig.update_xaxes(title_text='Date', showgrid=False, row=2, col=1)
            Visualize.add_volume_subplot(fig, df, row=3, col=1)
            
        else:
            fig.update_yaxes(title_text='Price $')
            fig.update_xaxes(title_text='Date')
            Visualize.add_volume_subplot(fig, df, row=2, col=1)
            

        fig.update_layout(
            legend_title='Legend',
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white'),
            xaxis=dict(
                rangeslider=dict(visible=False),
                showgrid=False,
                tickfont=dict(color='white')
            ),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(color='white')
            ),
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
        )

        fig.show()




    @staticmethod
    def plot_bollinger_bands(df, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y, position_strategy, profits):
        fig = Visualize.create_stock_chart(df)

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bollinger bands upper band'],
                fill=None,
                mode='lines',
                line_color='rgba(40,50,245,1)', 
                name='Upper Band'
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bollinger bands mid band'],
                mode='lines',
                fill='tonexty',
                line=dict(dash='dash'),
                fillcolor='rgba(40,40,200,0.3)',
                line_color='rgba(245,120,40,1)',
                name='Middle Band'
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bollinger bands lower band'],
                fill='tonexty',
                mode='lines',
                fillcolor='rgba(40,40,200,0.3)',
                line_color='rgba(40,50,245,1)',
                name='Lower Band'
            )
        )
        

        Visualize.add_buy_sell_signals(fig, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y)
        Visualize.finalize_plot(df, fig, 'Bollinger Bands', position_strategy, profits)


    @staticmethod
    def plot_heikin_ashi(df, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y, position_strategy, profits):
        fig = Visualize.create_stock_chart(df, open='HA open', close='HA close', high='HA high', low='HA low')
        Visualize.add_buy_sell_signals(fig, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y)
        Visualize.finalize_plot(df, fig, 'Heikin Ashi', position_strategy, profits)

    @staticmethod
    def plot_macd(df, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y, position_strategy, profits):
        
        fig = Visualize.create_stock_chart(df, subplot=True)

        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd short ma'],
                    mode='lines', name='MACD Line', line=dict(color='blue'), showlegend=False),
                row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd ma1'],
                    mode='lines', name='MACD Line', line=dict(color='blue')),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd long ma'], mode='lines',
                    name='Signal Line', line=dict(color='orange'), showlegend=False),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd ma2'], mode='lines',
                    name='Signal Line', line=dict(color='orange')),
            row=2, col=1
        )

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['macd ma1'] - df['macd ma2'],
                name='Histogram',
                marker=dict(
                    color=np.where(df['macd ma1'] - df['macd ma2'] > 0, 'green', 'red'),
                    line=dict(
                        color=np.where(df['macd ma1'] - df['macd ma2'] > 0, 'green', 'red'),
                        width=1
                    )
                )
            ),
            row=2, col=1
        )
        Visualize.add_buy_sell_signals(fig, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y)
        Visualize.finalize_plot(df, fig, 'MACD', position_strategy, profits, subplot_title='MACD')
    
    @staticmethod
    def plot_sar(df, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y, position_strategy, profits):
        fig = Visualize.create_stock_chart(df, subplot=True)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['sar'], mode='lines', line=dict(color='blue', dash='dot'),
                                name='SAR Line'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['ep'], mode='markers', marker=dict(color='orange', size=5),
                                name='Extreme Points'), row=1, col=1)
        
        Visualize.add_buy_sell_signals(fig, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y)

        fig.add_trace(go.Scatter(x=df.index, y=df['af'], mode='lines', line=dict(color='white'),
                                name='Acceleration Factor'), row=2, col=1)

        
        Visualize.finalize_plot(df, fig, 'Parabolic SAR', position_strategy, profits, subplot_title="AF")

    @staticmethod
    def plot_rsi(df, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y, position_strategy, profits):
        
        fig = Visualize.create_stock_chart(df, subplot=True)
        Visualize.add_buy_sell_signals(fig, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y)
        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi'],
                    mode='lines', name='RSI', line=dict(color='purple')),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi overbought'],
                    mode='lines', name='Overbought', line=dict(color='blue', dash='dot')),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi oversold'],
                    mode='lines', name='Oversold', line=dict(color='orange', dash='dot')),
            row=2, col=1
        )
        Visualize.finalize_plot(df, fig, 'RSI', position_strategy, profits, subplot_title='RSI')

    @staticmethod
    def plot_golden_death_cross(df, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y, position_strategy, profits):
 
        fig = Visualize.create_stock_chart(df)

        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['gd ma_short'], 
                mode='lines', 
                name='golden cross short ma'
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['gd ma_long'],
                mode='lines',
                name='golden cross long ma',
                line=dict(dash='dot')
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index[df['golden_cross_points']],
                y=df['Close'][df['golden_cross_points']],
                mode='markers',
                marker=dict(color='gold', size=50),
                name='Golden Cross'
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index[df['death_cross_points']],
                y=df['Close'][df['death_cross_points']],
                mode='markers',
                marker=dict(color='brown', size=50),
                name='Death Cross'
            )
        )
        
        Visualize.add_buy_sell_signals(fig, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y)
        Visualize.finalize_plot(df, fig, 'Golden/Death Cross', position_strategy, profits)

    @staticmethod
    def plot_stochastic_oscillator(df, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y, position_strategy, profits):
        fig = Visualize.create_stock_chart(df, subplot = True)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['%K'], mode='lines', name='Fast stochastic', line=dict(color='blue')
            ), 
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['%D'], mode='lines', name='Slow stochastic', line=dict(color='red', dash='dot')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['stochastic oscillator overbought'], mode='lines', name='Overbought line', line=dict(color='orange', dash='dash')
            ), 
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['stochastic oscillator oversold'], mode='lines', name='Oversold line', line=dict(color='green', dash='dash')
            ),
            row=2, col=1
        )
        Visualize.add_buy_sell_signals(fig, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y)
        Visualize.finalize_plot(df, fig, 'Stochastic Oscillator', position_strategy, profits, subplot_title='Stochastic Oscillator')

    
    @staticmethod
    def plot_roc(df, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y, position_strategy, profits):
        
        fig = Visualize.create_stock_chart(df, subplot=True)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['roc'], mode='lines', name='ROC', line=dict(color='blue')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['roc overbought'], mode='lines', name='Overbought line', line=dict(color='orange', dash='dash')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['roc oversold'], mode='lines', name='Oversold line', line=dict(color='green', dash='dash')
            ),
            row=2, col=1
        )
        Visualize.add_buy_sell_signals(fig, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y)
        Visualize.finalize_plot(df, fig, 'ROC', position_strategy, profits, subplot_title='ROC')

    @staticmethod
    def plot_williams_r(df, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y, position_strategy, profits):
        fig = Visualize.create_stock_chart(df, subplot=True)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['Williams %R'], mode='lines', name='Williams %R', line=dict(color='blue')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['Williams %R overbought'], mode='lines', name='Overbought line', line=dict(color='orange', dash='dash')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['Williams %R oversold'], mode='lines', name='Oversold line', line=dict(color='green', dash='dash')
            ),
            row=2, col=1
        )
        Visualize.add_buy_sell_signals(fig, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y)
        Visualize.finalize_plot(df, fig, 'Williams %R', position_strategy, profits, subplot_title='Williams %R')

    @staticmethod
    def plot_cci(df, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y, position_strategy, profits):
        
        fig = Visualize.create_stock_chart(df, subplot=True)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['CCI'], mode='lines', name='CCI', line=dict(color='blue')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['CCI overbought'], mode='lines', name='Overbought line', line=dict(color='orange', dash='dash')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['CCI oversold'], mode='lines', name='Oversold line', line=dict(color='green', dash='dash')
            ),
            row=2, col=1
        )
        Visualize.add_buy_sell_signals(fig, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y)
        Visualize.finalize_plot(df, fig, 'CCI', position_strategy, profits, subplot_title='CCI')

    @staticmethod
    def plot_hv(df, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y, position_strategy, profits):
      
        fig = Visualize.create_stock_chart(df, subplot=True)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['historical_volatility'], mode='lines', name='historical volatility', line=dict(color='blue')
            ),
            row=2, col=1
        )
        fig.add_hline(
            y=df['hv threshold'].iloc[0], line_dash="dash", line_color="red",
                      annotation_text="Volatility Threshold", row=2, col=1)
        Visualize.add_buy_sell_signals(fig, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y)
        Visualize.finalize_plot(df, fig, 'Historical Volatility', position_strategy, profits, subplot_title='Historical Volatility')

    @staticmethod
    def plot_sd(df, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y, position_strategy, profits):
        
        fig = Visualize.create_stock_chart(df, subplot=True)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['standard_deviation'], mode='lines', name='standard deviation', line=dict(color='blue')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['sd threshold'], mode='lines', name='sd threshold', line=dict(color='red', dash='dash')
            ),
            row=2, col=1
        )
        Visualize.add_buy_sell_signals(fig, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y)
        Visualize.finalize_plot(df, fig, 'Standard Deviation', position_strategy, profits, subplot_title='standard deviation')

    @staticmethod
    def plot_vol_osc(df, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y, position_strategy, profits):
   
        fig = Visualize.create_stock_chart(df, subplot=True)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['vol_osc'], mode='lines', name='volume oscillator', line=dict(color='blue')
            ),
            row=2, col=1
        )
        Visualize.add_buy_sell_signals(
            fig, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y)
        Visualize.finalize_plot(df,
            fig, 'Volume Oscillator', position_strategy, profits, subplot_title='volume oscillator')

    @staticmethod
    def plot_vroc(df, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y, position_strategy, profits):
       
        fig = Visualize.create_stock_chart(df, subplot=True)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['VROC'], mode='lines', name='Volume ROC', line=dict(color='blue')
            ),
            row=2, col=1
        )
        Visualize.add_buy_sell_signals(fig, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y)
        Visualize.finalize_plot(df, fig, 'Volume Rate of Change', position_strategy, profits, subplot_title='VROC')
    
    @staticmethod
    def plot_mfi(df, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y, position_strategy, profits):
        
        fig = Visualize.create_stock_chart(df, subplot=True)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['MFI'], mode='lines', name='Money Flow Index', line=dict(color='blue')
            ),
            row=2, col=1
        )
        Visualize.add_buy_sell_signals(fig, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y)
        Visualize.finalize_plot(df, fig, 'Money Flow Index', position_strategy, profits, subplot_title="MFI")
    
    @staticmethod
    def plot_ridge_lasso(df, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y, position_strategy, profits):
        
        fig = Visualize.create_stock_chart(df)

        train_size = 0.2
        split_index = int(len(df) * train_size)

        fig.add_vrect(
            x0=df.index[0], x1=df.index[split_index],
            fillcolor="blue", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Training Data", annotation_position="top left"
        )

        fig.add_vrect(
            x0=df.index[split_index], x1=df.index[-1],
            fillcolor="yellow", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Testing Data", annotation_position="top right"
        )
        Visualize.add_buy_sell_signals(fig, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y)
        Visualize.finalize_plot(df, fig, 'Ridge/Lasso Regression', position_strategy, profits)
