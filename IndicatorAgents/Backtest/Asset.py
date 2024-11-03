'''
THings to track per trade:

Ticker/Asset Symbol: The cryptocurrency being traded (e.g., BTC, ETH).
Buy/Sell Timestamp: The exact time of the buy and sell actions.
Entry Price (Cost Basis): Price at which you bought the asset.
Exit Price: Price at which you sold the asset.
Position Size: Amount of the asset bought (e.g., 0.5 BTC).
Trade Duration: Time difference between buy and sell actions.
Trade Status: Whether the trade is "Open" or "Closed."
Slippage: Difference between the expected price and the actual executed price, due to market volatility.
Commission/Fees: Transaction fees paid to exchanges or brokers for each trade.

Things to track for the portfolio:

Total Portfolio Value: Sum of cash and asset values at current market prices.
Cash Balance: The amount of uninvested cash in the portfolio.
Allocated Capital per Ticker: The amount of money allocated to each asset in the portfolio.
Unrealized Gains/Losses: Profit or loss for positions that are still open.
Realized Gains/Losses: Profit or loss for positions that have been closed.
Exposure: Percentage of portfolio allocated to different assets (e.g., 40% in BTC, 30% in ETH, etc.).
Leverage: If you are using borrowed funds, track the leverage ratio (e.g., 2:1).
Liquidity: Cash reserves that are available for new trades.
Maximum Drawdown: The largest peak-to-trough decline in portfolio value.

Things to track for performance:

Total Returns: Percentage change in the portfolio value since the start date.
Annualized Returns: How much your portfolio would return annually, based on historical data.
Sharpe Ratio: Risk-adjusted return, measuring how much return you’re earning per unit of risk (volatility).
Sortino Ratio: Similar to Sharpe but focuses on downside risk.
Win Rate: Percentage of trades that were profitable.
Profit Factor: Ratio of gross profit to gross loss.
Average Trade Return: Average percentage gain or loss per trade.
Expectancy: The average amount you can expect to win or lose per trade.
Alpha: Excess return of the portfolio over a benchmark (e.g., Bitcoin or Ethereum).
Beta: Measure of the portfolio’s volatility compared to a benchmark.

Risk Metrics:
Position Sizing: How much capital is allocated to each trade based on your risk tolerance.
Stop Loss Level: Predetermined price level at which you’ll exit a losing trade.
Take Profit Level: Predetermined price level at which you’ll lock in profits.
Risk-Reward Ratio: The ratio between potential profit and potential loss for each trade.
Value at Risk (VaR): The maximum loss you can expect to incur over a specified period, with a certain confidence level.
Maximum Allowed Drawdown: The maximum drawdown level you are willing to tolerate before stopping the strategy.

Per technical indicator:
Signal Timestamp: When a buy/sell signal was generated.
Indicator Values at Time of Signal: E.g., RSI, MACD, or other technical indicator values at the time the signal was generated.
Trigger Event: What condition or threshold triggered the signal (e.g., RSI < 30).
Signal Validity: How often your signals have been profitable (historical success rate of each signal).

Try to come up with ways of quantifying market condition patterns

Benchmark with snp return 

Correlation

Portfolio optimization


'''