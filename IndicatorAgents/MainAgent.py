from TrendAgents import HeikinAshiAgent, ParabolicSARAgent, TrendIndicatorAgent
import yfinance as yf
import pandas as pd

df = yf.download("NVDA")


'''
Need to provide reward function
Need to adjust indicator's parameters dynamically
- adaptively optimize this later with reinforcement learning
Need to aggregate all signals and weigh them individually based on market conditions
'''


### Indicator Params - (TODO) Adjusted params dynamically by the agents 
params_heikin_ashi = {'stop_loss': 3}
params_parabolic_sar = {'initial_af': 0.02, 'step_af': 0.02, 'end_af': 0.2}

heikin_ashi_agent = HeikinAshiAgent(df, params_heikin_ashi)
parabolic_sar_agent = ParabolicSARAgent(df, params_parabolic_sar)


trend_agent = TrendIndicatorAgent(df, {
    'heikin_ashi': heikin_ashi_agent,
    'parabolic_sar': parabolic_sar_agent,
})


### (TODO) Add other indicator classes similarly


overall_signals = trend_agent.generate_signals()

# Adjust parameters based on market conditions
market_conditions = {'volatility': 1.8, 'trend_strength': 0.9}
trend_agent.adjust_all_parameters(market_conditions)

# Aggregate signals and make the final decision
final_decision = trend_agent.aggregate_signals()

### (TODO) Add main agent class to aggregate signals from trend, momentum, volatility, and volume indicator agents

print(f"Final Decision: {final_decision}")
