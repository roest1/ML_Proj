import yfinance as yf
from CalculateTechnicals import TrendIndicators


class HeikinAshiAgent:
   
    def __init__(self, df, params):
        self.df = TrendIndicators(df)
        self.params = params

    def generate_signal(self):
        stop_loss = self.params.get('stop_loss', 3)
        self.df.heikin_ashi_signal_generation(stop_loss=stop_loss)
        return self.df['HA signals']

    def adjust_parameters(self, market_conditions):
       pass
      # return self.params

class ParabolicSARAgent:
    def __init__(self, df, params):
        self.df = TrendIndicators(df)
        self.params = params

    def generate_signal(self):
        initial_af = self.params.get('initial_af', 0.02)
        step_af = self.params.get('step_af', 0.02)
        end_af = self.params.get('end_af', 0.2)
        self.df.parabolic_sar_signal_generation(
            initial_af=initial_af, step_af=step_af, end_af=end_af)
        return self.df['sar signals']

    def adjust_parameters(self, market_conditions):
        pass
        #return self.params

class MACDAgent:
    def __init__(self, df, params):
        self.df = TrendIndicators(df)
        self.params = params

    def generate_signal(self):
        ma_type = self.params.get('ma_type', 'ema')
        ma1 = self.params.get(ma1, 12)
        ma2 = self.params.get(ma2, 26)
        signal = self.params.get(signal, 9)
        self.df.macd_signal_generation(ma_type, ma1, ma2, signal)
        return self.df['macd signals']

    def adjust_parameters(self, market_conditions):
        pass # return self.params

class GDAgent:
    def __init__(self, df, params):
        self.df = TrendIndicators(df)
        self.params = params
    
    def generate_signal(self):
        short_window = self.params.get('short_window', 50)
        long_window = self.params.get('long_window', 200)
        ma_type = self.params.get('ma_type', 'sma')
        self.df.golden_death_cross_signal_generation(short_window, long_window, ma_type)
        return self.df['golden death cross signal']

    def adjust_parameters(self, market_conditions):
        pass # return self.params
        

class TrendIndicatorAgent:
    def __init__(self, df, indicator_agents):
        self.df = df
        self.indicator_agents = indicator_agents  

    def generate_signals(self):
        signals = {}
        for agent_name, agent in self.indicator_agents.items():
            signals[agent_name] = agent.generate_signal()
        return signals

    def adjust_all_parameters(self, market_conditions):
        for agent_name, agent in self.indicator_agents.items():
            agent.adjust_parameters(market_conditions)

    def aggregate_signals(self):
        """
        Combine signals from all indicators (e.g., weighted voting or majority vote)
        """
        signals = self.generate_signals()
        combined_signal = 0  # Aggregated signal: +1 for buy, -1 for sell, 0 for hold

        # Simple majority vote aggregation
        for sig in signals.values():
            # Example: Use the last signal value for decision
            combined_signal += sig.iloc[-1]

        if combined_signal > 0:
            return 'buy'
        elif combined_signal < 0:
            return 'sell'
        else:
            return 'hold'

def main():
    df = yf.download("NVDA")

    ha_params = {'stop_loss': 3}
    ha_agent = HeikinAshiAgent(df, ha_params)

    sar_params = {'initial_af': 0.02, 'step_af': 0.02, 'end_af': 0.2}
    sar_agent = ParabolicSARAgent(df, sar_params)

    macd_params = {'ma_type': 'ema', 'ma1': 12, 'ma2': 26, 'signal': 9}
    macd_agent = MACDAgent(df, macd_params)

    gd_params = {'short_window': 50, 'long_window': 200, 'ma_type': 'sma'}
    gd_agent = GDAgent(df, gd_params)

    trend_agent = TrendIndicatorAgent([ha_agent, sar_agent, macd_agent, gd_agent])

if __name__ == '__main__':
    main()