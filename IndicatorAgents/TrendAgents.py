from CalculateTechnicals import TrendIndicators


class HeikinAshiAgent:
    def __init__(self, df, params):
        self.df = TrendIndicators(df)
        self.params = params
        self.indicator = TrendIndicators(df)

    def generate_signal(self):
        stop_loss = self.params.get('stop_loss', 3)
        self.df.heikin_ashi_signal_generation(stop_loss=stop_loss)
        return self.df['HA signals']

    def adjust_parameters(self, market_conditions):
        # Adjust stop_loss based on volatility (example heuristic)
        volatility = market_conditions.get('volatility', 1)
        if volatility > 1.5:
            self.params['stop_loss'] += 1
        else:
            self.params['stop_loss'] = max(1, self.params['stop_loss'] - 1)
        return self.params


class ParabolicSARAgent:
    def __init__(self, df, params):
        self.df = df
        self.params = params
        # Assuming TrendIndicators has Parabolic SAR logic
        self.indicator = TrendIndicators(df)

    def generate_signal(self):
        initial_af = self.params.get('initial_af', 0.02)
        step_af = self.params.get('step_af', 0.02)
        end_af = self.params.get('end_af', 0.2)
        self.df = self.indicator.parabolic_sar_signal_generation(
            initial_af=initial_af, step_af=step_af, end_af=end_af)
        return self.df['sar signals']

    def adjust_parameters(self, market_conditions):
        # Adjust parameters based on market conditions
        trend_strength = market_conditions.get('trend_strength', 0.5)
        if trend_strength > 0.7:
            self.params['initial_af'] += 0.01
        else:
            self.params['initial_af'] = max(
                0.01, self.params['initial_af'] - 0.01)
        return self.params


class TrendIndicatorAgent:
    def __init__(self, df, indicator_agents):
        self.df = df
        self.indicator_agents = indicator_agents  # List of individual agent classes

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
