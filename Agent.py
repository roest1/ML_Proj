from dataclasses import dataclass
import pandas as pd

@dataclass
class Agent:
    '''
    Agent Class

    In charge of managing percept sequence
    Build a rational agent
    
    '''
    state: list# current state of agent
    fn: function # function mapping percept sequence to action
    fn_params: dict # action Agent performs is updating these parameters
    percept_sequence: pd.DataFrame # Historical observation sequence of data

