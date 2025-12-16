import numpy as np
import pandas as pd
from .base import BaseSignal, TypeSignal
from trading.portfolio import Portfolio

class MomentumSignal(BaseSignal):

    def __init__(self, ptf: Portfolio, window: int=20):
        super().__init__(ptf)
        self.window = window

    def __repr__(self):
        return f"Momentum({self.window})"
    
    @property
    def momentum(self) -> pd.Series:
        serie = self.ptf.daily_returns.rolling(self.window).mean()
        serie.name = "Momentum"
        return serie
    
    def compute_signal(self) -> TypeSignal:
        position = np.where(self.momentum > 0, 1, -1)
        return pd.Series(position, index=self.ptf.bh.index).shift(1).fillna(0)
