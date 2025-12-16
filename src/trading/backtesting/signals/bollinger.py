import numpy as np
import pandas as pd
from .base import BaseSignal, TypeSignal
from trading.portfolio import Portfolio

class BollingerSignal(BaseSignal):

    def __init__(self, ptf: Portfolio, window: int=20, nsigma: float=2.):
        super().__init__(ptf)
        self.window = window
        self.nsigma = nsigma

    def __repr__(self):
        return f"Bollinger({self.window}, {self.nsigma})"
    
    @property
    def lower_band(self) -> pd.Series:
        sma = self.bh.rolling(self.window).mean()
        std = self.bh.rolling(self.window).std()
        serie = sma - self.nsigma * std
        serie.name = "BollingerLower"
        return serie
    
    @property
    def upper_band(self) -> pd.Series:
        sma = self.bh.rolling(self.window).mean()
        std = self.bh.rolling(self.window).std()
        serie = sma + self.nsigma * std
        serie.name = "BollingerUpper"
        return serie

    def compute_signal(self) -> TypeSignal:
        pos = pd.Series(index=self.bh.index, dtype=float)
        pos[self.bh < self.lower_band] = 1
        pos[self.bh > self.upper_band] = -1
        return pos.ffill().fillna(0).shift(1)
