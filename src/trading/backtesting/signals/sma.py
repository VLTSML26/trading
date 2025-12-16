import numpy as np
import pandas as pd
from .base import BaseSignal, TypeSignal
from trading.portfolio import Portfolio

class SMASignal(BaseSignal):

    def __init__(self, ptf: Portfolio, ma_short: int=20, ma_long: int=50):
        super().__init__(ptf)
        self.ma_short = ma_short
        self.ma_long = ma_long
    
    def __repr__(self):
        return f"SMA({self.ma_short}, {self.ma_long})"

    @property
    def short_window(self) -> pd.Series:
        serie = self.bh.rolling(self.ma_short).mean()
        serie.name = "SMAShortWindow"
        return serie
        
    @property
    def long_window(self) -> pd.Series:
        serie = self.bh.rolling(self.ma_long).mean()
        serie.name = "SMALongWindow"
        return serie
    
    @property
    def spread(self) -> pd.Series:
        serie = self.short_window / self.long_window - 1.
        serie.name = "SMASpread"
        return serie
    
    @property
    def short_slope(self) -> pd.Series:
        serie = self.short_window.diff()
        serie.name = "SMAShortSlope"
        return serie
    
    @property
    def long_slope(self) -> pd.Series:
        serie = self.long_window.diff()
        serie.name = "SMALongSlope"
        return serie
    
    def compute_signal(self) -> TypeSignal:
        position = np.where(self.short_window > self.long_window, 1, -1)
        return pd.Series(position, index=self.bh.index).shift(1).fillna(0)
