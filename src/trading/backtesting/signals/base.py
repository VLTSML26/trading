import pandas as pd
from typing import Annotated
from abc import ABC, abstractmethod
from trading.portfolio import Portfolio

TypeSignal = Annotated[pd.Series, "Elements must be -1 or +1"]

class BaseSignal(ABC):
    """
    Base class for computing a short/long position signal in market data provided by
    a Portfolio object.
    """
    def __init__(self, ptf: Portfolio):
        self.ptf = ptf
        self.cache_signal = None

    @property
    def bh(self) -> pd.Series:
        """
        Andamento del valore del titolo per una strategia buy and hold sul periodo
        di riferimento.
        """
        serie = self.ptf.comp_returns + 1
        serie.name = "BHWealth"
        return serie
    
    @property
    def ndays(self) -> int:
        return len(self.bh.index)
    
    @property
    def signal(self) -> TypeSignal:
        if self.cache_signal is not None:
            return self.cache_signal
        computed_signal = self.compute_signal()
        self.cache_signal = computed_signal
        return computed_signal

    @abstractmethod
    def compute_signal(self) -> TypeSignal:
        pass

    @property
    def wealth(self) -> pd.Series:
        return (1 + self.signal * self.ptf.daily_returns).cumprod()
