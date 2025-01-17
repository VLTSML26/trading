import numpy as np
from abc import ABC, abstractmethod
from mytrading import TradingBackTester

class Optimizer(ABC):
    def __init__(self, trading_method: TradingBackTester):
        self._trading_method = trading_method
    
    @abstractmethod
    def _optimize(self) -> None:
        pass

class KFold(Optimizer):
    def __init__(self, backtester: TradingBackTester):
        super.__init__(backtester)

    def _optimize(self) -> None:
        from sklearn.model_selection import KFold
        