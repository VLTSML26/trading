import pandas as pd
from portfolio import Portfolio
from .wrappers import cache_plot_once_per_figure
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt, axes; plt.style.use('ggplot')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class BaseBackTester(ABC):
    """
    Unified abstract backtesting class for both portfolio and single-asset
    strategies. Single-asset are viewed as Portfolio objects.
    """
    called = False

    def __init__(self, ptf: Portfolio):
        self.ptf = ptf
        self._strategy = None

    @classmethod
    def reset(cls):
        cls.called = False
    
    @property
    def _bh(self) -> pd.Series:
        """
        Andamento del valore del titolo per una strategia buy and hold sul periodo
        di riferimento.
        """
        return self.ptf.ptf_comp_returns.dropna()+1

    @property
    def _rets(self) -> pd.Series:
        """
        Valore giornaliero dei rendimenti del portafoglio.
        """
        return self.ptf.ptf_daily_returns.dropna()
    
    @property
    def _ndays(self) -> int:
        return len(self._bh.index)
    
    @property
    @abstractmethod
    def strategy(self):
        """
        Must return:
        - a pandas Series (cumulative wealth)
        - a tuple (Series wealth, additional outputs)
        """
        pass

    def plot(self, *args, **kwargs):
        plotter = StrategyPlotter(self)
        plotter.plot(*args, *kwargs)

class StrategyPlotter:
    called = False

    def __init__(self, backtester: BaseBackTester):
        self._backtester = backtester

    @classmethod
    def reset(cls):
        cls.called = False

    # FIXME (ok cache plot per la prima volta ma poi diventa sempre assente il buy&hold)
    @cache_plot_once_per_figure
    def plot_bh(self, *args, **kwargs) -> axes.Axes:
        return self._backtester._bh.plot(label='Portfolio return', *args, **kwargs)

    def plot_strategy(self, *args, **kwargs) -> axes.Axes:
        # trycatch qui per evitare di unpackare oggetti non-iterabili (caso di SMA)
        try:
            keep, *_ = self._backtester.strategy
            return keep.plot(label=self._backtester.__repr__(), *args, **kwargs)
        except AttributeError:
            return self._backtester.strategy.plot(label=self._backtester.__repr__(), *args, **kwargs)

    def plot(self, *args, **kwargs):
        self.plot_bh(*args, **kwargs)
        self.plot_strategy(*args, **kwargs)
        plt.title("Returns of buy&hold against trading strategies")
        plt.legend()
