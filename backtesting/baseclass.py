import pandas as pd
from portfolio import Portfolio
from .wrappers import cache_plot_once_per_figure
from typing import Union
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt, axes; plt.style.use('ggplot')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class BaseBackTester(ABC):
    """
    Classe astratta unificata per il backtesting di strategie su portafoglio
    e singoli titoli. I singoli titoli sono trattati come oggetti Portfolio.
    """
    called = False

    def __init__(self, ptf: Portfolio):
        """
        Costruttore della classe BaseBackTester
        
        :param ptf: Oggetto Portfolio contenente il portafoglio su cui testare la strategia.
        :type ptf: Portfolio
        """
        self._ptf = ptf
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
        return self._ptf.ptf_comp_returns.dropna()+1

    @property
    def _rets(self) -> pd.Series:
        """
        Valore giornaliero dei rendimenti del portafoglio.
        """
        return self._ptf.ptf_daily_returns.dropna()
    
    @property
    def _ndays(self) -> int:
        return len(self._bh.index)
    
    @property
    @abstractmethod
    def strategy(self) -> Union[pd.Series, tuple]:
        """
        Metodo astratto per implementazione delle varie strategie di trading.
        
        :return: Serie storica dei rendimenti della strategia
        :rtype: Series[Any] | tuple
        """
        pass

    def plot(self, *args, **kwargs):
        """
        Metodo per la visualizzazione grafica dell'andamento della strategia.
        """
        plotter = StrategyPlotter(self) # chiamata al plotter dove vengono demandate le funzioni grafiche
        plotter.plot(*args, **kwargs)

class StrategyPlotter:
    called = False

    def __init__(self, backtester: BaseBackTester):
        self._backtester = backtester

    @classmethod
    def reset(cls):
        cls.called = False

    # il decoratore si rende utile siccome spesso ho riscontrato che questo metodo viene
    # chiamato più volte nella stessa Figure, così evito legende lunghissime
    @cache_plot_once_per_figure
    def plot_bh(self, *args, **kwargs) -> axes.Axes:
        """
        Metodo di visualizzazione grafica della serie storica dei prezzi del titolo.
        Corrisponde a una strategia semplice buy&hold.

        :return: Restituisce oggetto Axes per plottare più grafici sulla stessa figura
        :rtype: Axes
        """
        return self._backtester._bh.plot(label='Portfolio return', *args, **kwargs)

    def plot_strategy(self, *args, **kwargs) -> axes.Axes:
        """
        Metodo di visualizzazione grafica dei rendimenti della strategia adottata.
        
        :return: Restituisce oggetto Axes per plottare più grafici sulla stessa figura
        :rtype: Axes
        """
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
