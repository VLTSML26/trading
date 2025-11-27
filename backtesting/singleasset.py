import pandas as pd
import numpy as np
from .baseclass import BaseBackTester
from portfolio import Tickers
from .wrappers import single_asset_portfolio

class SMA(BaseBackTester):
    """
    Classe di backtesting della strategia di trading a media mobile (Simple Moving Average).
    Eredita dal BaseBackTester implementando il metodo strategy che va short quando la media
    mobile di breve periodo supera quella di lungo periodo, indicando un andamento negativo
    consolidato dei prezzi dell'asset.
    """
    def __init__(self, tickers: Tickers=None, short: int=20, long: int=50):
        """
        Costruttore della classe SMA.
        
        :param tickers: Oggetto Tickers contenente l'asset su cui testare la strategia SMA.
        :type tickers: Tickers
        :param short: Media mobile su finestra corta.
        :type short: int
        :param long: Media mobile su finestra lunga.
        :type long: int
        """
        super().__init__(single_asset_portfolio(tickers))
        self._short = short
        self._long = long
    
    def __repr__(self):
        return f"SMA({self._short}, {self._long})"

    @property
    def strategy(self):
        if self._strategy is not None:
            return self._strategy

        price = self._ptf._t.df.iloc[:, 0]
        rets = self._rets

        sma_s = price.rolling(self._short).mean()
        sma_l = price.rolling(self._long).mean()

        pos = np.where(sma_s > sma_l, 1, -1)
        pos = pd.Series(pos, index=price.index).shift(1).fillna(0)

        wealth = (1 + pos * rets).cumprod()
        return wealth

class Momentum(BaseBackTester):

    def __init__(self, tickers: Tickers=None, window: int=20):
        """
        Costruttore della classe Momentum.
        
        :param tickers: Oggetto Tickers contenente l'asset su cui testare la strategia SMA.
        :type tickers: Tickers
        :param window: Finestra temporale per il calcolo del momentum.
        :type window: int
        """
        super().__init__(single_asset_portfolio(tickers))
        self._window = window

    def __repr__(self):
        return f"Momentum({self._window})"

    @property
    def strategy(self):
        if self._strategy is not None:
            return self._strategy

        rets = self._rets
        mom = rets.rolling(self._window).mean()

        pos = np.where(mom > 0, 1, -1)
        pos = pd.Series(pos, index=rets.index).shift(1).fillna(0)

        wealth = (1 + pos * rets).cumprod()
        self._strategy = wealth
        return wealth


class Bollinger(BaseBackTester):

    def __init__(self, tickers: Tickers=None, window: int=20, nsigma: float=2.):
        """
        Costruttore della classe Bollinger.
        
        :param tickers: Oggetto Tickers contenente l'asset su cui testare la strategia SMA.
        :type tickers: Tickers
        :param window: Finestra temporale per il calcolo delle bande di Bollinger.
        :type window: int
        :param nsigma: Numero di deviazioni standard per il calcolo delle bande di Bollinger.
        :type nsigma: float
        """
        super().__init__(single_asset_portfolio(tickers))
        self._window = window
        self._nsigma = nsigma

    def __repr__(self):
        return f"Bollinger({self._window}, {self._nsigma})"

    @property
    def strategy(self):
        if self._strategy is not None:
            return self._strategy

        price = self._ptf._t.df.iloc[:, 0]
        rets = self._rets

        sma = price.rolling(self._window).mean()
        std = price.rolling(self._window).std()

        lower = sma - self._nsigma * std
        upper = sma + self._nsigma * std

        pos = pd.Series(index=price.index, dtype=float)
        pos[price < lower] = 1
        pos[price > upper] = -1
        pos = pos.ffill().fillna(0).shift(1)

        wealth = (1 + pos * rets).cumprod()
        self._strategy = wealth
        return wealth
