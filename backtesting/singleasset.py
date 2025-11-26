import pandas as pd
import numpy as np
from .baseclass import BaseBackTester
from .wrappers import single_asset_portfolio

class SMA(BaseBackTester):
    """
    # TODO: docstring per SMA (Simple Moving Average)
    """
    _DEFAULT = {
        'ticker': 'AAPL',
        'period': '1mo',
        'short': 20,
        'long': 50
    }

    def __init__(self, par=None):
        par = {**self._DEFAULT, **(par or {})}
        # call the wrapper inside the constructor
        super().__init__(single_asset_portfolio(par['ticker'], par['period']))
        self._short = par['short']
        self._long = par['long']
    
    def __repr__(self):
        return f"SMA({self._short},{self._long})"

    @property
    def strategy(self):
        if self._strategy is not None:
            return self._strategy

        price = self.ptf.df.iloc[:, 0]
        rets = self._rets

        sma_s = price.rolling(self._short).mean()
        sma_l = price.rolling(self._long).mean()

        pos = np.where(sma_s > sma_l, 1, -1)
        pos = pd.Series(pos, index=price.index).shift(1).fillna(0)

        wealth = (1 + pos * rets).cumprod()
        # import pdb; pdb.set_trace()
        return wealth

class Momentum(BaseBackTester):
    _DEFAULT = {
        'ticker': 'AAPL',
        'period': '1mo',
        'window': 20
    }

    def __init__(self, par=None):
        par = {**self._DEFAULT, **(par or {})}
        # call the wrapper inside the constructor
        super().__init__(single_asset_portfolio(par['ticker'], par['period']))
        self._window = par['window']

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
    _DEFAULT = {
        'ticker': 'AAPL',
        'period': '1mo',
        'window': 20,
        'nsigma': 2.0
    }

    def __init__(self, par=None):
        par = {**self._DEFAULT, **(par or {})}
        # call the wrapper inside the constructor
        super().__init__(single_asset_portfolio(par['ticker'], par['period']))
        self._window = par['window']
        self._nsigma = par['nsigma']

    def __repr__(self):
        return f"Bollinger({self._window},{self._nsigma})"

    @property
    def strategy(self):
        if self._strategy is not None:
            return self._strategy

        price = self.ptf.df.iloc[:, 0]
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
