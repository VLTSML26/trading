import numpy as np
import pandas as pd
from baseclass import Strategy
from portfolio import Portfolio
from matplotlib import pyplot as plt, axes; plt.style.use('ggplot')

class CPPI(Strategy):
    """
    # TODO: docstrings
    """
    _DEFAULT = {
        'M': 3,
        'rf': 0.03,
        'floor': 0.8,
        'type': 'static',
        'rebalance': 'W'
    }

    def __init__(self, ptf: Portfolio, par = None):
        super().__init__(ptf)
        par = {**self._DEFAULT, **(par or {})}
        self._m = par['M']
        self._rf = par['rf']
        self._daily_rf = np.expm1(np.log1p(self._rf)*(1/self._ndays))
        self._rebalance = par['rebalance']
        self._type = par['type']
        self._floor = par['floor']

    def __repr__(self):
        _m = "M=" + str(self._m)
        _r = "r=" + str(self._rf*100) + "%"
        _d = str(self._rebalance) + "-CPPI"
        _f = self._type + " " + str(self._floor*100) + "%"
        return _d + " (" + _m + ", " + _r + ", " + _f + ")"
    
    @property
    def _rebalancing(self) -> pd.DatetimeIndex:
        """
        Attributo di tipo pandas.DatetimeIndex che seleziona all'interno delle date
        della serie storia solamente quelle dove avviene il ribilanciamento del portafoglio
        tra comparto rischioso e comparto sicuro.

        Note
        ----
        Al fine di funzionare correttamente, devono essere campionate le date in cui
        i mercati erano aperti, ovvero per le quali sono disponibili i rendimenti del portafoglio.
        Questo viene ottenuto tramite la funzione adjust documentata sotto.
        """
        # parametro di configurazione del CPPI fornisce cadenza temporale -> campionamento date
        if isinstance(self._rebalance, str):
            # se la riallocazione è giornaliera evito di mettere in piedi il circo
            if self._rebalance == 'D':
                return self._bh.index
            else:
                def adjust(missing_dates, wider_index):
                    """
                    Funzione di supporto che permette di scartare le date campionate che non sono
                    presenti nella serie storica (mercati chiusi) e sostituirle con le più vicine
                    date disponibili.
                    """
                    adjusted_dates = []
                    for date in missing_dates:
                        if date in wider_index:
                            # se mercato aperto...
                            adjusted_dates.append(date) # ...mantieni data originale
                        else:
                            # se mercato chiuso trova la data più vicina...
                            closest_idx = wider_index.get_indexer([date], method='nearest')[0]
                            closest_date = wider_index[closest_idx]
                            adjusted_dates.append(closest_date) # ...e sostituisci tale data con la nuova
                    # restituisce un nuovo pandas.DatetimeIndex
                    return pd.DatetimeIndex(adjusted_dates)
                return adjust(
                    self._bh.resample(self._rebalance).sum().index,
                    self._bh.index
                )
        else:
            raise TypeError("Configuration parameter 'rebalance' must be a string.")

    @property
    def strategy(self):
        if self._strategy is None:
            self._strategy = self.cppi_strategy()
        return self._strategy

    def cppi_strategy(self):
        """
        Metodo che implementa la strategia Constant Proportion Portfolio Insurance.
        """
        # initial settings
        floor = self._floor # floor value (constant if type = static, else updated in loop)
        account_value = 1. # initial wealth value
        peak = 1. # if needed, initial peak is wealth value
        cushion = 1. - floor/account_value
        risk_w = np.maximum(np.minimum(self._m*cushion, 1), 0)
        risk_alloc = account_value*risk_w
        safe_alloc = account_value-risk_alloc

        # support lists to iterate to
        val_l = []
        floor_l = []

        for date in self._bh.index:
            # computation of propension to risk
            if self._type == 'max dd':
                # reset of floor value only in max dd floor CPPI
                peak = np.maximum(peak, account_value)
                floor = peak*self._floor

            # rebalancing of portfolio
            if date in self._rebalancing:
                cushion = 1 - floor/account_value # new cushion
                risk_w = np.maximum(np.minimum(self._m*cushion, 1), 0) # new risk allocation
                risk_alloc = account_value*risk_w
                safe_alloc = account_value-risk_alloc
            
            # update wealth according to risk-free and asset returns
            risk_alloc = risk_alloc*(1+self._rets[date])
            safe_alloc = safe_alloc*(1+self._daily_rf)
            account_value = risk_alloc + safe_alloc

            # append to support lists
            val_l += [account_value]
            floor_l += [floor]

        # returns tuple of wealth value for each date and floor value at that date
        vals = pd.Series(val_l, index=self._bh.index)
        floors = pd.Series(floor_l, index=self._bh.index)
        return vals, floors

    def plot_floor(self):
        self.strategy()[1].plot(label="CPPI floor")

    def plot_strategy_floor_rebalance(self, ax=None):
        """
        Plot che mostra l'andamento della strategia, indicando il cuscino di rischio
        allocato giornalmente e le date di ribilanciamento. Viene anche mostrato
        l'andamento della strategia equivalente Buy and Hold, ovvero il rendimento del
        portafoglio titoli se allocato 100% nel comparto rischioso.
        """
        vals, floor = self.strategy
        
        # figure settings
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_xlabel('Date')
        ax.set_ylabel('Return')
        ax.set_title(self.__repr__())

        # plots
        if self._rebalance != 'D':
            ax.scatter(x=self._rebalancing, y=vals[self._rebalancing], marker='o', c='k', label='Reallocations')
        ax.fill_between(x=vals.index, y1=vals.values, y2=floor.values, color='gray', alpha=.5, label='Risk cushion')
        ax.plot(vals, color='coral', label='Strategy')
        ax.plot(floor, color='gray')
        ax.plot(self._bh, c='darkturquoise', label='Portfolio return')

        ax.legend()
        return ax
