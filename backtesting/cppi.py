"""
Module contenente le classi di backtesting per strategie su portafogli complessi.
Le strategie implementate includono:
- Constant Proportion Portfolio Insurance (CPPI)

Sviluppato da Samuele Voltan durante e dopo il corso
"Introduction to Portfolio Construction and Analysis with Python" della EDHEC Business School.

Riferimenti:
- https://www.edhec.edu/en
- https://www.coursera.org/learn/introduction-portfolio-construction-python
"""

import numpy as np
import pandas as pd
from .baseclass import BaseBackTester
from portfolio import Portfolio
from matplotlib import pyplot as plt; plt.style.use('ggplot')

class CPPI(BaseBackTester):
    """
    Classe di backtesting della strategia di portfolio insurance CPPI.
    """
    def __init__(self, ptf: Portfolio, m: int=3, rf: float=.03, floor: float=.8, type: str='static', reb: str='W'):
        """
        Costruttore di CPPI.
        
        :param ptf: Portfolio su cui implementare la strategia.
        :type ptf: Portfolio
        :param m: Moltiplicatore di rischio.
        :type m: int
        :param rf: Tasso risk-free adottato.
        :type rf: float
        :param floor: Livello minimo di protezione del portafoglio.
        :type floor: float
        :param type: Tipo di strategia CPPI ('static' o 'dynamic').
        :type type: str
        :param reb: Frequenza di ribilanciamento del portafoglio.
        :type reb: str
        """
        super().__init__(ptf)
        self._m = m
        self._rf = rf
        self._daily_rf = np.expm1(np.log1p(self._rf)*(1/self._ndays))
        self._reb = reb
        self._type = type
        self._floor = floor

    def __repr__(self):
        _m = "M=" + str(self._m)
        _r = "r=" + str(self._rf*100) + "%"
        _f = self._type + " " + str(self._floor*100) + "%"
        return f"{self._type}-CPPI({_m}, {_r}, {_f})"
    
    @property
    def _rebalancing(self) -> pd.DatetimeIndex:
        """
        Metodo di campionamento delle date di ribilanciamento che implementa una funzione (adjust)
        per non incorrere in giorni di mercato chiuso.
        
        :return: Sottoinsieme di date nella serie storica dove avviene il ribilanciamento.
        :rtype: DatetimeIndex
        """
        if isinstance(self._reb, str):
            # se la riallocazione è giornaliera evito di mettere in piedi il circo
            if self._reb == 'D':
                return self._bh.index
            else:
                def adjust(missing_dates, wider_index):
                    """
                    Funzione di supporto che permette di scartare le date campionate che non sono
                    presenti nella serie storica (mercati chiusi) e sostituirle con le più vicine
                    date disponibili.
                    
                    :param missing_dates: Descrizione
                    :param wider_index: Descrizione
                    """
                    # assicura che l'indice sia ordinato e senza duplicati
                    wider_index = pd.DatetimeIndex(sorted(set(wider_index)))
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
                    self._bh.resample(self._reb).apply(lambda x: None).index,
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
        # impostazioni iniziali
        floor = self._floor # valore floor (costante se type = static, altrimenti aggiornato nel loop)
        account_value = 1. # valore del ptf iniziale
        peak = 1. # se necessario, il picco iniziale è il valore del ptf
        cushion = 1. - floor/account_value
        risk_w = np.maximum(np.minimum(self._m*cushion, 1), 0)
        risk_alloc = account_value*risk_w
        safe_alloc = account_value-risk_alloc

        # liste di supporto per l'iterazione
        val_l = []
        floor_l = []

        for date in self._bh.index:
            # calcolo della propensione al rischio
            if self._type == 'max dd':
                # reset del valore floor solo per CPPI max dd
                peak = np.maximum(peak, account_value)
                floor = peak*self._floor

            # ribilanciamento del portafoglio
            if date in self._rebalancing:
                cushion = 1 - floor/account_value # nuovo cuscino
                risk_w = np.maximum(np.minimum(self._m*cushion, 1), 0) # nuova allocazione rischiosa
                risk_alloc = account_value*risk_w
                safe_alloc = account_value-risk_alloc
            
            # aggiornamento del ptf secondo i rendimenti risk-free e dell'asset
            risk_alloc = risk_alloc*(1+self._rets[date])
            safe_alloc = safe_alloc*(1+self._daily_rf)
            account_value = risk_alloc + safe_alloc

            # append alle liste di supporto
            val_l += [account_value]
            floor_l += [floor]

        # restituisce tupla di valore del ptf per ogni data e valore floor a quella data
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
            _, ax = plt.subplots()
        ax.set_xlabel('Date')
        ax.set_ylabel('Return')
        ax.set_title(self.__repr__())

        # plots
        if self._reb != 'D':
            ax.scatter(x=self._rebalancing, y=vals[self._rebalancing], marker='o', c='k', label='Reallocations')
        ax.fill_between(x=vals.index, y1=vals.values, y2=floor.values, color='gray', alpha=.5, label='Risk cushion')
        ax.plot(vals, color='coral', label='Strategy')
        ax.plot(floor, color='gray')
        ax.plot(self._bh, c='darkturquoise', label='Portfolio return')

        ax.legend()
        return ax
