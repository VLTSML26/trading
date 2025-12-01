"""
Module di implementazione della classe YFinanceProvider che fornisce dati sui prezzi dei titoli
negoziati sul mercato tramite l'utilizzo della libreria yfinance.
I dati sono forniti da Yahoo Finance.

Sviluppato da Samuele Voltan durante e dopo il corso
"Introduction to Portfolio Construction and Analysis with Python" della EDHEC Business School.

Riferimenti:
- https://finance.yahoo.com/
- https://ranaroussi.github.io/yfinance/
"""

import yfinance as yf
import pandas as pd
from typing import Optional
from .base import BaseProvider

class YFinanceProvider(BaseProvider):
    """
    Provider basato su Yahoo Finance tramite libreria yfinance.
    """
    def download(
        self,
        tickers: list[str],
        period: Optional[str]=None,
        start_date: Optional[str]=None,  
        end_date: Optional[str]=None
    ) -> pd.DataFrame:
        """
        Scarica i prezzi di chiusura per uno o più ticker da Yahoo Finance tramite yfinance.
        
        Può essere usato in due modi:
            1. Specificando `period` (es. '1y', '6mo')
            2. Specificando `start_date` e `end_date` (es. '2024-01-01')

        :param tickers: Lista di nomi dei titoli.
        :type tickers: list[str]
        :param period: Periodo di riferimento.
        :type period: Optional[str]
        :param start_date: Data di inizio periodo di riferimento.
        :type start_date: Optional[str]
        :param end_date: Data di fine periodo di riferimento.
        :type end_date: Optional[str]
        :return: pandas DataFrame indicizzato con date.
        :rtype: DataFrame
        """
        # controllo dell'input (si potrebbe anche non fare se invoco sempre da Tickers... ma meglio farlo)
        if period and (start_date or end_date):
            raise ValueError("Specify either 'period' or 'start_date'/'end_date', not both.")
        if not period and not start_date:
            raise ValueError("Must specify either 'period' or at least 'start_date'.")
        
        tickers = tickers if isinstance(tickers, list) else [tickers]

        if period:
            df = yf.download(tickers, period=period, auto_adjust=True, group_by="ticker")
        else:
            # siccome yfinance usa la end_date in maniera esclusiva, la incremento di un giorno
            if end_date:
                end_date = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, group_by="ticker")
        
        # caso singolo asset: df è un MultiIndex diverso
        if len(tickers) == 1:
            df.columns = df.columns.droplevel(0)  # rimuovi livello ticker
            return df["Close"].to_frame(tickers[0]).dropna()

        # multi-asset
        closes = df.loc[:, pd.IndexSlice[:, "Close"]]
        closes.columns = [c[0] for c in closes.columns]
        return closes.dropna()
