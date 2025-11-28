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
from .base import MarketDataProvider

class YFinanceProvider(MarketDataProvider):
    """
    Provider basato su Yahoo Finance tramite libreria yfinance.
    """
    def download(self, tickers, period):
        tickers = tickers if isinstance(tickers, list) else [tickers]

        df = yf.download(tickers, period=period, auto_adjust=True, group_by="ticker")
        
        # caso singolo asset: df è un MultiIndex diverso
        if len(tickers) == 1:
            df.columns = df.columns.droplevel(0)  # rimuovi livello ticker
            return df["Close"].to_frame(tickers[0]).dropna()

        # multi-asset
        closes = df.loc[:, pd.IndexSlice[:, "Close"]]
        closes.columns = [c[0] for c in closes.columns]
        return closes.dropna()
