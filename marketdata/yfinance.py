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

import aiohttp
import asyncio
import pandas as pd
import yfinance as yf
from datetime import datetime
from functools import partial
from .base import BaseProvider
from concurrent.futures import ThreadPoolExecutor

class YFinanceProvider(BaseProvider):
    """
    Provider basato su Yahoo Finance tramite libreria yfinance.
    Implementa il metodo get_ticker asincrono usando i download sincroni di yfinance (non direttamente API).
    """

    def __init__(self, max_workers: int=10, **kwargs):
        """
        Costruttore per YFinanceProvider.
        Non usa direttamente API REST, ma un executor che sfrutta i download sincroni di yfinance.
        
        :param max_workers: Numero massimo di thread per l'executor.
        :type max_workers: int
        """
        super().__init__(**kwargs)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def get_ticker(
        self,
        session: aiohttp.ClientSession,
        ticker: str,
        time_params: dict[datetime, datetime],  
    ) -> pd.Series:
        """
        Metodo per il download dei dati di mercato di un singolo ticker da Yahoo Finance.
        Standardizza l'output come una Series di pandas con i prezzi di chiusura.
        
        :param session: Sessione HTTP aiohttp.ClientSession.
        :type session: aiohttp.ClientSession
        :param ticker: Ticker richiesto.
        :type ticker: str
        :param time_params: Dizionario con chiavi "start_date" e "end_date" per la serie storica richiesta.
        :type time_params: dict[datetime, datetime]
        :return: Prezzi di chiusura del ticker richiesto.
        :rtype: Series
        """
        # uso functools.partial per creare una funzione parziale di yf.download con i parametri fissi
        download_partial = partial(
            yf.download,
            ticker,
            start=time_params.get("start_date").strftime("%Y-%m-%d"),
            end=time_params.get("end_date").strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True
        )

        # creo loop ed eseguo i download sincroni con i workers dell'executor
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(self.executor, download_partial)

        # standardizzo l'output come Series con i prezzi di chiusura e restituisco
        series = df["Close"]
        series.name = ticker
        return series.dropna()
