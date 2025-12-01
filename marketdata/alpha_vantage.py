"""
Module di implementazione della classe AlphaVantageProvider che fornisce dati sui prezzi dei titoli
negoziati sul mercato tramite API e download dal sito Alpha Vantage.

Sviluppato da Samuele Voltan durante e dopo il corso
"Introduction to Portfolio Construction and Analysis with Python" della EDHEC Business School.

Riferimenti:
- https://www.alphavantage.co
"""

import os
import aiohttp
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
from .base import BaseProvider

load_dotenv() # carica variabili d'ambiente da .env

# TODO 1: implementare download asincroni
# TODO 2: implementare download anche di prezzi di apertura, intraday (abbonamento free non so...)
class AlphaVantageProvider(BaseProvider):
    """
    Provider per Alpha Vantage.
    Legge i dati storici tramite API REST.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str=None, timeout: float=5.0, retry: int=3):
        """
        Costruttore di AlphaVantageProvider.
        
        :param api_key: Chiave API (default=None, viene letta la variabile d'ambiente e solleva errore se non trovata).
        :type api_key: str
        :param timeout: Timeout per la richiesta HTTP.
        :type timeout: float
        :param retry: Numero di tentativi in caso di failure.
        :type retry: int
        """
        super().__init__(timeout=timeout, retry=retry)
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Missing Alpha Vantage API key.")

    async def get_ticker(
        self,
        session: aiohttp.ClientSession,
        ticker: str,
        time_params: dict[datetime, datetime]
    ) -> pd.Series:
        """
        Metodo per il download tramite chiamata API dei dati di mercato di un singolo ticker da FMP.
        Standardizza l'output come una Series di pandas con i prezzi di chiusura.
        
        :param ticker: Ticker richeisto.
        :type ticker: str
        :param start: Data di inizio serie storica richiesta.
        :type start: datetime
        :param end: Data di fine serie storica richiesta.
        :type end: datetime
        :return: Prezzi di chiusura del ticker richiesto.
        :rtype: Series
        """
        # parametri standardizzati per le richieste HTTP (Aplha Vantage non supporta filtri di data, li applicherò dopo)
        params = {
            "function": "TIME_SERIES_DAILY", # TODO: poi implementare anche altre frequenze
            "symbol": ticker,
            "apikey": self.api_key
        }
        
        # download dei dati tramite richiesta HTTP
        json_data = await self.http_request(session, self.BASE_URL, params)
        # TODO: qui migliorare l'error handling
        if not json_data:
            raise RuntimeError(f"Download failure for ticker {ticker} from FMP.")
        
        # conversione in DataFrame pandas
        time_series = json_data["Time Series (Daily)"]
        close_prices = pd.Series({date: float(values["4. close"]) for date, values in time_series.items()})
        close_prices.index = pd.to_datetime(close_prices.index)
        
        # filtro le date richieste
        from_date = time_params.get("start_date").strftime("%Y-%m-%d")
        to_date = time_params.get("end_date").strftime("%Y-%m-%d")
        
        # estrazione della serie dei prezzi di chiusura
        return close_prices.rename(ticker).sort_index().loc[from_date:to_date]
