"""
Module di implementazione della classe FMPProvider che fornisce dati sui prezzi dei titoli negoziati
sul mercato tramite API e download dal sito FMP.

Sviluppato da Samuele Voltan durante e dopo il corso
"Introduction to Portfolio Construction and Analysis with Python" della EDHEC Business School.

Riferimenti:
- https://financialmodelingprep.com
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
class FMPProvider(BaseProvider):
    """
    Provider per Financial Modeling Prep (FMP).
    Legge i dati storici tramite API REST (endpoint historical-price-full).
    """

    BASE_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"
    DIVIDEND_ADJ_URL = "https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted"

    def __init__(self, api_key: str=None, **kwargs):
        """
        Costruttore di FMPProvider.
        
        :param api_key: Chiave API (default=None, viene letta la variabile d'ambiente e solleva errore se non trovata).
        :type api_key: str
        :param timeout: Timeout per la richiesta HTTP.
        :type timeout: float
        :param retry: Numero di tentativi in caso di failure.
        :type retry: int
        """
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("Missing FMP API key.")

    async def get_ticker(
        self,
        session: aiohttp.ClientSession,
        ticker: str,
        time_params: dict[datetime, datetime]
    ) -> pd.Series:
        """
        Metodo per il download tramite chiamata API dei dati di mercato di un singolo ticker da FMP.
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
        # parametri standardizzati per le richieste HTTP
        params = {
            "symbol": ticker,
            "apikey": self.api_key,
            "from": time_params.get("start_date").strftime("%Y-%m-%d"),
            "to": time_params.get("end_date").strftime("%Y-%m-%d")
        }
        
        # download dei dati tramite richiesta HTTP
        json_data = await self.http_request(session, self.BASE_URL, params)
        # TODO: qui migliorare l'error handling
        if not json_data:
            raise RuntimeError(f"Download failure for ticker {ticker} from FMP.")
        
        # conversione in DataFrame pandas
        df = pd.DataFrame(json_data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        
        # estrazione della serie dei prezzi di chiusura
        return df["close"].rename(ticker)
