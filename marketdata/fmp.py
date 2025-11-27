import os
from dotenv import load_dotenv
import pandas as pd
import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from .base import MarketDataProvider

load_dotenv() # carica variabili d'ambiente da .env

# TODO 1: implementare download asincroni
# TODO 2: implementare download anche di prezzi di apertura, intraday (abbonamento free non so...)
# TODO 3: scrivere funzioni di download che prendono anche date di inizio e fine pre-definite (attualmente fine = oggi)
class FMPProvider(MarketDataProvider):
    """
    Provider per Financial Modeling Prep (FMP).
    Legge i dati storici tramite API REST (endpoint historical-price-full).
    """

    BASE_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"
    DIVIDEND_ADJ_URL = "https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted"

    def __init__(self, api_key: str=None, timeout: float=5.0, retry: int=3):
        """
        Costruttore di FMPProvider.
        
        :param api_key: Chiave API (default=None, viene letta la variabile d'ambiente e solleva errore se non trovata).
        :type api_key: str
        :param timeout: Timeout per la richiesta HTTP.
        :type timeout: float
        :param retry: Numero di tentativi in caso di failure.
        :type retry: int
        """
        self._api_key = api_key or os.getenv("FMP_API_KEY")
        if self._api_key is None:
            raise ValueError("Missing FMP API key.")
        self.timeout = timeout
        self.retry = retry

    def download(self, tickers: list[str], period: str) -> pd.DataFrame:
        """
        Scarica i prezzi di chiusura per uno o più ticker.
        
        :param tickers: Lista di nomi dei titoli.
        :type tickers: list[str]
        :param period: Periodo di riferimento dei prezzi dei titoli.
        :type period: str
        :return: pandas DataFrame indicizzato con date.
        :rtype: DataFrame
        """
        # conversione di period in una data di inizio
        start_date = self._period_to_start_date(period)

        # scarico ogni ticker singolarmente
        all_closes = {}
        for ticker in tickers:
            df_close = self._download_single(ticker, start_date)
            all_closes[ticker] = df_close

        df = pd.concat(all_closes, axis=1) # combino in un unico DataFrame
        return df.sort_index()

    def _download_single(self, ticker: str, start_date: datetime | None) -> pd.Series:
        """
        Download tramite chiamata API dei dati di mercato di un singolo ticker dal provider.
        
        :param ticker: Ticker richeisto.
        :type ticker: str
        :param start_date: Data di inizio serie storica richiesta.
        :type start_date: datetime | None
        :return: Prezzi di chiusura del ticker richiesto.
        :rtype: Series
        """
        params = {
            "symbol": ticker,
            "apikey": self._api_key,
            "from": start_date.strftime("%Y-%m-%d") if start_date else None,
            "to": None # TODO 3: implementare qua
        }

        # tentativi di download dei dati con logiche di retry e timeout
        for _ in range(self.retry):
            try:
                r = requests.get(self.DIVIDEND_ADJ_URL, params=params, timeout=self.timeout)
                # risposta in formato json
                if r.status_code != 200:
                    continue # esco dal for (considerato come un tentativo fallito)
                js = r.json()
                # converto json in DataFrame e indicizzo datetime
                df = pd.DataFrame(js)
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)

                # TODO 2: implementare qua
                try:
                    return df["close"].rename(ticker)
                except KeyError:
                    return df["adjClose"].rename(ticker)

            except Exception:
                continue

        raise RuntimeError(f"Download failure for ticker {ticker} from FMP.")

    def _period_to_start_date(self, period: str) -> datetime | None:
        """
        Converte un periodo ('1y', '6mo', '5y', '3d', etc.) nella data di inizio.
        
        :param period: Periodo di coverage richiesto al provider.
        :type period: str
        :return: Data di inizio corrispondente al periodo richiesto.
        :rtype: datetime | None
        """
        period = period.lower()
        today = pd.Timestamp.today()

        # NOTE: per i giorni si usa pd.Timedelta, mentre per mesi e anni si usa pd.DateOffset.
        # NOTE: Questo perché pd.DateOffset gestisce correttamente mesi/anni (inclusi anni bisestili),
        # NOTE: mentre pd.Timedelta è adatto solo per intervalli di giorni.
        if period == 'max':
            return None
        elif period.endswith("y"):
            years = int(period[:-1])
            return today - pd.DateOffset(years=years)
        elif period.endswith("mo"):
            months = int(period[:-2])
            return today - pd.DateOffset(months=months)
        elif period.endswith("d"):
            days = int(period[:-1])
            return today - pd.Timedelta(days=days)
        else:
            raise ValueError(f"Invalid period format: {period}")
