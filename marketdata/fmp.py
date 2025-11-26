import os
from dotenv import load_dotenv
import pandas as pd
import requests
from datetime import datetime, timedelta
from .base import MarketDataProvider

load_dotenv() # carica variabili d'ambiente da .env

class FMPProvider(MarketDataProvider):
    """
    Provider per Financial Modeling Prep (FMP).
    Legge i dati storici tramite API REST (endpoint historical-price-full).
    """

    BASE_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"

    def __init__(self, api_key: str = None, timeout: float = 5.0, retry: int = 3):
        """
        Parametri
        ---------
        api_key : str
            Chiave API: None di default. In tal caso, legge la variabile d'ambiente FMP_API_KEY.
            Solleva errore se non trovata.
        timeout : float
            Timeout per la richiesta HTTP.
        retry : int
            Numero di tentativi in caso di failure.
        """
        self._api_key = api_key or os.getenv("FMP_API_KEY")
        if self._api_key is None:
            raise ValueError(
                "Missing FMP API key. Set FMP_API_KEY in your .env file "
                "or pass api_key=\"...\" explicitly."
            )
        self.timeout = timeout
        self.retry = retry

    def download(self, tickers, period):
        """
        Scarica i prezzi di chiusura per uno o più ticker.
        Restituisce un DataFrame: colonne = tickers, index = date.
        """
        tickers = tickers if isinstance(tickers, list) else [tickers] # assicuro lista
        start_date = self._period_to_start_date(period) # converto 'period' in giorni

        # scarico ogni ticker singolarmente
        all_closes = {}
        for ticker in tickers:
            df_close = self._download_single(ticker, start_date)
            all_closes[ticker] = df_close

        df = pd.concat(all_closes, axis=1) # combino in un unico DataFrame
        return df.sort_index()

    def _download_single(self, ticker, start_date):
        """
        Scarica i dati storici per un singolo ticker da FMP.
        Restituisce una Series con i prezzi di chiusura.
        """
        params = {
            "symbol": ticker.upper(),
            "apikey": self._api_key,
            "from": start_date.strftime("%Y-%m-%d")#,
            # "to": datetime.today().strftime("%Y-%m-%d")
        }

        # tentativi di download dei dati con logiche di retry e timeout
        for _ in range(self.retry):
            try:
                # effettuo richiesta
                r = requests.get(self.BASE_URL, params=params, timeout=self.timeout)

                # risposta in formato json
                if r.status_code != 200:
                    continue # esco dal ciclo for (considerato come un tentativo fallito)
                js = r.json()

                # converto json in DataFrame e indicizzo datetime
                df = pd.DataFrame(js)
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)

                # TODO: restituire tutti i prezzi nella logica di sostituire la classe Tickers
                return df["close"].rename(ticker)

            except Exception:
                continue

        raise RuntimeError(f"Download failure for ticker {ticker} from FMP.")

    # FIXME: metodo vecchio non conta bene i giorni
    def _period_to_days(self, period: str) -> int:
        """
        Converte '1y', '6mo', '5y', '3d', etc. in un numero di giorni.
        """
        period = period.lower()

        if "y" in period:
            return int(period.replace("y", "")) * 365
        elif "mo" in period:
            return int(period.replace("mo", "")) * 30
        elif "d" in period:
            return int(period.replace("d", ""))
        else:
            raise ValueError(f"Unrecognized period format: {period}")
        
    
    def _period_to_start_date(self, period: str) -> datetime:
        period = period.lower()
        today = pd.Timestamp.today()

        if period.endswith("y"):
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
