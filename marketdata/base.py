import asyncio
import aiohttp
from datetime import datetime
import pandas as pd
from typing import Any
from abc import ABC, abstractmethod

# TODO 1: implementare logging.
# TODO 2: implementare rate limiting.
class BaseProvider(ABC):
    """
    Classe astratta per provider di dati di mercato.
    """
    def __init__(self, timeout: float=10., retry: int=3):
        """
        Costruttore di BaseProvider.
        Ogni downlader funziona con logiche di retry e timeout.
        
        :param timeout: Timeout per la richiesta HTTP.
        :type timeout: float
        :param retry: Numero di tentativi in caso di failure.
        :type retry: int
        """
        self.timeout = timeout
        self.retry = retry

    @abstractmethod
    async def get_ticker(
        self,
        session: aiohttp.ClientSession,
        ticker: str,
        time_params: dict[datetime, datetime]
    ) -> pd.Series:
        """
        Scarica i dati per un singolo ticker.
        """
        pass

    async def download_async(
        self,
        tickers: list[str],
        **kwargs
    ) -> pd.DataFrame:
        """
        Scarica tutto l'archivio storico richiesto.
        """
        # estrazione date di inizio e fine serie storica richiesta
        start, end = self.from_to(**kwargs)
        time_params = {
            "start_date": start,
            "end_date": end
        }

        # esecuzione delle richieste asincrone in parallelo
        async with aiohttp.ClientSession() as session:
            tasks = [self.get_ticker(session, ticker, time_params) for ticker in tickers]
            results = await asyncio.gather(*tasks)

        # combinazione dei risultati in un unico DataFrame
        return pd.concat(results, axis=1).sort_index()
    
    def from_to(self, **kwargs):
        """
        Estrae i parametri delle richieste da inoltrare all'API e li rende compatibili con
        lo standard della classe, ovvero una data di inizio e una di fine.

        Nota: se viene specificato solo il periodo o solo la data di inizio, la data di fine sarà oggi.
        """
        # estrazione dai kwargs
        period = kwargs.get("period")
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")

        # controllo sui valori passati in input al metodo
        if period and (start_date or end_date):
            raise ValueError("Specify either 'period' or 'start_date'/'end_date', not both.")
        if not period and not start_date:
            raise ValueError("Must specify either 'period' or at least 'start_date'.")
        
        # definizione delle date di inizio e fine
        if period:
            start_date_dt = self.period_to_start_date(period)
            end_date_dt = pd.Timestamp.today()
        else:
            """
            # NOTE
            Non è necessario implementare funzioni che escludano i giorni di chiusura del mercato, in
            quanto il provider restituisce i dati a partire dal primo giorno di apertura antecedente
            alla data richiesta e fino all'ultimo giorno di apertura antecedente alla data di fine richiesta.
            """
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp.today()
        
        return start_date_dt, end_date_dt

    def period_to_start_date(self, period: str) -> datetime:
        """
        Converte un periodo ('1y', '6mo', '5y', '3d', etc.) nella data di inizio.
        
        :param period: Periodo di coverage richiesto al provider.
        :type period: str
        :return: Data di inizio corrispondente al periodo richiesto.
        :rtype: datetime | None
        """
        period = period.lower()
        today = pd.Timestamp.today()

        """
        # NOTE
        Per i giorni si usa pd.Timedelta, mentre per mesi e anni si usa pd.DateOffset perché
        pd.DateOffset gestisce correttamente mesi/anni (inclusi anni bisestili), mentre pd.Timedelta
        è adatto solo per intervalli di giorni.
        """
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
        
    async def http_request(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        query_params: dict
    ) -> Any:
        """
        Effettua una richiesta HTTP GET asincrona con retry e timeout.
        
        :param session: Sessione aiohttp.
        :type session: aiohttp.ClientSession
        :param endpoint: Endpoint della richiesta.
        :type endpoint: str
        :param query_params: Parametri della query: chiave API, from/to, ticker.
        :type query_params: dict
        :return: Risposta (JSON oppure None se fallisce).
        :rtype: Any
        """
        # validazione minima dell'endpoint
        if not endpoint.startswith("http"):
            raise ValueError("Endpoint must be a valid URL.")

        for attempt in range(self.retry):
            try:
                # configurazione del timeout
                timeout_config = aiohttp.ClientTimeout(total=self.timeout)

                # esecuzione della richiesta
                async with session.get(
                    endpoint,
                    params=query_params,
                    timeout=timeout_config
                ) as response:
                    # controllo dello stato dell'errore
                    response.raise_for_status()
                    return await response.json()
            
            # gestione degli errori
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                if attempt == self.retry - 1:
                    raise e
                continue
                await asyncio.sleep(2**attempt) # backoff esponenziale

    def download(self, *args, **kwargs):
        """
        Wrapper sincrono per il download dei dati dai providers.
        Funzione effettivamente chiamata dalle librerie esterne.
        """
        return asyncio.run(self.download_async(*args, **kwargs))
