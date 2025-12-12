import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime
from .base import BaseProvider
from typing import Optional, Sequence, Mapping, Dict

class FakeProvider(BaseProvider):
    """
    FakeProvider. Simula con un metodo Monte Carlo i rendimenti giornalieri tramite override del metodo download_async.

    I dati OHLCV + Market Cap sono così simulati:
        - Close: prezzo di chiusura segue un moto browniano geometrico
        - Open: prezzo d'apertura uguale al prezzo di chiusura precedente (no pre-market)
        - High and Low: stimati con un rumore a partire dal prezzo di apertura
        - MarketCap: stimata partendo dallo start_price e dai prezzi di chiusura.
    """
    MAX_DEPTH: int = 30 # anni di massima profondità storica simulata

    def __init__(
        self,
        annual_rets: float | Mapping[str, float]=.05,
        annual_vol: float | Mapping[str, float]=.2,
        start_prices: float | Mapping[str, float]=100.,
        base_mkcaps: float | Mapping[str, float]=1e11,
        correlation_matrix: Optional[np.ndarray]=None,
        seed: Optional[int]=13,
        **kwargs
    ):
        """
        Costruttore della classe FakeProvider.
        
        :param annual_rets: Rendimenti annui stimati per ogni ticker (per stimare drift del GBM).
        :type annual_rets: float | Mapping[str, float]
        :param annual_vol: Volatilità annua stimata per ogni ticker (per stimare sigma del GBM).
        :type annual_vol: float | Mapping[str, float]
        :param start_prices: Prezzi di chiusura a t0 per ogni ticker.
        :type start_prices: float | Mapping[str, float]
        :param base_mkcaps: Capitalizzazione alla data di inizio della serie storica simulata.
        :type base_mkcaps: float | Mapping[str, float]
        :param correlation_matrix: Opzionale, matrice di correlazione tra i rendimenti dei diversi titoli chiesti.
            Se fornita, i rendimenti vengono correlati tramite GBM multivariato.
        :type correlation_matrix: np.ndarray
        :param seed: Seed del generatore di numeri casuali.
        :type seed: Optional[int]
        :param kwargs: Elementi di costruzione della classe parent BaseProvider.
        """
        super().__init__(**kwargs)
        self.annual_rets = annual_rets
        self.annual_vol = annual_vol
        self.start_prices = start_prices
        self.base_mkcaps = base_mkcaps
        self.correlation_matrix = correlation_matrix
        self.dt = 252. # giorni di mercato aperto
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _date_index(start: Optional[datetime], end: datetime, max_depth: int) -> pd.DatetimeIndex:
        """
        Utility che costruisce un indice a giorni lavorativi.
        """
        if start is None:
            start = pd.Timestamp(end) - pd.DateOffset(years=max_depth)
        return pd.date_range(start, end, freq="B")

    @staticmethod
    def _to_per_ticker(values: float | Mapping[str, float], tickers: Sequence[str]) -> Dict[str, float]:
        """
        Utility che converte scalare o mapping in dict per ticker (fallback allo stesso valore per i mancanti).
        """
        if isinstance(values, Mapping):
            first = next(iter(values.values()))
            return {t: float(values.get(t, first)) for t in tickers}
        return {t: float(values) for t in tickers}

    @staticmethod
    def _parkinson(
        sigma_d: np.ndarray,
        open_mat: np.ndarray,
        close_mat: np.ndarray
    ) -> tuple[np.ndarray]:
        """
        Implementa il modello di Parkinson di stima della volatilità intraday basata sui prezzi di apertura e chiusura.
        """
        # stima del range di fluttuazione basata sulla volatilità giornaliera
        range_factor = np.sqrt(4 * np.log(2))
        expected_range = range_factor * sigma_d[None, :] * open_mat

        # campionamento fluttuazioni (NOTE: utilizzo di uniform è molto base, si può studiare qualcosa di meglio)
        u = np.random.uniform(0, 1)
        high_mat = np.maximum(open_mat, close_mat) + u * expected_range
        low_mat = np.minimum(open_mat, close_mat) - (1. - u) * expected_range
        low_mat = np.clip(low_mat, 0., None) # evita negativi

        return high_mat, low_mat
    
    async def get_ohlcv(self) -> pd.DataFrame:
        pass

    async def get_marketcap(self) -> pd.DataFrame:
        pass

    async def download_async(self, tickers: list[str], **kwargs) -> pd.DataFrame:
        """
        Override del metodo download async della classe parent per simulare congiuntamente i rendimenti
        dei diversi titoli richiesti secondo la loro correlazione.
        """
        # normalizza i parametri temporali come BaseProvider e crea l'indice
        start, end = self.from_to(**kwargs)
        idx = self._date_index(start, end, self.MAX_DEPTH)
        s0 = self._to_per_ticker(self.start_prices, tickers)
        cap0 = self._to_per_ticker(self.base_mkcaps, tickers)

        """
        # NOTE
        Nota tecnica: il simulatore Monte Carlo riceve in input i rendimenti annui attesi dei titoli
        come rendimenti semplici. Il GBM invece assume che i logaritmi dei rendimenti si muovano di moto
        Browniano. Dunque bisogna attuare le seguenti trasformazioni:
            1 - trasformare i rendimenti annui semplici in rendimenti annui log-continui
            2 - ottenere i drift giornalieri dai rendimenti log-continui
            3 - simulare i rendimenti mediante GBM
            4 - trasformare tali rendimenti in fattori di crescita giornalieri tramite esponenziale
        """

        # passo 1: rendimenti annui log-continui e volatilità annua
        mu_a = self._to_per_ticker(self.annual_rets, tickers)
        mu_a_log = np.array([np.log1p(mu_a[t]) for t in tickers])
        sg_a = self._to_per_ticker(self.annual_vol, tickers)

        # passo 2: drift e volatilità giornalieri
        mu_d = mu_a_log / self.dt
        sigma_d = np.array([sg_a[t] / np.sqrt(self.dt) for t in tickers])

        # matrice di covarianza per GBM
        if self.correlation_matrix is not None:
            diag = np.diag(sigma_d)
            cov_d = diag @ self.correlation_matrix @ diag
        else:
            cov_d = np.diag(sigma_d**2)
        
        # passi 3 e 4: GBM e trasformazione in rendimenti semplici
        r_d_log = self.rng.multivariate_normal(mean=mu_d, cov=cov_d, size=len(idx))
        growth_fct = np.exp(r_d_log)

        # costruzione dei prezzi di apertura e chiusura
        s0_vec = np.array([s0[t] for t in tickers])
        close_mat = np.cumprod(growth_fct, axis=0) * s0_vec[None, :]
        open_mat = np.vstack([s0_vec, close_mat[:-1, :]])

        # costruzione dei prezzi intraday con rumore di Parkinson
        high_mat, low_mat = self._parkinson(sigma_d, open_mat, close_mat)

        # costruzione capitalizzazione
        cap0_vec = np.array([cap0[t] for t in tickers])
        mkcap_mat = cap0_vec[None, :] * (close_mat / s0_vec[None, :])

        # multi index sulle colonne del dataframe
        cols_ohlc = pd.MultiIndex.from_product([tickers, ["open", "high", "low", "close"]])
        mats = np.stack([open_mat, high_mat, low_mat, close_mat], axis=2)
        data = mats.reshape(mats.shape[0], -1)
        ohlc = pd.DataFrame(data, index=idx, columns=cols_ohlc)
        
        cols_mcap = pd.MultiIndex.from_product([tickers, ["marketCap"]])
        mcap = pd.DataFrame(mkcap_mat, index=idx, columns=cols_mcap)

        df = pd.concat([ohlc, mcap], axis=1).sort_index(axis=1)
        return df

