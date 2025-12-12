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

    async def get_ohlcv(self) -> pd.DataFrame:
        pass

    async def get_marketcap(self) -> pd.DataFrame:
        pass

    async def download_async(self, tickers: list[str], **kwargs) -> pd.DataFrame:
        """
        Override del metodo download async della classe parent per simulare congiuntamente i rendimenti
        dei diversi titoli richiesti secondo la loro correlazioen.
        """
        # normalizza i parametri temporali come BaseProvider.from_to()
        start, end = self.from_to(**kwargs)
        idx = self._date_index(start, end, self.MAX_DEPTH)

        # parametri per ticker
        mu_a = self._to_per_ticker(self.annual_rets, tickers)
        sg_a = self._to_per_ticker(self.annual_vol, tickers)
        s0_map = self._to_per_ticker(self.start_prices, tickers)
        cap0_map = self._to_per_ticker(self.base_mkcaps, tickers)

        mu_d = np.array([mu_a[t] / 252.0 for t in tickers]) # vettore drift giornaliero
        sigma_d = np.array([sg_a[t] / np.sqrt(252.0) for t in tickers]) # vettore vol giornaliera

        # matrice di covarianza giornaliera
        if self.correlation_matrix is not None:
            diag = np.diag(sigma_d)
            Sigma_d = diag @ self.correlation_matrix @ diag
        else:
            Sigma_d = np.diag(sigma_d ** 2)

        # campionamento rendimenti multivariati (n_days × n_tickers)
        R = self.rng.multivariate_normal(mean=mu_d, cov=Sigma_d, size=len(idx))

        # costruzione prezzi
        S0_vec = np.array([s0_map[t] for t in tickers])
        close_mat = np.cumprod(1.0 + R, axis=0) * S0_vec[None, :]
        open_mat = np.vstack([S0_vec, close_mat[:-1, :]])
        # rumore intraday condiviso (puoi differenziarlo per ticker se vuoi)
        eps = np.abs(self.rng.normal(loc=0.0, scale=float(np.mean(sigma_d)), size=len(idx)))
        high_mat = np.maximum(open_mat, close_mat) * (1.0 + 0.25 * eps[:, None])
        low_mat = np.minimum(open_mat, close_mat) * (1.0 - 0.25 * eps[:, None])
        low_mat = np.clip(low_mat, 0.0, None)

        cap0_vec = np.array([cap0_map[t] for t in tickers])
        mkcap_mat = cap0_vec[None, :] * (close_mat / S0_vec[None, :])

        # multi index sulle colonne del dataframe
        cols_ohlc = pd.MultiIndex.from_product([tickers, ["open", "high", "low", "close"]])
        mats = np.stack([open_mat, high_mat, low_mat, close_mat], axis=2)
        data = mats.reshape(mats.shape[0], -1)
        ohlc = pd.DataFrame(data, index=idx, columns=cols_ohlc)
        
        cols_mcap = pd.MultiIndex.from_product([tickers, ["marketCap"]])
        mcap = pd.DataFrame(mkcap_mat, index=idx, columns=cols_mcap)

        df = pd.concat([ohlc, mcap], axis=1).sort_index(axis=1)
        return df

