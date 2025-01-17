import yfinance
import numpy as np
import pandas as pd
from typing import Union

class Tickers(yfinance.Tickers):
    """
    Classe Tickers che eredita da yfinance.Tickers
    """
    def __init__(self, *args, period: str, **kwargs):
        """
        Costruttore della classe Tickers.

        Parametri
        ---------
        period: str
            Estensione nel passato dei dati richiesti a yfinance.
        *args, **kwargs: Any
            Eventuali argomenti aggiuntivi da passare al costruttore della classe padre (ad esempio
            la stringa dei tickers)
        """
        # costruttore della classe padre
        super().__init__(*args, **kwargs)
        # unico attributo inizializzato qui è il dataframe, per gli altri si fa lazy sotto
        self.df = self.history(period=period).Close.dropna()
        self.n_assets = len(self.df.columns)

    @property
    def daily_returns(self) -> pd.DataFrame:
        """
        Daily returns computed as R_j = P_{j+1} / P_j - 1.
        """
        return self.df.pct_change()
    
    @property
    def comp_returns(self) -> pd.DataFrame:
        return (1+self.daily_returns).prod()
    
    @property
    def daily_volatility(self) -> pd.Series:
        return self.daily_returns.std(ddof=0)
    
    @property
    def annual_returns(self) -> pd.Series:
        days_passed = (self.df.index[-1] - self.df.index[0]).days
        return self.comp_returns**(365.24/days_passed)-1
    
    @property
    def annual_volatility(self) -> pd.Series:
        days_passed = (self.df.index[-1] - self.df.index[0]).days
        return self.daily_returns.std()*((365.24/days_passed)**0.5)
    
    @property
    def skewness(self) -> pd.Series:
        return self.df.skew()
    
    @property
    def kurtosis(self) -> pd.Series:
        return self.df.kurtosis()
    
    def measured_var(self, level: float) -> pd.Series:
        """
        Calcolo il Value at Risk misurato nel periodo di campionamento della serie storica
        mediante la sua definizione.
        """
        return -self.df.quantile(level)
    
    def cornish_fischer_var(self, level: float) -> pd.Series:
        """
        Approccio semi-parametrico al calcolo del Value at Risk mediante l'espansione di
        Cornish-Fischer per le distribuzioni non-gaussiane.
        """
        # calcolo F^-1(level) per una distribuzione gaussiana normalizzata
        from scipy.stats import norm
        z = norm.ppf(level/100)

        # performo su z l'espansione di Cornish-Fischer
        esp_1 = (z**2 - 1) * self.skewness/6
        esp_2 = (z**3 - 3*z) * (self.kurtosis - 3)/24
        esp_3 = (2*z**3 - 5*z) * (self.skewness**2)/36
        z = z + esp_1 + esp_2 - esp_3
    
        return -(self.daily_returns.mean() + z*self.daily_returns.std(ddof=0))

class Portfolio(Tickers):
    """
    Classe portafoglio, che specifica anche dei
    pesi mediante i quali si può costruire la frontiere efficiente del portafoglio dato da tali tickers.
    """
    def __init__(self, *args, weights: Union[list, np.ndarray, pd.Series] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._w = weights

    @property
    def weights(self):
        if self._w is None:
            self._w = np.repeat(1/self.n_assets, self.n_assets)
            return self.weights
        elif isinstance(self._w, list):
            return pd.Series(self._w, index=list(self.tickers.keys()))
        elif isinstance(self._w, np.ndarray):
            return pd.Series(self._w, index=list(self.tickers.keys()))
        else:
            return self._w

    @property
    def ptf_return(self):
        return self.weights.T @ self.annual_returns
    
    @property
    def ptf_volatility(self):
        return (self.weights.T @ self.covmat @ self.weights)**0.5
    
    @property
    def covmat(self):
        return self.daily_returns.cov()
