import copy
import yfinance
import numpy as np
import pandas as pd
from scipy import linalg as la
from typing import Union, Self
from matplotlib import pyplot as plt, axes; plt.style.use('ggplot')
from marketdata.yfinance import YFinanceProvider

# TODO 1: implementare possibilità di chiedere sia un periodo (con data fine = oggi) che data inizio e fine
class Tickers:
    """
    Classe Tickers: è un catalogo di dati storici riferiti a una serie di asset
    che viene letto da diversi possibili provider. Vengono implementati attributi classici
    dell'analisi finanziaria per determinare i rendimenti e le statistiche di tali asset
    nel periodo di riferimento.

    Importante: è un oggetto data-centric. Nessuna logica finanziaria.
    """
    def __init__(self, tickers: Union[str, list[str]], period: str='1mo', provider=None):
        """
        Costruttore della classe Tickers.
        
        :param tickers: Lista di ticker o singolo ticker
        :type tickers: Union[str, list[str]]
        :param period: Periodo storico (es: '1mo', '1y', '5y')
        :type period: str
        :param provider: Provider di dati (default: Yahoo Finance)
        :type provider: MarketDataProvider | None
        """
        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self._period = period
        self._provider = provider or YFinanceProvider()

        # Scarico i dati attraverso il provider
        self.df = self._provider.download(self.tickers, period).dropna()
    
    def copy(self):
        return copy.copy(self)
    
    # TODO: da re-implementare dopo che non eredito più da yfinance.Tickers
    # @property
    # def _all_prices_df(self) -> pd.DataFrame:
    #     """
    #     Attributo privato: dataframe con tutti i prezzi (non solo chiusura, anche apertura, max e min)
    #     utilizzato per invocare il metodo del diagramma a candela.
    #     """
    #     return self.history(period=self._period).dropna()
    
    @property
    def n_assets(self) -> int:
        """
        Numero di ticker (o assets) scaricati e presenti nel dataframe.
        """
        return len(self.df.columns)
    
    @property
    def daily_returns(self) -> pd.DataFrame:
        """
        Rendimento giornaliero per ogni data della serie storica.
        """
        return self.df.pct_change()
    
    @property
    def returns(self) -> pd.DataFrame:
        """
        Rendimento composto calcolato per ogni data della serie storica partendo dall'inizio
        del periodo di osservazione.
        """
        return np.expm1(np.log1p(self.daily_returns).cumsum()) # TODO: check this
    
    @property
    def drawdown(self) -> pd.DataFrame:
        """
        Drawdown giornaliero calcolato per ogni data della serie storica.
        """
        return self.returns.cummax()-self.returns
    
    @property
    def comp_returns(self) -> pd.Series:
        """
        Rendimento composto dall'inizio del periodo di osservazione calcolato su
        tutto il periodo.
        """
        return np.expm1(np.log1p(self.daily_returns).sum()) # TODO: check this
    
    @property
    def daily_volatility(self) -> pd.Series:
        return self.daily_returns.std(ddof=0)
    
    @property
    def annual_returns(self) -> pd.Series:
        days_passed = (self.df.index[-1] - self.df.index[0]).days
        return (self.comp_returns+1)**(365.24/days_passed)-1
    
    @property
    def annual_volatility(self) -> pd.Series:
        days_passed = (self.df.index[-1] - self.df.index[0]).days
        return np.expm1(np.log1p(self.comp_returns)*365.24/days_passed) # TODO: check this
    
    @property
    def skewness(self) -> pd.Series:
        return self.df.skew()
    
    @property
    def kurtosis(self) -> pd.Series:
        return self.df.kurtosis()
    
    @property
    def return_on_risk(self) -> pd.Series:
        return self.annual_returns / self.annual_volatility
    
    @property
    def max_drawdown(self) -> pd.Series:
        return self.drawdown.max()
    
    def sharpe_ratio(self, rf: float) -> pd.Series:
        return (self.annual_returns - rf) / self.annual_volatility
    
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

    def plot(self, rescale: bool=True, *args, **kwargs) -> axes.Axes:
        """
        Metodo di visualizzazione del valore dei titoli presenti nell'istanza invocata.
        Il plot avviene in un solo grafico dotato di legenda.
        
        :param rescale: Se vero, riscala i titoli mostrando così il rapporto tra il valore puntuale e quello
            all'inizio del periodo e permette dunque di comparare a occhio i vari rendimenti
        :type rescale: bool
        :return: Restituisce oggetto Axes
        :rtype: Axes
        """
        if rescale:
            return (self.df/self.df.iloc[0]).plot(*args, **kwargs)
        else:
            return self.df.plot(*args, **kwargs)

class Portfolio:
    """
    Classe che simula un portafoglio di asset costruito tramite dei pesi.
    Utilizza la classe Tickers, che contiene i dati storici, ma non è un tipo di Tickers.
    In termini OOP: non implementa inheritance bensì composition (HAS-A).
    Un portafoglio è dunque un insieme di asset (tickers) con dei pesi associati.
    
    Importante: è un oggetto finance-centric. Implementa logica finanziaria.
    """
    _WEIGHT_BOUNDS = ((0., 1.),)

    def __init__(self, tickers: Tickers, weights: Union[list, np.ndarray, pd.Series]=None):
        """
        Costruttore di Portfoio
        
        :param tickers: Oggeto tickers contenente gli asset del portafoglio.
        :type tickers: Tickers
        :param weights: Pesi associati agli asset del portafoglio.
        :type weights: Union[list, np.ndarray, pd.Series]
        """
        self._t = tickers
        self._w = weights

    @property
    def weights(self) -> pd.Series:
        if self._w is None:
            self._w = np.repeat(1/self._t.n_assets, self._t.n_assets)
            return self.weights
        elif isinstance(self._w, list):
            return pd.Series(self._w, index=self._t.tickers)
        elif isinstance(self._w, np.ndarray):
            return pd.Series(self._w, index=self._t.tickers)
        else:
            return self._w

    @weights.setter
    def weights(self, new_value: Union[list, np.ndarray, pd.Series]) -> None:
        if not all(lower <= val <= upper for (lower, upper), val in zip(self._WEIGHT_BOUNDS*len(new_value), new_value)):
            raise ValueError("Each weight must be a number between 0 and 1.")
        if len(new_value) != self._t.n_assets:
            raise ValueError("New weights must be equal to the number of assets in the portfolio.")
        if not np.isclose(sum(new_value), 1.0):
            new_value = [nv / sum(new_value) for nv in new_value]
        self._w = new_value

    @property
    def ptf_return(self) -> float:
        return self.weights.T @ self._t.annual_returns
    
    @property
    def ptf_comp_returns(self) -> pd.Series:
        return self._t.returns @ self.weights
    
    @property
    def ptf_daily_returns(self) -> pd.Series:
        return self._t.daily_returns @ self.weights
    
    @property
    def covmat(self) -> pd.DataFrame:
        return self._t.daily_returns.cov()

    @property
    def ptf_volatility(self) -> float:
        return (self.weights.T @ self.covmat @ self.weights)**0.5

    # def minimize_vol(self, target_return: float = None) -> None:
    #     """
    #     Metodo che resetta i pesi del portafoglio minimizzandone la volatilità per un dato
    #     rendimento. Viene minimizzata la forma quadratica sqrt(w^T C w) mediante l'utilizzo della
    #     libreria minimize di scipy.optimize.
        
    #     Parametri
    #     ---------
    #     target_return: float
    #         Rendimento target per il quale si desidera trovare i pesi che minimizzano la volatilità.
    #     """
    #     def to_min(w):
    #         chol = la.cholesky(self.covmat)
    #         y = chol.T @ w
    #         # FIXME: indagare qui sotto perchè preferisce la forma quadratica e non la sqrt
    #         return y.T @ y
        
    #     def match_ret(w):
    #         self.weights = w # qui avviene il reset dei pesi con il Setter della classe
    #         return target_return - self.ptf_return if target_return is not None else 0.
        
    #     def sum_weights(w):
    #         return np.sum(w) - 1

    #     w0 = self.weights.values # proposta iniziale
    #     bounds = self._WEIGHT_BOUNDS*self.n_assets # vincoli
    #     constr = (
    #         {'type': 'eq', 'fun': sum_weights}, # constraint somma pesi
    #         {'type': 'eq', 'fun': match_ret} # constraint rendimento
    #     )

    #     # minimizzazione
    #     from scipy.optimize import minimize
    #     _ = minimize(
    #         to_min,
    #         w0,
    #         method='SLSQP',
    #         options={'disp': False},
    #         constraints=constr,
    #         bounds=bounds
    #     )

    def msr(self, rf: float = 0.03) -> Self:
        """
        Maximum Sharpe Ratio.
        Metodo che modifica i pesi della classe al fine di ottenere il MSR Portfolio.
        
        :param rf: Tasso risk-free di riferimento.
        :type rf: float
        :return: Restituisce la classe stessa, con i pesi modificati per MSR.
        :rtype: Self
        """
        # funzione da minimizzare (sharpe ratio negativo)
        def to_min(w):
            # prova decomposizione di Cholesky per aumentare performance
            try:
                chol = la.cholesky(self.covmat)
                y = chol.T @ w
                return (rf - w.T @ self._t.annual_returns) / (y.T @ y)**0.5
            # a volte non riesce (qualche minore non definito positivo per errori numerici)
            except la.LinAlgError:
                # in questo caso fa il calcolo più pesante senza la Cholesky
                return (rf - w.T @ self._t.annual_returns) / (self.weights.T @ self.covmat @ self.weights)**0.5
        
        # funzione di constraint sui pesi e di reset dell'attributo della classe
        def update_w_and_sum_to_1(w):
            self.weights = w # reset attributo col setter
            return np.sum(w) - 1 # constraint
        
        # massimizzazione sharpe ratio
        from scipy.optimize import minimize
        _ = minimize(
            to_min,
            self.weights.values,
            method='SLSQP',
            options={'disp': False},
            constraints=({'type': 'eq', 'fun': update_w_and_sum_to_1}),
            bounds=self._WEIGHT_BOUNDS*self._t.n_assets 
        )
        return self

    def gmv(self) -> Self:
        """
        Global Minimum Variance.
        Metodo che modifica i pesi del portafoglio per minimizzare la varianza.
        Si mostra che tale portafoglio è il MSR con tasso risk-free posto a 0.
        
        :return: Restituisce la classe stessa, con i pesi modificati per GMV.
        :rtype: Self
        """
        return self.msr(rf=0.)

    def ew(self) -> Self:
        """
        Equally Weighted portfolio.
        
        :return: Restituisce la classe stessa, con i pesi modificati per EW.
        :rtype: Self
        """
        self._w = None
        return self

# TODO: eventualmente creare una classe per la frontiera efficiente anche vedendo cosa segue nel corso di EDHEC
def efficient_frontier(
    tickers: list[str],
    n_samples: int = 20,
    period: str = None,
    return_range: list[float] = None,
) -> list[Portfolio]:
    """
    Calcola la frontiera efficiente per un insieme di titoli.

    :param tickers: Lista di ticker da includere nel portafoglio.
    :type tickers: list[str]
    :param n_samples: Numero di portafogli campionati sulla frontiera (default: 20).
    :type n_samples: int
    :param period: Periodo storico per il download dei dati (es. '1mo', '1y'). Se None viene usato il default del costruttore.
    :type period: str | None
    :param return_range: Intervallo [min, max] dei rendimenti target da esplorare. Se None si usa l'intervallo derivato dai singoli asset.
    :type return_range: list[float] | None
    :return: Lista di oggetti Portfolio corrispondente ai punti della frontiera efficiente.
    :rtype: list[Portfolio]
    """
    start_ptf = Portfolio(tickers, period=period)

    if return_range is None:
        return_range = (start_ptf.annual_returns.min(), start_ptf.annual_returns.max())

    # definizione della griglia dei rendimenti
    ptfs = [start_ptf.copy() for _ in range(n_samples)]
    target_returns = np.linspace(*return_range, n_samples)

    # definizione dei portafoglio sulla frontiera efficiente
    for tr, ptf in zip(target_returns, ptfs):
        ptf.minimize_vol(tr)
    
    return ptfs

def plot_efficient_frontier(ptfs: list[Portfolio]):
    """
    Funzione che plotta la frontiera efficiente ed i singoli asset in portafoglio
    nel piano rischio-rendimento.
    """
    returns = [ptf.ptf_return for ptf in ptfs]
    volatilities = [ptf.ptf_volatility for ptf in ptfs]
    sharpe_ratios = [ptf.ptf_return / ptf.ptf_volatility for ptf in ptfs]

    fig, ax = plt.subplots()
    ax.scatter(volatilities, returns, c=sharpe_ratios, marker='*', label='Efficient frontier')
    ax.scatter(ptfs[0].annual_volatility, ptfs[0].annual_returns, c='gray', label='Single assets')

    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    fig.suptitle('Risk-return space')

    plt.legend()

def plot_ef_cml(ptfs: list[Portfolio], rf_ratio: float):
    """
    Funzione che plotta la frontiera efficiente ed i singoli asset in portafoglio
    nel piano rischio-rendimento.
    """
    returns = [ptf.ptf_return for ptf in ptfs]
    volatilities = [ptf.ptf_volatility for ptf in ptfs]
    max_sharpe = ptfs[0].copy()
    max_sharpe.maximize_sharpe(rf_ratio)
    cml_x = [0, max_sharpe.annual_volatility.max()]
    m = (max_sharpe.ptf_return - rf_ratio) / max_sharpe.ptf_volatility
    cml_y = [rf_ratio, m*max_sharpe.annual_volatility.max()+rf_ratio]
    
    fig, ax = plt.subplots()
    ax.plot(volatilities, returns, label='Efficient frontier')
    ax.plot(cml_x, cml_y, c='b', label='Capital market line')
    ax.scatter(max_sharpe.annual_volatility, max_sharpe.annual_returns, c='gray', label='Single assets')
    
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    fig.suptitle('Risk-return space')

    plt.legend()
