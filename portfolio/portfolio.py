"""
Modulo portfolio: gestione e analisi di portafogli di asset finanziari.
Contiene le classi:
- Tickers: data-centric container.
- Portfolio: finance-centric implementazione di logiche finanziarie.

Sviluppato da Samuele Voltan durante e dopo il corso
"Introduction to Portfolio Construction and Analysis with Python" della EDHEC Business School.

Riferimenti:
- https://www.edhec.edu/en
- https://www.coursera.org/learn/introduction-portfolio-construction-python
- https://www.paolocoletti.com/financialtrading
"""

import copy
import numpy as np
import pandas as pd
from scipy import linalg as la
from .plotter import GLOBAL_PLOTTER, PortfolioPlotter
from typing import Union, Any, Optional
from marketdata.yfinance import YFinanceProvider
from matplotlib import pyplot as plt, axes; plt.style.use('ggplot')

WEIGHT_BOUNDS = ((0., 1.),)

class Tickers:
    """
    Classe Tickers: è un catalogo di dati storici riferiti a una serie di asset
    che viene letto da diversi possibili provider. Vengono implementati attributi classici
    dell'analisi finanziaria per determinare i rendimenti e le statistiche di tali asset
    nel periodo di riferimento.

    Viene invocata in due modi possibili:
        1. Specificando un periodo storico (es: '1mo', '1y', '5y', 'max')
        2. Specificando una data di inizio e una di fine in formato YYYY-MM-DD

    Importante: è un oggetto data-centric. Nessuna logica finanziaria.
    """
    def __init__(
        self,
        tickers: Union[str, list[str]],
        period: Optional[str]=None,
        start: Optional[str]=None,
        end: Optional[str]=None,
        provider=None
    ):
        """
        Costruttore della classe Tickers.
        
        :param tickers: Lista di ticker o singolo ticker
        :type tickers: Union[str, list[str]]
        :param period: Periodo storico (es: '1mo', '1y', '5y')
        :type period: Optional[str]
        :param start: Data di inizio periodo storico in formato YYYY-MM-DD
        :type start: Optional[str]
        :param end: Data di fine periodo storico in formato YYYY-MM-DD
        :type end: Optional[str]
        :param provider: Provider di dati (default: Yahoo Finance)
        :type provider: BaseProvider | None
        """
        # controllo dell'input
        if period and (start or end):
            raise ValueError("Specify either 'period' or 'start'/'end', not both.")
        if not period and not start:
            raise ValueError("Must specify either 'period' or at least 'start'.")

        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self.provider = provider or YFinanceProvider()

        # scarico i dati attraverso il provider
        if period:
            self.df = self.provider.download(self.tickers, period=period)
        else:
            self.df = self.provider.download(self.tickers, start_date=start, end_date=end)
    
    def _repr_html_(self):
        summary = pd.DataFrame({
            "Annual Returns": self.annual_returns,
            "Annual Volatility": self.annual_volatility,
            "Max Drawdown": self.max_drawdown,
            "Sharpe Ratio": self.sharpe_ratio(),
            "VaR (95%)": self.measured_var(),
            "VaR Cornish-Fischer (95%)": self.cornish_fischer_var(),
            "CVaR (95%)": self.expected_shortfall()
        })
        return f"<h3>Summary for {', '.join(self.tickers)}</h3>" + summary.sort_index().to_html()
    
    def copy(self):
        return copy.copy(self)
    
    @property
    def close(self) -> pd.DataFrame:
        """
        Prezzi di chiusura di tutti i ticker: è una tabella n_assets x n_date.
        Rispetto a self.df (dati scaricati dal provider) rimuove la dimensione MultiIndex delle
        colonne selezionando solamente i prezzi di chiusura.

        :return: DataFrame con i prezzi di chiusura.
        :rtype: pd.DataFrame
        """
        # NOTE: dropna si rende utile siccome df ha sempre dati aggiornati a oggi (colpa di marketCap)
        return self.df.xs('close', axis=1, level=-1).dropna()
    
    @property
    def last_mkcap(self) -> pd.Series:
        """
        Restituisce l'ultimo dato disponibile sulla capitalizzazione dei titoli.
        """
        # NOTE: funzione utile soltanto poichè il provider FMP non fornisce tutto lo storico nella versione free
        mkcap = self.df.xs('marketCap', axis=1, level=-1).iloc[-1]
        mkcap.name = 'Market CAP'
        return mkcap
    
    @property
    def n_assets(self) -> int:
        """
        Numero di ticker (o assets) scaricati e presenti nel dataframe.
        """
        return len(self.close.columns)
    
    @property
    def daily_returns(self) -> pd.DataFrame:
        """
        Rendimento giornaliero per ogni data della serie storica.

        :return: DataFrame della stessa dimensione e caratteristiche di self.close
        :rtype: pd.DataFrame
        """
        return self.close.pct_change()
    
    @property
    def daily_volatility(self) -> pd.Series:
        """
        Restituisce la volatilità calcolata sui rendimenti giornalieri (da qui il "daily").
        
        :return: Serie con la volatilità di ogni ticker.
        :rtype: pd.Series
        """
        return self.daily_returns.std(ddof=0)
    
    @property
    def comp_returns(self) -> pd.DataFrame:
        """
        Rendimento composto calcolato per ogni data della serie storica partendo dall'inizio
        del periodo di osservazione.

        :return: DataFrame della stessa dimensione e caratteristiche di self.close
        :rtype: pd.DataFrame
        """
        return np.expm1(np.log1p(self.daily_returns).cumsum())
    
    @property
    def annual_returns(self) -> pd.Series:
        """
        Rendimenti annui calcolati proiettando i rendimenti composti ottenuti durante il periodo di osservazione.

        :return: Serie con i rendimenti annui di ogni ticker.
        :rtype: pd.Series
        """
        days_passed = (self.close.index[-1] - self.close.index[0]).days
        last_row = (self.comp_returns[-1:]+1)**(365.24/days_passed)-1
        rets = last_row.iloc[0]
        rets.name = "Annualized returns"
        return rets
    
    @property
    def annual_volatility(self) -> pd.Series:
        """
        Volatilità annua calcolata proiettando la volatilità calcolata sui rendimenti giornalieri.

        :return: Serie con la volatilità annua di ogni ticker.
        :rtype: pd.Series
        """
        vol = self.daily_returns.std(ddof=0) * np.sqrt(252)
        vol.name = "Annualized volatility"
        return vol
    
    @property
    def return_on_risk(self) -> pd.Series:
        """
        Rapporto tra rendimento annuo e volatilità annua.
        
        :return: Serie con il return on risk di ogni ticker.
        :rtype: pd.Series
        """
        return self.annual_returns / self.annual_volatility
    
    @property
    def skewness(self) -> pd.Series:
        return self.daily_returns.skew()
    
    @property
    def kurtosis(self) -> pd.Series:
        return self.daily_returns.kurtosis()
    
    @property
    def drawdown(self) -> pd.DataFrame:
        """
        Drawdown giornaliero calcolato per ogni data della serie storica.

        :return: DataFrame della stessa dimensione e caratteristiche di self.close
        :rtype: pd.DataFrame
        """
        return self.comp_returns.cummax()-self.comp_returns
    
    @property
    def max_drawdown(self) -> pd.Series:
        """
        Massimo drawdown giornaliero calcolato sul periodo di osservazione.
        
        :return: Serie con il massimo drawdown giornaliero di ogni ticker.
        :rtype: pd.Series
        """
        return self.drawdown.max()
    
    def sharpe_ratio(self, rf: float=0.03) -> pd.Series:
        """
        Sharpe Ratio (return on risk modificato per il tasso di redimento risk-free).
        
        :param rf: Tasso di rendimento risk-free
        :type rf: float
        :return: Serie con lo Sharpe Ratio di ogni ticker.
        :rtype: pd.Series
        """
        return (self.annual_returns - rf) / self.annual_volatility
    
    def measured_var(self, level: float=0.95) -> pd.Series:
        """
        Calcolo il Value at Risk misurato nel periodo di campionamento della serie storica
        mediante la sua definizione.
        """
        # controllo level
        if not 0 < level < 1:
            raise ValueError("The confidence level must be between 0 and 1.")
        
        return -self.daily_returns.quantile(level)

    def cornish_fischer_var(self, level: float=0.95) -> pd.Series:
        """
        Approccio semi-parametrico al calcolo del Value at Risk mediante l'espansione di
        Cornish-Fischer per le distribuzioni non-gaussiane.
        """
        # controllo level
        if not 0 < level < 1:
            raise ValueError("The confidence level must be between 0 and 1.")

        # calcolo F^-1(level) per una distribuzione gaussiana normalizzata
        from scipy.stats import norm
        z = norm.ppf(level)

        # performo su z l'espansione di Cornish-Fischer
        esp_1 = (z**2 - 1) * self.skewness/6
        esp_2 = (z**3 - 3*z) * (self.kurtosis - 3)/24
        esp_3 = (2*z**3 - 5*z) * (self.skewness**2)/36
        z = z + esp_1 + esp_2 - esp_3
    
        return -(self.daily_returns.mean() + z*self.daily_returns.std(ddof=0))

    def expected_shortfall(self, level: float=0.95) -> pd.Series:
        var_threshold = self.daily_returns.quantile(level)
        return -self.daily_returns[self.daily_returns < var_threshold].mean()
    
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
        
    def plot_dailyret_dist(self, level: float=0.95):
        """
        Grafico distribuzione rendimenti con VaR e CVaR evidenziati

        :param level: Livello di confidenza per il calcolo del VaR e CVaR
        :type level: float
        """
        _, axes = plt.subplots(len(self.tickers), 1, figsize=(8, 4*len(self.tickers)))
        if len(self.tickers) == 1:
            axes = [axes]
        
        for i, ticker in enumerate(self.tickers):
            data = self.daily_returns[ticker]
            var = self.measured_var(level)[ticker]
            cvar = self.expected_shortfall(level)[ticker]
            
            axes[i].hist(data, bins=50, color='skyblue', edgecolor='black')
            axes[i].axvline(-var, color='red', linestyle='--', label=f'VaR {level*100:.0f}%')
            axes[i].axvline(-cvar, color='blue', linestyle='--', label=f'CVaR {level*100:.0f}%')
            axes[i].set_title(f"Distribuzione rendimenti giornalieri {ticker}")
            axes[i].legend()
        
        plt.tight_layout()
        plt.show()

class Portfolio:
    """
    Classe che simula un portafoglio di asset costruito tramite dei pesi.
    Utilizza la classe Tickers, che contiene i dati storici, ma non è un tipo di Tickers.
    In termini OOP: non implementa inheritance bensì composition (HAS-A).
    Un portafoglio è dunque un insieme di asset (tickers) con dei pesi associati.
    
    Importante: è un oggetto finance-centric. Implementa logica finanziaria.
    """
    plotter: PortfolioPlotter = GLOBAL_PLOTTER

    def __init__(self, tickers: Tickers, weights: Any=None, name: str=None):
        """
        Costruttore di Portfolio.
        
        :param tickers: Oggeto tickers contenente gli asset del portafoglio.
        :type tickers: Tickers
        :param weights: Pesi associati agli asset del portafoglio.
        :type weights: Union[list, np.ndarray, pd.Series]
        """
        self.tickers = tickers
        self.weights = self.validate_weights(weights)
        self.name = name
    
    # BUG: must pass an index (rivedere tutta la funzione)
    def _repr_html_(self):
        summary = pd.DataFrame({
            "Annualized returns": self.annual_return,
            "Annualized volatility": self.annual_volatility,
            "Max drawdown": self.max_drawdown,
            "Sharpe Ratio (3%)": self.sharpe_ratio()
        })
        return f"<h3>Summary for {self.name} portfolio</h3>" + summary.sort_index().to_html()
    
    def validate_weights(self, weights: Union[list, pd.Series, np.ndarray, None]) -> pd.Series:
        """
        Validazione dei pesi passati al costruttore.
        """
        # se weights è None, equally weighted portfolio
        weights = np.repeat(1/self.tickers.n_assets, self.tickers.n_assets) if weights is None else weights
        
        # formattazione pesi a pd.Series
        weights_s = pd.Series(weights, index=self.tickers.tickers) if not isinstance(weights, pd.Series) else weights

        # controllo su coerenza num. titoli e pesi forniti
        if self.tickers.n_assets != len(weights_s):
            raise ValueError("Tickers and weights must have the same length.")

        # controllo su normalizzazione pesi
        if not np.isclose(weights_s.sum(), 1.):
            # non sollevo errori ma semplicemente normalizzo
            weights_s /= weights_s.sum()

        # controllo sui limiti dei pesi tra 0 e 1
        # NOTE: logicamente questo controllo deve esser fatto dopo il precedente
        lowbound, upbound = WEIGHT_BOUNDS[0]
        if (weights_s < lowbound).any() or (weights_s > upbound).any():
            raise ValueError("Each weight must be a number between 0 and 1.")
        
        return weights_s

    @property
    def enc(self) -> float:
        """
        ENC (Effective Number of Constituents).
        """
        wsquared = self.weights**2
        return 1/wsquared.sum()

    @property
    def daily_returns(self) -> pd.Series:
        """
        Rendimenti giornalieri del portafoglio per ogni data della serie storica.
        """
        return self.tickers.daily_returns @ self.weights

    @property
    def comp_returns(self) -> pd.Series:
        """
        Rendimenti composti del portafoglio sul periodo di osservazione (data per data).
        """
        return self.tickers.comp_returns @ self.weights

    @property
    def annual_return(self) -> float:
        """
        Rendimento annualizzato del portafoglio.
        """
        return self.weights.T @ self.tickers.annual_returns
    
    @property
    def return_on_risk(self) -> float:
        """
        Rapporto tra rendimento annuo e volatilità annua.
        """
        return self.annual_return / self.annual_volatility
    
    @property
    def drawdown(self) -> pd.Series:
        """
        Drawdown giornaliero calcolato per ogni data della serie storica.
        """
        return self.comp_returns.cummax()-self.comp_returns
    
    @property
    def max_drawdown(self) -> float:
        """
        Massimo drawdown giornaliero calcolato sul periodo di osservazione.
        """
        return self.drawdown.max()
    
    @property
    def covmat(self) -> pd.DataFrame:
        """
        Matrice di covarianza calcolata con correlazioni e varianze del campione di titoli in portafoglio.
        """
        return self.tickers.daily_returns.cov()

    @property
    def daily_volatility(self) -> float:
        try:
            # prova decomposizione di Cholesky per aumentare performance
            chol = la.cholesky(self.covmat)
            y = chol.T @ self.weights
            return (y.T @ y)**0.5
        except la.LinAlgError:
            # in questo caso fa il calcolo più pesante senza la Cholesky
            return (self.weights.T @ self.covmat @ self.weights)**0.5
    
    @property
    def annual_volatility(self) -> float:
        return self.daily_volatility * np.sqrt(252)
    
    def sharpe_ratio(self, rf: float=0.03) -> float:
        """
        Sharpe ratio del portafoglio.
        """
        return (self.annual_return - rf) / self.annual_volatility
    
    @classmethod
    def set_plotter(cls, plotter: PortfolioPlotter) -> None:
        """Sostituisce il plotter usato da tutti gli oggetti Portfolio."""
        cls.plotter = plotter

    @classmethod
    def get_plotter(cls) -> PortfolioPlotter:
        """Restituisce il plotter attuale."""
        return cls.plotter
    
    def plot_returns(self, ax: axes.Axes | None = None, *args, **kwargs) -> axes.Axes:
        """
        Plot dei rendimenti del portafoglio.
        - Se 'ax' è fornito: grafica sull'axes specificato (nessuna condivisione automatica).
        - Se 'ax' è None: delega al Plotter per condivisione basata su DatetimeIndex.
        Parametri supportati: 'rescale' (bool), 'legend_loc', più kwargs di pandas.plot().
        """
        plotter = self.__class__.get_plotter()
        return plotter.plot_returns(self, ax=ax, *args, **kwargs)

    def plot_drawdown(self, ax: axes.Axes | None = None, *args, **kwargs) -> axes.Axes:
        """Plot del drawdown; stessa delega e regole di condivisione dell'axes."""
        plotter = self.__class__.get_plotter()
        return plotter.plot_drawdown(self, ax=ax, *args, **kwargs)

    @staticmethod
    def get_shared_figure():
        """Figura usata più di recente dal Plotter (utile per fig.show())."""
        return Portfolio.get_plotter().get_last_figure()

    # def plot_returns(self, ax: axes.Axes, *args, **kwargs) -> axes.Axes:
    #     """
    #     Plot rendimenti.
    #     """
    #     if not kwargs.get('rescale'):
    #         return self.comp_returns.plot(ax=ax, label=self.name, *args, **kwargs)
    #     return (self.comp_returns/self.comp_returns.iloc[0]).plot(ax=ax, label=self.name, *args, **kwargs)

def get_msr(tickers: Tickers, rf: float=0.03) -> Portfolio:
    """
    Dato un oggetto Tickers (insieme di titoli) restituisce il MSR (Max Sharpe-Ratio) Portfolio.
    
    :param tickers: Insieme di titoli dai quali costruire il MSR Portfolio.
    :type tickers: Tickers
    :param rf: Tasso risk-free di riferimento.
    :type rf: float
    :return: Oggetto Portfolio costruito con i titoli di Tickers ed i pesi che massimizzano lo SR.
    :rtype: Portfolio
    """
    # funzione da minimizzare (sharpe ratio negativo)
    def negative_sharpe_ratio(w):
        try_ptf = Portfolio(tickers, w)
        return -try_ptf.sharpe_ratio(rf=rf)
    
    # funzione di constraint (normalizzazione pesi)
    def normalization(w):
        return np.sum(w) - 1
    
    # configurazione iniziale dei pesi
    w0 = np.repeat(1/tickers.n_assets, tickers.n_assets)

    # massimizzazione sharpe ratio
    from scipy.optimize import minimize
    new_weights = minimize(
        negative_sharpe_ratio,
        w0,
        method='SLSQP',
        options={'disp': False},
        constraints=({'type': 'eq', 'fun': normalization}),
        bounds=WEIGHT_BOUNDS*tickers.n_assets 
    )
    return Portfolio(tickers, new_weights.x, "MSR")

def get_gmv(tickers: Tickers) -> Portfolio:
    """
    Dato un oggetto Tickers (insieme di titoli) restituisce il GMV (Global Minimum Variance) Portfolio.
    """
    gmv_ptf = get_msr(tickers, rf=0.) # si mostra che GMV = MSR(rf: 0)
    gmv_ptf.name = "GMV"
    return gmv_ptf

def get_eqw(tickers: Tickers) -> Portfolio:
    """
    Dato un oggetto Tickers (insieme di titoli) restituisce il EW (Equally Weighted) Portfolio.
    """
    return Portfolio(tickers, None, "EqW") # sfrutta le proprietà del costruttore di Portfolio

def get_capw(tickers: Tickers) -> Portfolio:
    """
    Dato un oggetto Tickers (insieme di titoli) restituisce il CW (Cap Weighted) Portfolio.
    """
    weights = tickers.last_mkcap / tickers.last_mkcap.sum()
    return Portfolio(tickers, weights, "CapW")

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
        return_range = (start_ptf._t.annual_returns.min(), start_ptf._t.annual_returns.max())

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
    ax.scatter(ptfs[0]._t.annual_volatility, ptfs[0]._t.annual_returns, c='gray', label='Single assets')

    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    fig.suptitle('Risk-return space')

    plt.legend()
