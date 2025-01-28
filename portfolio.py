import copy
import yfinance
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy import linalg as la
from typing import Union, Any, Dict
from matplotlib import pyplot as plt, axes; plt.style.use('ggplot')

class Tickers(yfinance.Tickers):
    """
    Classe Tickers che eredita da yfinance.Tickers
    """
    def __init__(self, *args, period: str = "1mo", **kwargs):
        """
        Costruttore della classe Tickers.

        Parametri
        ---------
        period: str
            Estensione nel passato dei dati richiesti a yfinance.
        *args, **kwargs: Any
            Eventuali argomenti aggiuntivi da passare al costruttore della classe padre (ad esempio
            la stringa dei tickers)

        Note
        ----
        Gli unici attributi che vengono inizializzati direttamente in questo metodo sono l'argomento
        posizionale periods in self._periods ed il dataframe self.df contenente i prezzi di chiusura
        di tutti gli asset specificati. Questo viene inizializzato per evitare di dover effettuare
        un download dall'API di yfinance ogni volta che l'attributo viene chiamato, dato che viene
        chiamato spesso nei metodi della classe.
        """
        # costruttore della classe padre
        super().__init__(*args, **kwargs)
        # attributi non lazy
        self._period = period
        self.df = self.history(period=self._period).Close.dropna()
    
    def copy(self):
        return copy.copy(self)
    
    @property
    def _all_prices_df(self) -> pd.DataFrame:
        """
        Attributo privato: dataframe con tutti i prezzi (non solo chiusura, anche apertura, max e min)
        utilizzato per invocare il metodo del diagramma a candela.
        """
        return self.history(period=self._period).dropna()
    
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

    def plot(self, rescale: bool = True, *args, **kwargs) -> axes.Axes:
        """
        Metodo di disegno del valore dei titoli presenti nell'istanza invocata. Il plot avviene
        in un solo grafico dotato di legenda.
        
        Parametri
        ---------
        rescale: bool = True
            Se vero, riscala i titoli mostrando così il rapporto tra il valore puntuale e quello
            all'inizio del periodo e permette dunque di comparare a occhio i vari rendimenti
        *args e **kwargs: Any
            Eventuali ulteriori argomenti passati alla funzione plot() di pd.DataFrame
        """
        if rescale:
            return (self.df/self.df.iloc[0]).plot(*args, **kwargs)
        else:
            return self.df.plot(*args, **kwargs)

class Portfolio(Tickers):
    """
    Classe portafoglio, eredita da Tickers implementando metodi e attributi che dipendono dalla
    distribuzione dei titoli nel portafoglio, ottenuta tramite la specifica dei pesi di ciascuno.
    """
    _WEIGHT_BOUNDS = ((0., 1.),)

    def __init__(self, *args, weights: Union[list, np.ndarray, pd.Series] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._w = weights

    @property
    def weights(self) -> pd.Series:
        if self._w is None:
            self._w = np.repeat(1/self.n_assets, self.n_assets)
            return self.weights
        elif isinstance(self._w, list):
            return pd.Series(self._w, index=list(self.tickers.keys()))
        elif isinstance(self._w, np.ndarray):
            return pd.Series(self._w, index=list(self.tickers.keys()))
        else:
            return self._w

    @weights.setter
    def weights(self, new_value: Union[list, np.ndarray, pd.Series]) -> None:
        if not all(lower <= val <= upper for (lower, upper), val in zip(self._WEIGHT_BOUNDS*len(new_value), new_value)):
            raise ValueError("Each weight must be a number between 0 and 1.")
        if len(new_value) != self.n_assets:
            raise ValueError("New weights must be equal to the number of assets in the portfolio.")
        if not np.isclose(sum(new_value), 1.0):
            new_value = [nv / sum(new_value) for nv in new_value]
        self._w = new_value

    @property
    def ptf_return(self) -> float:
        return self.weights.T @ self.annual_returns
    
    @property
    def ptf_comp_returns(self) -> pd.Series:
        return self.returns @ self.weights
    
    @property
    def ptf_daily_returns(self) -> pd.Series:
        return self.daily_returns @ self.weights
    
    @property
    def covmat(self) -> pd.DataFrame:
        return self.daily_returns.cov()

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

    def msr(self, rf: float = 0.03):
        """
        Maximum Sharpe Ratio.
        Metodo che modifica i pesi del portafoglio in maniera da ottenere la massima
        remunerazione del rischio al netto di un tasso risk-free specificato mediante
        il metodo SLSQP di scipy.optimize.

        Parametri
        ---------
        rf: float
            Tasso di rendimento risk-free (e.g. US Treasury) nel periodo analizzato.
            Di default è settato al 3% ma sarebbe meglio sempre fare delle analisi
            e definirlo in maniera coerente con l'inflazione e i tassi del periodo
            storico analizzato.
        """
        # funzione da minimizzare (sharpe ratio negativo)
        def to_min(w):
            # prova decomposizione di Cholesky per aumentare performance
            try:
                chol = la.cholesky(self.covmat)
                y = chol.T @ w
                return (rf - w.T @ self.annual_returns) / (y.T @ y)**0.5
            # a volte non riesce (qualche minore non definito positivo per errori numerici)
            except la.LinAlgError:
                # in questo caso fa il calcolo più pesante senza la Cholesky
                return (rf - w.T @ self.annual_returns) / (self.weights.T @ self.covmat @ self.weights)**0.5
        
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
            bounds=self._WEIGHT_BOUNDS*self.n_assets 
        )
        return self

    def gmv(self):
        """
        Global Minimum Variance.
        Metodo che modifica i pesi del portafoglio in maniera da minimizzarne la
        varianza. Si può mostrare che tale portafoglio è il msr con tasso risk-free
        posto a 0.
        """
        return self.msr(rf=0.)

    def ew(self):
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
    Funzione di calcolo della frontiera efficiente per un portafoglio di titoli.
    
    Parametri
    ---------
    tickers: list[str]
        Lista contenente i tickers dei titoli da inserire in portafoglio, nello
        stesso formato richiesto dai costruttori di yfinance.Tickers e delle classi
        qui derivate da essa.
    n_samples: int
        Numero di punti da campionare sulla frontiera efficiente, corrisponderà
        alla lunghezza della lista di portafogli che viene restituita dalla funzione.
        Di default è fissato a 20.
    period: str
        Periodo di osservazione richiesto dei dati storici per i tickers inseriti
        in portafoglio, nello stesso formato richiesto dai costruttori di Tickers
        e Portfolio. Default è None, dunque tali costruttori vengono chiamati con
        periodo di un mese.
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
    
from functools import wraps
# def cache_plot(label):
#     def decorator(func):
#         plotted_labels = set()  # Set to keep track of plotted labels
        
#         @wraps(func)
#         def wrapper(self, *args, **kwargs):
#             if label not in plotted_labels:
#                 plotted_labels.add(label)  # Mark this label as plotted
#                 return func(self, *args, **kwargs)
#             else:
#                 print(f"Plot with label '{label}' already plotted. Skipping plot.")
        
#         return wrapper
#     return decorator

def cache_plot(func):

    @wraps(func)
    def wrapper(self: StrategyPlotter, *args, **kwargs):
        # check if the method has already been called for this class
        if not self.__class__.has_been_called:
            self.__class__.has_been_called = True  # mark as called
            return func(self, *args, **kwargs)
        else:
            print(f"Plotting method '{func.__name__}' has already been called for this class. Skipping plot.")
    
    return wrapper

class StrategyPlotter(ABC):
    has_been_called = False

    def __init__(self, ptf: Portfolio):
        self.ptf = ptf

    @property
    def bh(self) -> pd.Series:
        """
        Andamento del valore del titolo per una strategia buy and hold sul periodo
        di riferimento.
        """
        return self.ptf.ptf_comp_returns.dropna()+1

    @property
    def rets(self) -> pd.Series:
        """
        Valore giornaliero dei rendimenti del portafoglio.
        """
        return self.ptf.ptf_daily_returns.dropna()
    
    @property
    def _n_days(self) -> int:
        return len(self.bh.index)
    
    @abstractmethod
    def _name(self) -> str:
        pass

    @abstractmethod
    def strategy(self, *args, **kwargs):
        pass
    
    @classmethod
    def reset(cls):
        """
        Reset the class-level tracking variable.
        """
        cls.has_been_called = False

    @cache_plot
    def plot_bh(self, *args, **kwargs) -> axes.Axes:
        return self.bh.plot(label='Buy&Hold', *args, **kwargs)
    
    def plot_strategy(self, *args, **kwargs) -> axes.Axes:
        keep, *skip = self.strategy()
        return keep.plot(label=self._name(), *args, **kwargs)

    def plot(self, *args, **kwargs):
        self.plot_bh(*args, **kwargs)
        self.plot_strategy(*args, **kwargs)

class CPPI(StrategyPlotter):
    _DEFAULT = {
        'M': 3,
        'rf': 0.03,
        'floor': 0.8,
        'type': 'static',
        'rebalance': 'W'
    }

    def __init__(
        self,
        ptf: Portfolio,
        par: Dict[str, Union[int, str, float]] = None
    ):
        super().__init__(ptf)
        par = {**self._DEFAULT, **(par or {})}
        self._m = par['M']
        self._rf = par['rf']
        self._daily_rf = np.expm1(np.log1p(self._rf)*(1/self._n_days))
        self._rebalance = par['rebalance']
        self._type = par['type']
        self._floor = par['floor']

    @property
    def _rebalance_dates(self):
        if isinstance(self._rebalance, str):
            return self.bh.index.to_period(self._rebalance).drop_duplicates().to_timestamp()
        elif isinstance(self._rebalance, int):
            return int(len(self.bh.index)/self._rebalance)
        else:
            raise TypeError("Configuration parameter 'rebalance' must be an integer or a string.")
        
    def _name(self):
        _m = "M=" + str(self._m)
        _r = "r=" + str(self._rf*100) + "%"
        _d = str(self._rebalance) + "-CPPI"
        _f = self._type + " " + str(self._floor*100) + "%"
        return _d + " (" + _m + ", " + _r + ", " + _f + ")"

    # FIXME: al momento il ribilanciamento avviene giornalmente qualsiasi sia il rebalance messo
    def strategy(self):
        """
        Docstrings.
        """
        # initial settings
        floor = self._floor # floor value (constant if type = static, else updated in loop)
        account_value = 1. # initial wealth value
        peak = 1. # if needed, initial peak is wealth value
        cushion = 1. - floor/account_value
        risk_w = np.maximum(np.minimum(self._m*cushion, 1), 0)
        risk_alloc = account_value*risk_w
        safe_alloc = account_value-risk_alloc

        # support lists to iterate to
        val_l = []
        floor_l = []

        for date in self.bh.index:
            # computation of propension to risk
            if self._type == 'max dd':
                # reset of floor value only in max dd floor CPPI
                peak = np.maximum(peak, account_value)
                floor = peak*self._floor

            # rebalancing of portfolio
            if date in self._rebalance_dates:
                cushion = 1 - floor/account_value # new cushion
                risk_w = np.maximum(np.minimum(self._m*cushion, 1), 0) # new risk allocation
                risk_alloc = account_value*risk_w
                safe_alloc = account_value-risk_alloc
            
            # update wealth according to risk-free and asset returns
            risk_alloc = risk_alloc*(1+self.rets[date])
            safe_alloc = safe_alloc*(1+self._daily_rf)
            account_value = risk_alloc + safe_alloc

            # append to support lists
            val_l += [account_value]
            floor_l += [floor]

        # returns tuple of wealth value for each date and floor value at that date
        vals = pd.Series(val_l, index=self.bh.index)
        floors = pd.Series(floor_l, index=self.bh.index)
        return vals, floors

    def plot_floor(self):
        self.strategy()[1].plot(label="CPPI floor")
