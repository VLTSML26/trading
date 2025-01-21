import copy
import yfinance
import numpy as np
import pandas as pd
from typing import Union, Any
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
        return (1+self.daily_returns).cumprod()-1
    
    @property
    def drawdown(self) -> pd.DataFrame:
        """
        Drawdown giornaliero calcolato per ogni data della serie storica.
        """
        return self.returns.cummax()-self.returns
    
    @property
    def comp_returns(self) -> pd.Series:
        """
        Rendimento composto dall'inizio del periodo di osservazione calcolato su tutto il periodo.
        """
        return (1+self.daily_returns).prod()-1
    
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
        return self.daily_returns.std()*((365.24/days_passed)**0.5)
    
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
    
    def sharpe_ratio(self, risk_free_rate: float) -> pd.Series:
        return (self.annual_returns - risk_free_rate) / self.annual_volatility
    
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
    def covmat(self) -> pd.DataFrame:
        return self.daily_returns.cov()

    @property
    def ptf_volatility(self) -> float:
        return (self.weights.T @ self.covmat @ self.weights)**0.5

    def minimize_vol(self, target_return: float) -> None:
        """
        Metodo che resetta i pesi del portafoglio minimizzandone la volatilità per un dato
        rendimento. Viene minimizzata la forma quadratica sqrt(w^T C w) mediante l'utilizzo della
        libreria minimize di scipy.optimize.
        
        Parametri
        ---------
        target_return: float
            Rendimento target per il quale si desidera trovare i pesi che minimizzano la volatilità.
        """
        def to_min(w):
            return (w.T @ self.covmat @ w)**0.5
        
        def update_w_and_match_ret(w):
            self.weights = w # qui avviene il reset dei pesi con il Setter della classe
            return target_return - self.ptf_return

        w0 = self.weights.values # proposta iniziale
        bounds = self._WEIGHT_BOUNDS*self.n_assets # vincoli
        weights_sum_to_1 = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1} # constraint somma pesi
        return_is_target = {'type': 'eq', 'fun': update_w_and_match_ret} # constraint rendimento

        # minimizzazione
        from scipy.optimize import minimize
        _ = minimize(
            to_min,
            w0,
            method='SLSQP',
            options={'disp': False},
            constraints=(weights_sum_to_1, return_is_target),
            bounds=bounds
        )

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

    fig, ax = plt.subplots()
    ax.plot(volatilities, returns, label='Efficient frontier')
    ax.scatter(ptfs[0].annual_volatility, ptfs[0].annual_returns, label='Single assets')
    
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    fig.suptitle('Risk-return space')

    plt.legend()
