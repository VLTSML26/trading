import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt; plt.style.use('ggplot')
from typing import Dict, Union
from scipy.stats import norm

def _modified_var_(returns: pd.Series, alpha: float):
        z = norm.ppf(alpha)
        s = returns.skew()
        k = returns.kurtosis()
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*k/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
        return -(returns.mean() + z*returns.std(ddof=0))

class Asset:
    """
    Classe Asset: wrappa un pandas.DataFrame ma purtroppo non riesco ad ereditare
    perchè c'è un errore di massima ricorsione quando viene chiamato il costruttore
    di pandas.DataFrame.
    """
    DEFAULT_PARAMS = {
        'ticker': 'EURUSD=X',
        'start': '1980-01-01',
        'end': '2020-01-01',
        'price': 'Close',
        'perc_var': 0.1
    }
    
    def __init__(self, par: Dict[str, Union[str, float]] = None):
        """
        Costruttore della classe.

        Parametri
        ---------
        ticker: str
            Simbolo del ticker su Yahoo Finance
        start: str
            Data richiesta di inizio della serie storica considerata
        end: str
            Data richiesta di fine della serie storica considerata
        price: str
            Prezzo da indicare tra Open, High, Low e Close
        perc_var: float
            Ordine alpha del quantile mediante il quale calcolare il VaR
        """
        par = {**self.DEFAULT_PARAMS, **(par or {})}
        self.ticker = par['ticker']
        self.start = par['start']
        self.end = par['end']
        self.perc_var = par['perc_var']
        self.price = par['price']
        self.df = self._get_dataframe_()
        self.start = self.df.index[0]
        self.end = self.df.index[-1]
    
    def __repr__(self):
         return f'Asset(ticker={self.ticker}, start={self.start}, end={self.end}, price={self.price})'
    
    def _get_dataframe_(self):
        df = self._download_data_()
        df["Return"] = np.log(df.Price/df.Price.shift(1))
        df.dropna(inplace=True)
        df["CumReturn"] = df.Return.cumsum()
        df['CumVolatility'] = df.Return.expanding().std(ddof=0)
        df['CumSemiDeviation'] = df.Return.expanding().apply(lambda x:x[x<0].std(ddof=0))
        df['Drawdown'] = df.CumReturn.cummax()-df.CumReturn
        df['CumVaRH'] = df.Return.expanding().apply(lambda x: -np.percentile(x, self.perc_var*100))
        df['CumVaRMod'] = df.Return.expanding().apply(lambda x: _modified_var_(x, self.perc_var))
        return df
    
    def results(self):
        index = ['Total return', 'Annualized return', 'Volatility', 'Max Drawdown', 'Historic VaR']
        values = [
            self.df.CumReturn.iloc[-1],
            self.df.CumReturn.iloc[-1] / (self.end-self.start).days * 365.24,
            self.df.CumVolatility.iloc[-1],
            self.df.Drawdown.max(),
            self.df.CumVaRH.iloc[-1]
        ]
        return pd.DataFrame(data=values, index=index, columns=[self.ticker])

    def _download_data_(self):
        """
        Metodo privato di download dei dati tramite API di Yahoo Finance.
        """
        raw_df = yf.download(
            tickers=self.ticker,
            start=self.start,
            end=self.end,
            progress=False
        )
        return pd.DataFrame(
            data=raw_df[self.price].values,
            index=raw_df.index,
            columns=['Price']
        )

    def plot(self, *args):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0, 'height_ratios':[3,1]}, *args)

        ax1.plot(self.df.CumReturn, c='k', label='Return')
        ax1.twinx().bar(self.df.index, self.df.CumVolatility, label='Volatility')
        ax1.set_ylabel('Risk-return time serie')
        ax1.legend()
        
        ax2.plot(-self.df.Drawdown, c='r', lw=0.5)
        ax2.set_ylabel('Drawdown')
        
        fig.suptitle(self.ticker)
    