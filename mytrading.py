import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from abc import ABC, abstractmethod

class TradingBackTester(ABC):
    """
    Classe virtuale per il backtesting di strategie di trading.
    """

    def __init__(self, ticker: str, asked_start: str, asked_end: str, params: tuple):
        """
        Costruttore della classe virtuale che si occupa di settare le variabili di inizio e fine
        periodo e di scaricare i dati da Yahoo Finance.
        
        Parametri
        ---------
        ticker: str
            Simbolo del ticker su Yahoo Finance
        asked_start: str
            Data richiesta di inizio della serie storica considerata
        asked_end: str
            Data richiesta di fine della serie storica considerata
        params: tuple
            Parametri del modello, dipendono dalla classe figlia che viene chiamata
        """
        self._ticker = ticker
        self._asked_start = asked_start
        self._asked_end = asked_end
        self._params = params
        self._check_model_params()
        self._rawdata = self._download_data().droplevel(1, axis=1)
        self._obtained_start = self._rawdata.index[0]
        self._obtained_end = self._rawdata.index[-1]
        self._buyhold = self._buyhold_strategy()
        self._strategy = None
        self._strategy_name = None

    @abstractmethod
    def _check_model_params(self):
        pass

    def _download_data(self):
        """
        Metodo privato di download dei dati tramite API di Yahoo Finance.
        """
        return yf.download(tickers=self._ticker, start=self._asked_start, end=self._asked_end, progress=False)

    def _buyhold_strategy(self):
        """
        Metodo privato che definisce i rendimenti logaritmici giornalieri e cumulati su tutte le
        date della serie storica corrispondente a una strategia "Buy & Hold" adottata tra la data di
        inizio e di fine della tal serie.
        """
        # copia dei dati raw in dataframe di appoggio
        df = self._rawdata.copy()
        # definizione dei rendimenti logaritmici della strategia "Buy & Hold"
        df["LogReturn"] = np.log(df["Close"]/df["Close"].shift(1))
        df["CumLogReturn"] = df.LogReturn.cumsum()
        # drop del primo elemento per il quale non è definito il rendimento
        df.dropna(inplace=True)
        return df
    
    @abstractmethod
    def _trading_strategy(self):
        pass

    def _strategy_results(self):
        years_passed = (self._obtained_end-self._obtained_start).days/365.24
        log_performance = self._strategy.CumStrategyReturn[-1]
        log_outperformance = log_performance - self._strategy.CumLogReturn[-1]
        max_drawdown = (self._strategy.CumStrategyReturn.cummax()-self._strategy.CumStrategyReturn).max()
        results_dict = {
            "LogReturn": {
                "Overall": log_performance,
                "Annualized": log_performance / years_passed
            },
            "Return": {
                "Overall": np.exp(log_performance) - 1,
                "Annualized": np.exp(log_performance / years_passed) - 1
            },
            "Outperformance": {
                "Overall": log_outperformance,
                "Annualized": log_outperformance / years_passed
            },
            "MaxDrawdown": max_drawdown
        }
        df = pd.DataFrame(results_dict)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.applymap(lambda x: f"{x * 100:.2f}%" if isinstance(x, (int, float)) else x)
        return df
    
    def get_ticker(self):
        return self._ticker
    
    def get_asked_start(self):
        return self._asked_start
    
    def get_asked_end(self):
        return self._asked_end
    
    def get_obtained_start(self):
        return self._obtained_start
    
    def get_obtained_end(self):
        return self._obtained_end
    
    def get_strategy_dataframe(self):
        return self._strategy
    
    def get_strategy_results(self):
        return self._strategy_results()
    
    def get_performance(self):
        return self._strategy.CumStrategyReturn[-1]
    
    def subset_buyhold(self, mask) -> None:
        self._buyhold = self._buyhold.iloc[mask,:]
    
    def plot_buyhold_return(self):
        self._strategy.CumLogReturn.plot(label="Buy&Hold")
        plt.title("Log returns " + self._ticker)
        plt.legend()
    
    def plot_strategy_return(self):
        self._strategy.CumStrategyReturn.plot(label=self._strategy_name)
        plt.title("Log returns " + self._ticker)
        plt.legend()

    def plot_strategy_position(self):
        mask_buy = self._strategy.Position == 1
        mask_sell = self._strategy.Position == -1
        fill_min = self._strategy.Close.min()
        fill_max = self._strategy.Close.max()
        self._strategy.Close.plot(label="Price")
        plt.fill_between(self._strategy.index, fill_min, fill_max, where=mask_buy, color="gray", alpha=0.2, label="Buy")
        plt.fill_between(self._strategy.index, fill_min, fill_max, where=mask_sell, color="white", alpha=0.2, label="Sell")
        plt.legend()
    
class SMABackTester(TradingBackTester):
    """
    Classe per il backtesting di strategie di trading SMA: è una sottoclasse di TradingBackTester.
    """

    def __init__(self, ticker, asked_start, asked_end, params):
        """
        Costruttore della classe che si occupa di settare i parametri della classe parent e...
        """
        super().__init__(ticker, asked_start, asked_end, params)
        self._sma_short = self._params[0]
        self._sma_long = self._params[1]
        self._strategy_name = "SMA (" + str(self._sma_short) + "-" + str(self._sma_long) + ")"
        self._strategy = self._trading_strategy()
    
    def _check_model_params(self) -> None:
        if len(self._params) != 2:
            raise KeyError("Errato numero di parametri.")
        if self._params[0] >= self._params[1]:
            raise KeyError("Finestra corta maggiore di quella lunga.")
            
    def _trading_strategy(self) -> pd.DataFrame:
        """
        Metodo privato di definizione della strategia di trading adottata dalla sottoclasse.
        """
        df = self._buyhold.copy()
        # definizione delle medie mobili
        df["SMAShort"] = df["Close"].rolling(self._sma_short, min_periods=1).mean()
        df["SMALong"] = df["Close"].rolling(self._sma_long, min_periods=1).mean()
        # definizione della posizione (short o long)
        df["Position"] = np.where(df.SMAShort>df.SMALong, 1, -1)
        # rendimento della strategia
        df["StrategyReturn"] = df.Position.shift(1)*df.LogReturn
        df["CumStrategyReturn"] = df.StrategyReturn.cumsum()
        return df
    
    def get_sma_short(self) -> int:
        return self._sma_short
    
    def get_sma_long(self) -> int:
        return self._sma_long
    
class MomentumBackTester(TradingBackTester):
    """
    Classe per il backtesting di strategie di trading Momentum: è una sottoclasse di
    TradingBackTester.
    """

    def __init__(self, ticker, asked_start, asked_end, params):
        """
        Costruttore della classe che si occupa di settare i parametri della classe parent e...
        """
        super().__init__(ticker, asked_start, asked_end, params)
        self._strategy_name = "Momentum (" + str(params[0]) + ")"
        self._window = params[0]
        self._strategy = self._trading_strategy()
    
    def _check_model_params(self) -> None:
        if len(self._params) != 1:
            raise KeyError("Errato numero di parametri.")
        
    def _trading_strategy(self) -> pd.DataFrame:
        """
        Metodo privato di definizione della strategia di trading adottata dalla sottoclasse.
        """
        df = self._buyhold.copy()
        # definizione della posizione (short o long)
        df["Position"] = np.where(df.LogReturn.rolling(self._window).mean()>0, 1, -1)
        # rendimento della strategia
        df["StrategyReturn"] = df.Position.shift(1)*df.LogReturn
        df["CumStrategyReturn"] = df.StrategyReturn.cumsum()
        return df
    
    def get_window(self) -> int:
        return self._window
    
class BollingerBackTester(TradingBackTester):
    """
    Classe per il backtesting di strategie di trading tramite bande di Bollinger: è una sottoclasse
    di TradingBackTester.
    """

    def __init__(self, ticker, asked_start, asked_end, params):
        """
        Costruttore della classe che si occupa di settare i parametri della classe parent e...
        """
        super().__init__(ticker, asked_start, asked_end, params)
        self._strategy_name = "Bollinger (" + str(params[0]) + "-" + str(params[1]) + ")"
        self._sma_window = params[0]
        self._n_sigma = params[1]
        self._strategy = self._trading_strategy()
    
    def _check_model_params(self) -> None:
        if len(self._params) != 2:
            raise KeyError("Errato numero di parametri.")
        
    def _trading_strategy(self) -> pd.DataFrame:
        """
        Metodo privato di definizione della strategia di trading adottata dalla sottoclasse.
        """
        df = self._buyhold.copy()
        # calcolo della media mobile e della deviazione standard
        df["SMA"] = df["Close"].rolling(self._sma_window, min_periods=1).mean()
        df["DevSt"] = df["Close"].rolling(self._sma_window, min_periods=1).std()
        # definizione delle bande di Bollinger
        df["LowerBand"] = df.SMA - self._n_sigma * df.DevSt
        df["UpperBand"] = df.SMA + self._n_sigma * df.DevSt
        # definizione della posizione (short o long)
        df["Position"] = np.where(df.Close < df.LowerBand, 1, None)
        df["Position"] = np.where(df.Close > df.UpperBand, -1, df.Position)
        df["Position"] = np.where(np.sign(df.Close-df.SMA) != np.sign(df.Close-df.SMA).shift(1), 0, df.Position)
        df["Position"] = df.Position.ffill()
        # rendimento della strategia
        df["StrategyReturn"] = df.Position.shift(1)*df.LogReturn
        df["CumStrategyReturn"] = df.StrategyReturn.cumsum()
        return df
    
    def get_sma_window(self) -> int:
        return self._sma_window
    
    def get_n_sigma(self) -> int:
        return self._n_sigma
