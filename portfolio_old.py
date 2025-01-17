import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.style.use('ggplot')
from asset import Asset
from typing import Dict, Union, List
from scipy.optimize import minimize
import yfinance as yf

def efficient_frontier(assets:list[Asset], n_samples: int):
    frontier_portfolios = _efficient_frontier_(assets, n_samples)
    single_asset_returns = [asset.results().loc['Annualized return'].iloc[0] for asset in assets]
    single_asset_volatilities = [asset.results().loc['Volatility'].iloc[0] for asset in assets]
    returns = [ptf.ptf_return for ptf in frontier_portfolios]
    volatilities = [ptf.ptf_volatility for ptf in frontier_portfolios]

    fig, ax = plt.subplots()
    ax.plot(volatilities, returns, label='Efficient frontier')
    ax.scatter(single_asset_volatilities, single_asset_returns)
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    for i, ass in enumerate(assets):
        plt.annotate(
            ass.ticker,
            xy=(single_asset_volatilities[i]+0.001, single_asset_returns[i]+0.001),
            fontsize=10
        )
    
    fig.suptitle('Efficient frontier')

class Portfolio:

    def __init__(self, assets: Union[List[Asset], Dict[Asset, float]]):
        if isinstance(assets, list):
            self._list_given_init_(assets)
        elif isinstance(assets, dict):
            self._dict_given_init_(assets)
        else:
            raise TypeError("Input must be a list of Asset or a dictionary of Asset and floats.")
        
    def _list_given_init_(self, assets: List[Asset]):
        if not all(isinstance(item, Asset) for item in assets):
            raise TypeError("All elements in the list must be instances of Asset.")
        weights = [1/len(assets) for _ in assets]
        self._dict_given_init_(dict(zip(assets, weights)))

    def _dict_given_init_(self, assets: Dict[Asset, float]):
        if not all(isinstance(key, Asset) and isinstance(value, (int, float)) for key, value in assets.items()):
            raise TypeError("Dictionary keys must be instances of Asset and values must be floats or ints.")
        
        # definisco la lista di Asset
        self.assets = [key for key in assets.keys()]
        self.n_assets = len(self.assets)

        # definisco i pesi come pd.Series indicizzata a ticker
        self.weights = pd.Series(
            [value for value in assets.values()],
            index=[ass.ticker for ass in self.assets]
        )

        # definisco i rendimenti giornalieri come pd.DataFrame indicizzato a data e incolonnato a ticker
        self.daily_returns = pd.DataFrame(
            [ass.df['Return'] for ass in self.assets],
            index=[ass.ticker for ass in self.assets]
        ).T.dropna()

        # definisco la matrice di covarianza tra gli assets
        self.daily_cov = self.daily_returns.cov()

        # definisco i rendimenti annualizzati come pd.Series indicizzata a ticker
        self.annualized_returns = pd.Series(
            [ass.results().loc['Annualized return'].iloc[0] for ass in self.assets],
            index=[ass.ticker for ass in self.assets]
        )

        # definisco il punto nel piano rischio-rendimento
        self.ptf_return, self.ptf_volatility = self._risk_return_point_()

    def _risk_return_point_(self) -> tuple:
        ret = self.weights.T @ self.annualized_returns
        vol = (self.weights.T @ self.daily_cov @ self.weights)**0.5
        return ret, vol

def _minimize_vol_(assets: list[Asset], target_return) -> Portfolio:
    """
    Funzione che, data una serie di assets e specificato un rendimento target, deve determinare i
    pesi necessari a comporre un portafoglio con gli stessi asset e che abbia il rendimento target.
    """

    init_guess = np.repeat(1/len(assets), len(assets))
    bounds = ((0., 1.),)*len(assets)

    # constraint
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {
        'type': 'eq',
        'fun': lambda weights: target_return - Portfolio(dict(zip(assets, weights))).ptf_return
    }

    # minimizzazione
    weights = minimize(
        lambda weights: Portfolio(dict(zip(assets, weights))).ptf_volatility,
        init_guess,
        method='SLSQP',
        options={'disp': False},
        constraints=(weights_sum_to_1,return_is_target),
        bounds=bounds
    )

    return Portfolio(dict(zip(assets, weights.x)))

def _efficient_frontier_(assets: list[Asset], n_samples: int) -> list[Portfolio]:
    """
    Funzione che deve restituire una lista di istanze di Portfolio (insieme di portafogli) ognuno
    dei quali a rendimento diverso abbia il set di pesi che minimizza la volatilità. Per farlo prende
    in input sempre la lista di asset dai quali vogliamo costruire i portafogli che stanno sulla
    frontiera efficiente. La lunghezza della lista va specificata qui con n_samples
    """

    # definizione della griglia dei rendimenti
    min_return = Portfolio(assets).annualized_returns.min() # partenza della griglia (minimo rendimento singolo asset)
    max_return = Portfolio(assets).annualized_returns.max() # arrivo della griglia (massimo rendimento singolo asset)
    target_returns = np.linspace(min_return, max_return, n_samples)

    # definizione dei portafoglio sulla frontiera efficiente
    portfolios = [_minimize_vol_(assets, target_return) for target_return in target_returns]
    return portfolios


