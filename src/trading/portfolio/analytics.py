import numpy as np
from matplotlib import pyplot as plt
from .core import Tickers, Portfolio, WEIGHT_BOUNDS

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
