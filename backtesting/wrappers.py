import numpy as np
from functools import wraps
from portfolio import Portfolio

def cache_plot(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # check if the method has already been called for this class
        if not self.__class__.called:
            self.__class__.called = True  # mark as called
            return func(self, *args, **kwargs)
        else:
            print(f"Plotting method '{func.__name__}' has already been called for this class. Skipping plot.")
    return wrapper

def single_asset_portfolio(ticker: str, start: str, end: str) -> Portfolio:
    p = Portfolio(ticker, period=None)
    p.df = p.history(start=start, end=end).Close.to_frame()
    p._w = np.array([1.])
    return p
