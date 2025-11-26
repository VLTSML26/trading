import numpy as np
from functools import wraps
from portfolio import Portfolio
from matplotlib import pyplot as plt

def cache_plot_once_per_figure(func):
    """
    Decorator che esegue la funzione al massimo 1 volta per figura.
    Resetta il cache quando viene creata una nuova figura.
    """
    cache = {}
    
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        fig = plt.gcf()
        fig_id = id(fig)
        
        # Se questa figura non è stata vista prima, esegui la funzione
        if fig_id not in cache:
            cache[fig_id] = True
            result = func(self, *args, **kwargs)
            return result
        
        return None
    
    return wrapper

def cache_plot_once_per_object(func):
    """
    Decorator che esegue la funzione al massimo 1 volta per istanza della classe.
    """
    attr_name = f"_{func.__name__}_has_run"

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if getattr(self, attr_name, False):
            # already called for THIS instance
            return getattr(self, f"_{func.__name__}_result")
        
        # run and cache result
        result = func(self, *args, **kwargs)
        setattr(self, attr_name, True)
        setattr(self, f"_{func.__name__}_result", result)
        return result

    return wrapper

def cache_plot_once_per_class(func):
    """
    Decorator che esegue la funzione al massimo 1 volta per classe.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # check if the method has already been called for this class
        if not self.__class__.called:
            self.__class__.called = True  # mark as called
            return func(self, *args, **kwargs)
        else:
            print(f"Plotting method '{func.__name__}' has already been called for this class. Skipping plot.")
    return wrapper

def single_asset_portfolio(ticker: str, period: str) -> Portfolio:
    p = Portfolio(ticker, period=period)
    p._w = np.array([1.])
    return p
