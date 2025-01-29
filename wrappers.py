from functools import wraps

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
