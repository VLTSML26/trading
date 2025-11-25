from baseclass import Strategy
from wrappers import single_asset_portfolio

class SMA(Strategy):
    _DEFAULT = {
        'M': 3,
        'rf': 0.03,
        'floor': 0.8,
        'type': 'static',
        'rebalance': 'W'
    }
    def __init__(
        self,
        ticker: str,
        start: str,
        end: str,
        short: int,
        long: int
    ):
        # call the wrapper inside the constructor
        super().__init__(single_asset_portfolio(ticker, start, end))
        self.short = short
        self.long = long
