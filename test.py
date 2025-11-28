from backtesting.singleasset import SMA
from backtesting.cppi import CPPI
from portfolio import Tickers, Portfolio
from marketdata.fmp import FMPProvider

def main():
    from portfolio import Tickers

    # classe istanziata fornendo una lista di asset, un periodo ed eventualmente un provider di dati
    titoli = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'META']
    dict_tickers = {"tickers": titoli, 'period': '1y'}
    tickers_yf = Tickers(**dict_tickers)

    SMA(tickers_yf).plot()

if __name__ == "__main__":
    main()
