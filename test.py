from portfolio import Tickers
from marketdata.fmp import FMPProvider

def main():
    yf = Tickers(["AAPL", "TSLA"])
    fmp = Tickers(["AAPL", "TSLA"], provider=FMPProvider())
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
