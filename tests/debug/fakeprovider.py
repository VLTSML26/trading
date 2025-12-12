import sys
from pathlib import Path
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from trading.marketdata import DEFAULT_FAKEDATA
from trading.portfolio import Tickers, get_capw

def main():
    t = Tickers(["AAPL", "MSFT"], period="max", provider=DEFAULT_FAKEDATA)
    t.plot()
    t.plot_dailyret_dist()

if __name__ == "__main__":
    main()
