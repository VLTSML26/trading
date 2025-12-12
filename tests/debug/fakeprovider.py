
import sys
from pathlib import Path
from matplotlib import pyplot as plt

# add `<repo>/src` to sys.path so `import trading` works when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from trading.marketdata.fake import FakeProvider
from trading.portfolio import Tickers

def main():
    fake = FakeProvider(
        annual_rets={"AAPL": 0., "MSFT": 0.18},
        annual_vol={"AAPL": 0.2, "MSFT": 0.1},
        start_prices={"AAPL": 100.0, "MSFT": 200.0},
        base_mkcaps={"AAPL": 2.5e11, "MSFT": 2.0e11},
        seed=123
    )

    t = Tickers(["AAPL", "MSFT"], period="max", provider=fake)
    t.plot()
    print(t.measured_var())
    plt.show()
    breakpoint()

if __name__ == "__main__":
    main()
