import sys
from pathlib import Path
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from trading.portfolio import get_parity_ptf, Tickers, Portfolio
from trading.marketdata import FMPProvider

def main():
    t = Tickers(["AAPL", "MSFT", "GOOGL", "META"], period="1y", provider=FMPProvider())
    parity_ptf = get_parity_ptf(t)
    print(parity_ptf.hhi_enb)
    print(Portfolio.risk_contribution(parity_ptf.covmat, parity_ptf.weights))

if __name__ == "__main__":
    main()
