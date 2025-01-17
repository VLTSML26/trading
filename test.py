from asset import Asset
from portfolio_old import Portfolio

def main():
    assets = [Asset({'ticker': 'AAPL'}), Asset()]
    p = Portfolio(assets)

if __name__ == "__main__":
    main()
