from portfolio import Portfolio, CPPI

def main():
    sp_500 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JPM', 'JNJ', 'V', 'PG', 'TSLA']
    ew = Portfolio(sp_500, period="1y")
    ews_d = CPPI(ew, {'floor': 0.80, 'rebalance': 'W', 'type': 'max dd'})
    ews_m = CPPI(ew, {'floor': 0.80, 'rebalance': 'M', 'type': 'max dd'})
    ews_d.plot(legend=True)
    ews_m.plot(legend=True)

if __name__ == "__main__":
    main()
