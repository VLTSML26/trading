
# Trading Toolkit

Toolkit Python per testare strategie di trading e data-visualization, organizzato con layout moderno `src/`.
Sviluppato 
Include provider di dati **reali** (es. FMP) e **fittizi** (`FakeProvider`) basato su **Monte Carlo**.

---

## Installazione

> **Prerequisito**: Python ≥ 3.10

```bash
git clone https://github.com/VLTSML26/trading.git
cd trading
# per adesso la versione installabile è in un altro branch
git checkout package

python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

pip install -U pip
pip install -e .
```

**Nota:** `-e` (editable) installa il pacchetto puntando alla cartella `src/`. Ok per sviluppo e debug.

---

## Configurazione (.env)

Per usare provider **reali** (es. FMP), crea `.env` nella root e inserisci:

```
FMP_API_KEY=chiave_api_personale
```

Il codice carica `.env` con `python-dotenv`.
Se usi solo `FakeProvider`, **non serve** alcuna API key.

---

## Uso rapido

### Provider fittizio (FakeProvider)

```python
from trading.marketdata.fake import FakeProvider
from trading.portfolio.core import Tickers, Portfolio

fake = FakeProvider(seed=123)
t = Tickers(["AAPL","MSFT"], period="6mo", provider=fake)
ptf = Portfolio(t, None, "EqW")

print("Sharpe:", ptf.sharpe_ratio(0.03))
```

### Provider FMP (Financial Modeling Prep)

```python
from trading.marketdata.fmp import FMPProvider
from trading.portfolio.core import Tickers

fmp = FMPProvider()  # legge la chiave da .env
t = Tickers(["AAPL","MSFT"], start="2024-01-01", end="2024-09-30", provider=fmp)
print(t.daily_returns.tail())
```

### Portfolio & Analytics

```python
from trading.portfolio.core import Tickers, Portfolio
from trading.portfolio.analytics import get_msr, get_gmv, get_eqw, get_capw

t = Tickers(["AAPL","MSFT"], period="1y", provider=fake)
ptf_msr = get_msr(t, rf=0.03)
ptf_gmv = get_gmv(t)
ptf_eqw = get_eqw(t)
ptf_capw = get_capw(t)

print("MSR Sharpe:", ptf_msr.sharpe_ratio(0.03))
```

## Licenza

MIT.
