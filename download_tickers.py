import pandas as pd
import yfinance as yf
import requests
from io import StringIO

# -----------------------------
# 1️⃣ Download NSE EQUITY_L.csv
# -----------------------------
url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
}

resp = requests.get(url, headers=headers)
if resp.status_code != 200:
    raise Exception("Failed to download EQUITY_L.csv from NSE.")

df = pd.read_csv(StringIO(resp.text))

# Strip whitespace from column names
df.columns = df.columns.str.strip()

print(f"Columns in CSV: {df.columns.tolist()}")

# -----------------------------
# 2️⃣ Keep only equity securities (filter by SERIES or equivalent column)
# -----------------------------
# Check if 'SERIES' column exists, else try 'Series'
if "SERIES" in df.columns:
    equity_df = df[df["SERIES"] == "EQ"].copy()
elif "Series" in df.columns:
    equity_df = df[df["Series"] == "EQ"].copy()
else:
    # If no SERIES column, take all (fallback)
    print("No SERIES column found, taking all rows as equity")
    equity_df = df.copy()

print(f"Total equity securities: {len(equity_df)}")

# Limit to 1000 symbols
equity_df = equity_df.head(10)

# Prepare symbol list for yfinance
symbol_col = "SYMBOL" if "SYMBOL" in equity_df.columns else "Symbol"
equity_df["YTICKER"] = equity_df[symbol_col].astype(str) + ".NS"
symbols = equity_df["YTICKER"].tolist()

# -----------------------------
# 3️⃣ Fetch company names using yfinance
# -----------------------------
tickers_data = []

for symbol in symbols:
    try:
        t = yf.Ticker(symbol)
        info = t.info
        name = info.get("longName") or info.get("shortName") or symbol
        tickers_data.append({"Symbol": symbol.split(".")[0], "Company": name})
        print(f"Added: {symbol} -> {name}")
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")

# -----------------------------
# 4️⃣ Save to tickers.csv
# -----------------------------
df_out = pd.DataFrame(tickers_data)
df_out.to_csv("tickers2.csv", index=False)
print(f"\n✅ Saved {len(df_out)} tickers to tickers.csv")
