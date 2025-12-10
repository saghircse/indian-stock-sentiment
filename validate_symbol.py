import yfinance as yf

def validate_symbol(symbol: str):
    try:
        ticker = yf.Ticker(symbol + ".NS")  # NSE suffix
        info = ticker.info
        print(info)
        # If info has a name, symbol is valid
        if "longName" in info:
            return True, info["longName"]
        else:
            return False, None
    except Exception:
        return False, None

valid, company_name = validate_symbol("TCS")
print(valid, company_name)  # True, "Tata Consultancy Services Ltd"
