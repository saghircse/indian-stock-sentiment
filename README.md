# 🇮🇳 Indian Stock Market Sentiment Analyzer

A simple app to fetch **Indian stock news** and analyze **sentiment** using **FinBERT**.  

**Enter an NSE stock symbol** to get the latest headlines with:  

- ✅ **Sentiment** (Positive / Neutral / Negative)  
- ✅ **Confidence score**  
- ✅ **Clickable links** to news  
- ✅ **Publication dates** (sorted newest first)  
- ⚠️ Low-confidence predictions highlighted  

---

## How It Works

1. Users input a **stock symbol** (like `TCS` or `RELIANCE`).  
2. Symbol validation:
   - **Local CSV lookup** for popular symbols (~100 symbols).  
   - **Dynamic check via Yahoo Finance** for other NSE symbols.  
3. Fetches news from **Google News RSS** (India).  
4. Analyzes sentiment with **FinBERT** (ProsusAI model).  

---

## Notes

- Low-confidence threshold: **0.6**  
- News limited to **latest 20 headlines**  
- Built with **Gradio + Transformers + yfinance + pandas**  

---

### Quick Start

Just enter the **symbol** in the input box and hit **Submit**. Results are displayed instantly.
