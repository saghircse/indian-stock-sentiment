---
title: Indian Stock Sentiment
emoji: 🐢
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 6.1.0
app_file: app.py
pinned: false
---

# 🇮🇳 Indian Stock Market Sentiment Analyzer

A Gradio-based app to fetch **Indian stock news** and analyze **sentiment** using **FinBERT**.  

Enter an NSE stock symbol to get **recent headlines** along with:

- ✅ **Sentiment** (Positive / Neutral / Negative)  
- ✅ **Confidence scores**  
- ✅ **Clickable links** to news  
- ✅ **Publication dates** (sorted newest first)  
- ✅ **Summary table** with counts, percentages, and weighted sentiment  
- ✅ **Charts**:
  - Daily headline counts by sentiment  
  - Daily sentiment trend  
  - Stock price + sentiment trend  
- ✅ **Downloadable CSV** of all news and sentiment scores  

---

## How It Works

1. Users input a **stock symbol** (like `TCS` or `RELIANCE`).  
2. Symbol validation:
   - **Local CSV lookup** for popular symbols (~100 symbols).  
   - **Dynamic check via Yahoo Finance** for other NSE symbols.  
3. Fetches news from **Google News RSS (India)**.  
4. Filters news by **user-selected period**:  
   - Last 7 days  
   - Last 10 days  
   - Last 1 month  
5. Analyzes sentiment with **FinBERT (ProsusAI model)**.  
6. Generates **interactive outputs**:
   - Summary Markdown table  
   - Colored HTML table of headlines  
   - Sentiment & headline count charts  
   - Stock price + sentiment chart  
   - CSV download link  

---

## Understanding the UI Calculations

The summary table shows several key metrics for the fetched headlines:

| Metric | Description |
|--------|------------|
| **Count** | Number of headlines predicted as Positive / Neutral / Negative. |
| **%** | Percentage of headlines in each sentiment category. |
| **Avg Sentiment (Overall)** | Average **overall score** for headlines in this category. Calculated as **positive probability − negative probability** per headline, then averaged. |
| **Weighted Count** | Sum of the raw sentiment probabilities for each category across all headlines. Provides a “confidence-weighted” measure of sentiment dominance. |
| **Overall Score (per headline)** | `positive − negative` probability. Shows whether the headline is more positive or negative. |
| **Dominant Sentiment (per headline)** | The sentiment with the **highest probability** among positive, neutral, or negative. Highlighted in the table. |

**Charts:**
- **Daily Headline Counts**: Shows how many headlines per day fall into each sentiment.  
- **Daily Sentiment Trend**: Shows average overall sentiment per day (positive − negative).  
- **Stock Price + Sentiment Trend**: Plots stock closing price alongside daily sentiment for easy correlation.  

---

## App Features

- Select **number of headlines** to fetch (20–100).  
- Highlights the **dominant sentiment** for each headline.  
- Alternating row colors in the headlines table for readability.  
- Charts are dynamically generated for the selected period.  

---

## Notes

- Low-confidence predictions are included but can be inferred from the weighted counts.  
- Google News RSS may have rate limits; for large-scale usage, consider caching or using a news API.  
- Built with **Gradio + Transformers + yfinance + pandas + matplotlib + BeautifulSoup**.  

---

### Quick Start

1. Run `app.py`.  
2. Enter a **stock symbol** in the input box.  
3. Select the **period** and **number of headlines** to fetch.  
4. Click **Analyze**.  
5. View **summary**, **charts**, **headlines table**, and download the CSV.

## Links / Deployment

- **GitHub Repository:** [https://github.com/saghircse/indian-stock-sentiment](https://github.com/saghircse/indian-stock-sentiment)  
- **Hugging Face Space:** [https://huggingface.co/spaces/saghircse/indian-stock-sentiment](https://huggingface.co/spaces/saghircse/indian-stock-sentiment)  
- **Live App (Gradio Web):** [https://saghircse-indian-stock-sentiment.hf.space/](https://saghircse-indian-stock-sentiment.hf.space/)  

