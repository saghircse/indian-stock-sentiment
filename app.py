import gradio as gr
import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime
import yfinance as yf

# -----------------------------
# LOAD TICKERS
# -----------------------------
tickers = pd.read_csv("tickers.csv")

def get_company_name(symbol: str):
    """
    Hybrid symbol validation:
    1️⃣ Check in tickers.csv
    2️⃣ If not found, check dynamically via yfinance
    """
    symbol = symbol.upper().strip()

    # 1️⃣ Check in CSV
    row = tickers[tickers["Symbol"] == symbol]
    if not row.empty:
        return True, row.iloc[0]["Symbol"], row.iloc[0]["Company"]

    # 2️⃣ Check via yfinance
    try:
        ticker = yf.Ticker(symbol + ".NS")
        info = ticker.info
        if "longName" in info:
            return True, symbol, info["longName"]
        else:
            return False, None, None
    except Exception:
        return False, None, None


# -----------------------------
# FETCH NEWS (FREE GOOGLE RSS)
# -----------------------------
def fetch_news(query):
    url = f"https://news.google.com/rss/search?q={query}+stock&hl=en-IN&gl=IN&ceid=IN:en"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "lxml-xml")

    items = soup.find_all("item")
    news_list = []
    for item in items[:20]:  # limit to 20 headlines
        news_list.append({
            "title": item.title.text,
            "link": item.link.text,
            "published": item.pubDate.text,
        })
    return news_list


# -----------------------------
# LOAD FinBERT MODEL
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

def analyze_sentiment(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True)
    output = model(**tokens)
    probs = torch.nn.functional.softmax(output.logits, dim=1)[0]
    sentiment = labels[torch.argmax(probs)]
    return sentiment, float(probs.max())


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def run_pipeline(user_input):
    valid, symbol, company = get_company_name(user_input)

    if not valid:
        return f"❌ '{user_input}' is not a valid NSE stock symbol.", ""

    query = company + " stock"
    news = fetch_news(query)

    if len(news) == 0:
        return f"No news found for {company}", ""

    results = []
    counts = {"positive": 0, "neutral": 0, "negative": 0}

    for article in news:
        text = article["title"]
        sent, conf = analyze_sentiment(text)
        counts[sent] += 1
        
        published_str = str(article.get("published", ""))
        results.append({
            "headline": str(text),
            "sentiment": str(sent),
            "confidence": float(round(conf, 3)),
            "published": published_str,
            "link": str(article.get("link", "")),
        })

    # -------------------
    # Sort by published date (newest first)
    # -------------------
    def parse_date(date_str):
        try:
            return datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")
        except:
            return datetime.min  # fallback if parsing fails
    
    results.sort(key=lambda x: parse_date(x["published"]), reverse=True)

    # -------------------
    # Sentiment summary
    # -------------------
    summary = f"""
### 📊 Sentiment Summary for {company} ({symbol})

| Sentiment | Count |
|----------|-------|
| 😊 Positive | {counts['positive']} |
| 😐 Neutral | {counts['neutral']} |
| 😞 Negative | {counts['negative']} |
"""

    # -------------------
    # Build HTML table with highlights
    # -------------------
    table_html = "<table style='width:100%; border-collapse: collapse;'>"
    table_html += "<tr><th>Headline</th><th>Sentiment</th><th>Confidence</th><th>Published</th><th>Link</th></tr>"

    color_map = {
        "positive": "#00C853",  # green
        "neutral": "#9E9E9E",   # gray
        "negative": "#D50000",  # red
    }

    LOW_CONFIDENCE_THRESHOLD = 0.6

    for r in results:
        color = color_map.get(r["sentiment"], "#000000")
        if r["confidence"] < LOW_CONFIDENCE_THRESHOLD:
            highlight_style = "background-color:#FFF59D; cursor: help;"
            tooltip = "⚠️ Low confidence: model prediction may be unreliable"
        else:
            highlight_style = ""
            tooltip = ""

        table_html += f"<tr style='{highlight_style}' title='{tooltip}'>"
        table_html += f"<td>{r['headline']}</td>"
        table_html += f"<td style='color:{color}; font-weight:bold'>{r['sentiment'].capitalize()}</td>"
        table_html += f"<td>{r['confidence']}</td>"
        table_html += f"<td>{r['published']}</td>"
        table_html += f"<td><a href='{r['link']}' target='_blank'>Open</a></td>"
        table_html += "</tr>"

    table_html += "</table>"

    return summary, table_html


# -----------------------------
# GRADIO UI
# -----------------------------
ui = gr.Interface(
    fn=run_pipeline,
    inputs=gr.Textbox(label="Enter Indian Stock Symbol"),
    outputs=[gr.Markdown(), gr.HTML()],
    title="🇮🇳 Indian Stock Market Sentiment Analyzer",
    description="Enter an NSE stock symbol. Uses FinBERT + Google News to generate sentiment."
)

ui.launch()
