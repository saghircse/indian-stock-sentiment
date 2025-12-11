import gradio as gr
import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64

# -----------------------------
# LOAD TICKERS
# -----------------------------
tickers = pd.read_csv("tickers.csv")

def get_company_name(symbol: str):
    symbol = symbol.upper().strip()
    row = tickers[tickers["Symbol"] == symbol]
    if not row.empty:
        return True, row.iloc[0]["Symbol"], row.iloc[0]["Company"]
    try:
        ticker = yf.Ticker(symbol + ".NS")
        info = ticker.info
        if "longName" in info:
            return True, symbol, info["longName"]
    except Exception:
        pass
    return False, None, None

# -----------------------------
# FETCH NEWS (RSS)
# -----------------------------
def fetch_news(query, max_items=100):
    url = f"https://news.google.com/rss/search?q={query}+stock&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
    except Exception:
        return [], 0

    soup = BeautifulSoup(r.text, "lxml-xml")
    items = soup.find_all("item")
    news_list = []

    for item in items[:max_items]:
        news_list.append({
            "title": item.title.text,
            "link": item.link.text,
            "published": item.pubDate.text,
        })

    return news_list, len(items)

# -----------------------------
# DATE PARSER
# -----------------------------
def parse_date(date_str):
    if not date_str:
        return None
    fmts = [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z"
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            pass
    return None

# -----------------------------
# FILTER BY PERIOD
# -----------------------------
def filter_news_by_period(news_list, period_days=7):
    cutoff = datetime.utcnow() - timedelta(days=period_days)
    filtered = []
    for item in news_list:
        dt = parse_date(item.get("published", ""))
        if dt and dt.replace(tzinfo=None) >= cutoff:
            filtered.append(item)
    return filtered

# -----------------------------
# LOAD FINBERT
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

def analyze_sentiment_batch(texts):
    batch = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**batch)
        probs = torch.nn.functional.softmax(output.logits, dim=1)

    results = []
    for p in probs:
        results.append({
            "positive": float(p[labels.index("positive")]),
            "neutral": float(p[labels.index("neutral")]),
            "negative": float(p[labels.index("negative")]),
        })
    return results

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def run_pipeline(user_input, period_option):
    period_map = {
        "Last 7 days": 7,
        "Last 10 days": 10,
        "Last 1 month": 30
    }
    period_days = period_map.get(period_option, 7)

    # Validate symbol
    valid, symbol, company = get_company_name(user_input)
    if not valid:
        return f"❌ '{user_input}' is not a valid NSE stock symbol.", "", "", "", ""

    query = company + " stock"
    news, total_items = fetch_news(query, max_items=100)
    news = filter_news_by_period(news, period_days=period_days)

    if len(news) == 0:
        return f"No news found for {company} in {period_option}", "", "", "", ""

    info_msg = (
        f"**Showing {len(news)} out of {total_items} headlines fetched (max 100) "
        f"from the {period_option.lower()}.**"
    )

    # Sentiment
    texts = [n["title"] for n in news]
    sentiments = analyze_sentiment_batch(texts)

    results = []
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    date_sentiments = {}
    date_counts = {}

    for item, sent in zip(news, sentiments):
        pos, neu, neg = sent["positive"], sent["neutral"], sent["negative"]
        overall = round(pos - neg, 3)

        # predicted class
        pred = max(["positive", "neutral", "negative"], key=lambda k: sent[k])
        counts[pred] += 1

        dt = parse_date(item["published"])
        if not dt:
            continue
        date_key = dt.date().isoformat()

        date_sentiments.setdefault(date_key, []).append(overall)
        date_counts.setdefault(date_key, []).append(pred)

        results.append({
            "headline": item["title"],
            "positive": round(pos, 3),
            "neutral": round(neu, 3),
            "negative": round(neg, 3),
            "overall": overall,
            "published": item["published"],
            "link": item["link"]
        })

    results.sort(key=lambda x: parse_date(x["published"]), reverse=True)

    # -----------------------------
    # Summary
    # -----------------------------
    summary = f"""
### 📊 Sentiment Summary for {company} ({symbol}) — {period_option}

| Sentiment | Count | Percentage |
|----------|-------|------------|
| 😊 Positive | {counts['positive']} | {counts['positive']/len(news)*100:.1f}% |
| 😐 Neutral | {counts['neutral']} | {counts['neutral']/len(news)*100:.1f}% |
| 😞 Negative | {counts['negative']} | {counts['negative']/len(news)*100:.1f}% |

**Total Headlines Fetched:** {total_items}  
**Headlines in Period:** {len(news)}
"""

    # -----------------------------
    # HTML TABLE
    # -----------------------------
    table = "<table style='width:100%; border-collapse: collapse;'>"
    table += "<tr><th>Headline</th><th>Positive</th><th>Neutral</th><th>Negative</th><th>Overall</th><th>Published</th><th>Link</th></tr>"

    for r in results:
        color = "green" if r["overall"] > 0 else "red" if r["overall"] < 0 else "black"
        table += f"<tr>"
        table += f"<td>{r['headline']}</td>"
        table += f"<td>{r['positive']}</td>"
        table += f"<td>{r['neutral']}</td>"
        table += f"<td>{r['negative']}</td>"
        table += f"<td style='color:{color}; font-weight:bold'>{r['overall']}</td>"
        table += f"<td>{r['published']}</td>"
        table += f"<td><a href='{r['link']}' target='_blank'>Open</a></td>"
        table += "</tr>"

    table += "</table>"

    # -----------------------------
    # CHARTS
    # -----------------------------
    chart_counts_html = ""
    chart_sentiment_html = ""

    if len(date_sentiments) > 0:
        dates = sorted(date_sentiments.keys())
        avg_sentiments = [sum(date_sentiments[d]) / len(date_sentiments[d]) for d in dates]

        pos_counts = [date_counts[d].count("positive") for d in dates]
        neu_counts = [date_counts[d].count("neutral") for d in dates]
        neg_counts = [date_counts[d].count("negative") for d in dates]

        # ---- HEADLINE COUNTS (no labels) ----
        fig1, ax1 = plt.subplots(figsize=(6,4))
        ax1.bar(dates, pos_counts, color="green", label="Positive")
        ax1.bar(dates, neu_counts, bottom=pos_counts, color="gray", label="Neutral")
        ax1.bar(dates, neg_counts, bottom=[p+n for p,n in zip(pos_counts, neu_counts)], color="red", label="Negative")
        ax1.tick_params(axis='x', rotation=60, labelsize=7)
        ax1.set_title("Daily Headline Counts")
        ax1.set_ylabel("Headlines")
        ax1.legend()
        plt.tight_layout()

        buf1 = io.BytesIO()
        fig1.savefig(buf1, format="png")
        buf1.seek(0)
        chart_counts_html = f"<img src='data:image/png;base64,{base64.b64encode(buf1.read()).decode()}' style='width:100%;'/>"
        plt.close(fig1)

        # ---- SENTIMENT TREND (smaller rotated labels) ----
        fig2, ax2 = plt.subplots(figsize=(6,4))
        colors = ["green" if x > 0 else "red" if x < 0 else "gray" for x in avg_sentiments]
        ax2.bar(dates, avg_sentiments, color=colors)
        ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax2.tick_params(axis='x', rotation=60, labelsize=7)
        ax2.set_title("Daily Sentiment Trend (Overall)")
        ax2.set_ylabel("Sentiment Score")

        # Add  ALL labels (small font)
        for i, val in enumerate(avg_sentiments):
            ax2.text(
                i,
                val + (0.01 if val >= 0 else -0.02),
                f"{val:.2f}",
                ha='center', 
                va='bottom' if val >= 0 else 'top',
                fontsize=7
            )

        plt.tight_layout()

        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png")
        buf2.seek(0)
        chart_sentiment_html = f"<img src='data:image/png;base64,{base64.b64encode(buf2.read()).decode()}' style='width:100%;'/>"
        plt.close(fig2)

    return summary, info_msg, chart_counts_html, chart_sentiment_html, table


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title="🇮🇳 Indian Stock Market Sentiment Analyzer") as ui:

    gr.Markdown("<h1 style='text-align:center'>🇮🇳 Indian Stock Market Sentiment Analyzer</h1>")
    gr.Markdown("<p style='text-align:center'>Enter an NSE stock symbol. The app uses FinBERT + Google News to generate sentiment analysis of recent headlines.</p>")

    # First row: input + summary
    with gr.Row():
        with gr.Column(scale=1):
            symbol_in = gr.Textbox(label="Enter Indian Stock Symbol")
            period_in = gr.Dropdown(
                ["Last 7 days","Last 10 days","Last 1 month"],
                value="Last 7 days",
                label="Select Period"
            )
            btn = gr.Button("Analyze")
        with gr.Column(scale=2):
            summary_out = gr.Markdown()

    info_out = gr.Markdown()

    with gr.Row():
        chart1_out = gr.HTML()
        chart2_out = gr.HTML()

    table_out = gr.HTML()

    btn.click(
        run_pipeline,
        inputs=[symbol_in, period_in],
        outputs=[summary_out, info_out, chart1_out, chart2_out, table_out]
    )

ui.launch()
