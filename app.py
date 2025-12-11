import gradio as gr
import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64
import html

# -----------------------------
# LOAD LOCAL TICKERS
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
# FETCH NEWS
# -----------------------------
def fetch_news(query, max_items=50):
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
    fmts = ["%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"]
    for fmt in fmts:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            pass
    return None

# -----------------------------
# FILTER NEWS BY PERIOD
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
def run_pipeline(user_input, period_option, max_news):

    # Extract symbol if user selected "SYMBOL - Company"
    if " - " in user_input:
        raw_symbol = user_input.split(" - ")[0].strip()
    else:
        raw_symbol = user_input.strip()

    period_map = {"Last 7 days": 7, "Last 10 days": 10, "Last 1 month": 30}
    period_days = period_map.get(period_option, 7)

    valid, symbol, company = get_company_name(raw_symbol)
    if not valid:
        return f"❌ '{user_input}' is not a valid NSE stock symbol.", "", "", "", "", ""

    # Fetch news
    query = company + " stock"
    news, total_items = fetch_news(query, max_items=int(max_news))
    news = filter_news_by_period(news, period_days=period_days)

    fetched_count = min(total_items, int(max_news))

    if len(news) == 0:
        return f"No news found for {company} in {period_option}", "", "", "", "", ""

    info_msg = f"**Showing {len(news)} headlines from the last {period_days} days (fetched {fetched_count} / requested {max_news}).**"

    # Sentiment
    texts = [n["title"] for n in news]
    sentiments = analyze_sentiment_batch(texts)

    results = []
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    overall_sums = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    weighted_counts = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    date_sentiments = {}
    date_counts = {}

    for item, sent in zip(news, sentiments):
        pos, neu, neg = sent["positive"], sent["neutral"], sent["negative"]
        overall = round(pos - neg, 3)

        pred = max(["positive", "neutral", "negative"], key=lambda k: sent[k])
        counts[pred] += 1
        overall_sums[pred] += overall

        for k in ["positive", "neutral", "negative"]:
            weighted_counts[k] += sent[k]

        dt = parse_date(item["published"])
        if dt:
            dkey = dt.date().isoformat()
            date_sentiments.setdefault(dkey, []).append(overall)
            date_counts.setdefault(dkey, []).append(pred)

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

    avg_overall = {k: (overall_sums[k] / counts[k] if counts[k] > 0 else 0.0)
                   for k in counts}

    # -----------------------------
    # SUMMARY
    # -----------------------------
    summary = f"""
### 📊 Sentiment Summary for {company} ({symbol}) — {period_option}

| Sentiment | Count | % | Avg Sentiment (Overall) | Weighted Count |
|----------|-------|------|--------------|-----------|
| 😊 Positive | {counts['positive']} | {counts['positive']/len(news)*100:.1f}% | {avg_overall['positive']:.2f} | {weighted_counts['positive']:.2f} |
| 😐 Neutral | {counts['neutral']} | {counts['neutral']/len(news)*100:.1f}% | {avg_overall['neutral']:.2f} | {weighted_counts['neutral']:.2f} |
| 😞 Negative | {counts['negative']} | {counts['negative']/len(news)*100:.1f}% | {avg_overall['negative']:.2f} | {weighted_counts['negative']:.2f} |

**Headlines in period:** {len(news)}  
"""

    # -----------------------------
    # HEADLINES TABLE
    # -----------------------------
    table = """
    <table style='width:100%; border-collapse: collapse;'>
    <tr style='background-color:#f2f2f2'>
        <th>Published</th>
        <th>Headline</th>
        <th>Positive</th>
        <th>Neutral</th>
        <th>Negative</th>
        <th>Overall (POS-NEG)</th>
        <th>Link</th>
    </tr>
    """

    for i, r in enumerate(results):
        row_color = "#ffffff" if i % 2 == 0 else "#f9f9f9"

        # Determine highest sentiment
        max_sent = max(["positive", "neutral", "negative"], key=lambda k: r[k])

        # Colors
        color_map = {"positive": "green", "neutral": "gray", "negative": "red"}

        # Helper: build cell HTML
        def cell_html(value, sent_type):
            if max_sent == sent_type:
                return f"<td style='color:{color_map[sent_type]}; font-weight:bold'>{value}</td>"
            else:
                return f"<td>{value}</td>"

        color = "green" if r["overall"] > 0 else "red" if r["overall"] < 0 else "black"

        table += f"<tr style='background-color:{row_color};'>"
        table += f"<td>{r['published']}</td>"
        table += f"<td>{html.escape(r['headline'])}</td>"
        table += cell_html(r['positive'], "positive")
        table += cell_html(r['neutral'], "neutral")
        table += cell_html(r['negative'], "negative")
        table += f"<td style='color:{color}; font-weight:bold'>{r['overall']}</td>"
        table += f"<td><a href='{r['link']}' target='_blank'>Open</a></td>"
        table += "</tr>"

    table += "</table>"

    # -----------------------------
    # CSV DOWNLOAD
    # -----------------------------
    df = pd.DataFrame(results)
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_data = "data:text/csv;base64," + base64.b64encode(csv_buf.getvalue().encode()).decode()
    csv_link_html = f"<a href='{csv_data}' download='{symbol}_news_sentiment.csv'>⬇️ Download CSV</a>"

    # -----------------------------
    # CHARTS
    # -----------------------------
    chart_counts_html = chart_sentiment_html = chart_price_sentiment_html = ""

    if date_sentiments:
        dates = sorted(date_sentiments.keys())
        avg_sentiments = [sum(date_sentiments[d])/len(date_sentiments[d]) for d in dates]
        pos_counts = [date_counts[d].count("positive") for d in dates]
        neu_counts = [date_counts[d].count("neutral") for d in dates]
        neg_counts = [date_counts[d].count("negative") for d in dates]

        # Chart 1
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.bar(dates, pos_counts, color="green")
        ax1.bar(dates, neu_counts, bottom=pos_counts, color="gray")
        ax1.bar(dates, neg_counts, bottom=[p + n for p, n in zip(pos_counts, neu_counts)], color="red")
        ax1.set_xlabel("")
        ax1.set_title("Daily Headline Counts")
        ax1.tick_params(axis='x', rotation=60, labelsize=7)
        plt.tight_layout()
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format="png")
        buf1.seek(0)
        chart_counts_html = f"<img src='data:image/png;base64,{base64.b64encode(buf1.read()).decode()}' style='width:100%; max-width:700px;'/>"
        plt.close(fig1)

        # Chart 2
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        colors = ["green" if x > 0 else "red" if x < 0 else "gray" for x in avg_sentiments]
        ax2.bar(dates, avg_sentiments, color=colors)
        ax2.axhline(0, color="black", linestyle="--")
        ax2.set_title("Daily Sentiment Trend")
        ax2.tick_params(axis='x', rotation=60, labelsize=7)
        plt.tight_layout()
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png")
        buf2.seek(0)
        chart_sentiment_html = f"<img src='data:image/png;base64,{base64.b64encode(buf2.read()).decode()}' style='width:100%; max-width:700px;'/>"
        plt.close(fig2)

        # Chart 3 - Stock Price + Sentiment
        try:
            ticker_data = yf.Ticker(symbol + ".NS").history(period=f"{period_days}d")

            if not ticker_data.empty:
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                ax3.plot(ticker_data.index, ticker_data['Close'], color="blue", label="Close Price")

                sentiment_dates = [pd.to_datetime(d) for d in dates]
                ax3_twin = ax3.twinx()
                ax3_twin.plot(sentiment_dates, avg_sentiments, color="orange", marker="o", label="Sentiment Score")

                lines1, labels1 = ax3.get_legend_handles_labels()
                lines2, labels2 = ax3_twin.get_legend_handles_labels()
                ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

                ax3.set_title("Daily Stock Price + Sentiment Trend")
                ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
                fig3.autofmt_xdate()

                plt.tight_layout()
                buf3 = io.BytesIO()
                fig3.savefig(buf3, format="png")
                buf3.seek(0)
                chart_price_sentiment_html = f"<img src='data:image/png;base64,{base64.b64encode(buf3.read()).decode()}' style='width:100%; max-width:700px;'/>"
                plt.close(fig3)

        except Exception:
            chart_price_sentiment_html = ""

    return (
        summary,
        info_msg,
        chart_counts_html,
        chart_sentiment_html,
        table + "<br>" + csv_link_html,
        chart_price_sentiment_html
    )

# -----------------------------
# GRADIO UI
# -----------------------------
with gr.Blocks(title="Indian Stock Market Sentiment Analyzer") as ui:

    gr.Markdown("<h1 style='text-align:center;'>🇮🇳 Indian Stock Market Sentiment Analyzer</h1>")
    gr.Markdown("<p style='text-align:center;'>Enter an NSE/BSE stock symbol. The app uses FinBERT + Google News to generate sentiment analysis of recent headlines.</p>")

    with gr.Row():
        with gr.Column(scale=1):
            symbol_in = gr.Textbox(label="Enter Stock Symbol (e.g., RELIANCE, TCS)")
            period_in = gr.Dropdown(
                ["Last 7 days", "Last 10 days", "Last 1 month"],
                value="Last 7 days",
                label="Select Period"
            )
            max_news_in = gr.Slider(
                minimum=20, maximum=100, step=1, value=50,
                label="Number of Headlines to Fetch"
            )
            btn = gr.Button("Analyze")

        with gr.Column(scale=2):
            summary_out = gr.Markdown()
            info_out = gr.Markdown()

    with gr.Row():
        chart1_out = gr.HTML(label="Daily Headline Counts")
        chart2_out = gr.HTML(label="Daily Sentiment Trend")
        chart3_out = gr.HTML(label="Daily Stock Price + Sentiment Trend")

    table_out = gr.HTML(label="Headlines Table")

    btn.click(
        run_pipeline,
        inputs=[symbol_in, period_in, max_news_in],
        outputs=[summary_out, info_out, chart1_out, chart2_out, table_out, chart3_out]
    )

ui.launch()
