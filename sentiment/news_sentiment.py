import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from utils.config import (
    TICKERS,
    START_DATE,
    END_DATE,
    CACHE_ENABLED,
    LOG_LEVEL,
    NEWS_API_KEY,
    NEWS_API_ENDPOINT,
    SENTIMENT_DIR,
)

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Fetch Headlines Function ─────────────────────────────────────────────────
def fetch_headlines(symbol, from_date, to_date, page_size=100):
    """
    Fetch all headlines for `symbol` between from_date and to_date.
    Returns a list of dicts: {'headline', 'publishedAt'}.
    """
    headlines = []
    page = 1

    while True:
        params = {
            "q": symbol,
            "from":   from_date,
            "to":     to_date,
            "language":"en",
            "pageSize": page_size,
            "page":    page,
            "apiKey":  NEWS_API_KEY,
            "sortBy":  "relevancy",
        }
        resp = requests.get(NEWS_API_ENDPOINT, params=params)
        if resp.status_code != 200:
            logger.error(f"[ERROR] NewsAPI {symbol} page {page}: {resp.text}")
            break

        data = resp.json()
        articles = data.get("articles", [])
        if not articles:
            break

        for art in articles:
            headlines.append({
                "headline":   art["title"].strip(),
                "publishedAt": art.get("publishedAt", "")
            })

        total = data.get("totalResults", 0)
        if page * page_size >= total:
            break
        page += 1

    # Dedupe by (headline, timestamp)
    unique = { (h["headline"], h["publishedAt"]) : h for h in headlines }
    return list(unique.values())


# ─── Sentiment Scoring Setup ──────────────────────────────────────────────────
logger.info("Loading FinBERT model for sentiment scoring…")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
nlp       = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


# ─── Main Pipeline ─────────────────────────────────────────────────────────────
def main():
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting sentiment pipeline for {len(TICKERS)} tickers")

    for symbol in TICKERS:
        out_path = SENTIMENT_DIR / f"{symbol}_sentiment.csv"
        if CACHE_ENABLED and out_path.exists():
            logger.info(f"[SKIP] {symbol}: cached sentiment at {out_path}")
            continue

        logger.info(f"[FETCH] Headlines for {symbol}")
        raw = fetch_headlines(symbol, START_DATE, END_DATE)
        if not raw:
            logger.warning(f"[EMPTY] No headlines for {symbol}")
            continue

        df = pd.DataFrame(raw)
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
        df.dropna(subset=["publishedAt"], inplace=True)
        df.sort_values("publishedAt", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Batch sentiment scoring
        logger.info(f"[SCORE] Scoring {len(df)} headlines for {symbol}")
        scores = []
        for head in df["headline"].tolist():
            res = nlp(head)[0]
            score = res["score"] * (1 if res["label"] == "Positive" else -1)
            scores.append(score)

        df["sentiment_score"] = scores

        # Save to CSV
        df.to_csv(out_path, index=False)
        logger.info(f"[SAVED] {symbol}: {len(df)} rows → {out_path}")

    logger.info("Sentiment pipeline complete!")


if __name__ == "__main__":
    if not NEWS_API_KEY:
        logger.error("Missing NEWS_API_KEY in environment. Exiting.")
    else:
        main()
