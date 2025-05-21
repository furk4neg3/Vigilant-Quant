# Vigilant Quant

> **Institution-grade** Stock Trading AI  
> Predict 1-day-ahead returns using price + news sentiment, with explainability, risk controls, backtesting, and a production-ready dashboard.

---

## 🏢 Overview

Vigilant Quant is a modular, end-to-end trading system designed for real-world deployment in institutional environments.  
- **Predictive Engine**: 1-day-ahead price-direction forecasts via LightGBM (Phase 1) and Temporal Fusion Transformer (Phase 2)  
- **Sentiment Signal**: Daily FinBERT-scored news headlines  
- **Explainable AI**: SHAP-powered local & global explanations integrated into dashboard  
- **Risk Management**: Position sizing, drawdown limits, and volatility targeting  
- **Backtesting**: Realistic slippage, latency, and P&L metrics (Sharpe, Max DD, Calmar)  
- **Dashboard**: Streamlit app for signal monitoring, explanations, and simulation results  
- **Containerization**: Docker + Docker-Compose for reproducible builds  
- **CI/CD Ready**: Placeholder for GitHub Actions or Jenkins pipelines  


## ⚙️ Prerequisites

- Docker & Docker-Compose  
- Python 3.10+  
- Alpaca API key (for future paper trading)  
- Internet access for fetching market data & news  

---

## 🚀 Installation & Quick Start

1. **Clone & cd**  
   ```bash
   git clone https://github.com/your-org/quantex.git](https://github.com/furk4neg3/Vigilant-Quant.git
   cd vigilant-quant
Environment
Copy .env.example → .env and fill in:
TICKERS=AAPL,MSFT,GOOG
ALPACA_API_KEY=…
ALPACA_SECRET_KEY=…
NEWS_API_KEY=…
Build & Run
‣ Docker
docker-compose up --build
‣ Native

pip install -r requirements.txt
python data/data_pipeline.py
python sentiment/news_sentiment.py
python features/build_features.py
python models/train_lightgbm.py
streamlit run dashboard/app.py
Inspect
Open http://localhost:8501 in your browser.
