# BTC Price Movement Prediction – Project Documentation

## Overview
This project is an end-to-end machine learning pipeline that predicts the next-hour price movement of Bitcoin using real-time data. It includes:

- Live data ingestion  
- Feature engineering  
- Model training (XGBoost)  
- A Streamlit-based frontend  
- Docker deployment  
- A lightweight model monitoring dashboard  

---

## 1. Process Map  
![Pipeline](images/pipeline.png)

**Pipeline Steps**  
CryptoCompare API → Feature Engineering → XGBoost Model → Streamlit App → Docker Deployment → Monitoring Dashboard

Each step is modular and designed for retraining, model updates, and scaling.

---

## 2. Data Ingestion  
**Source:** [CryptoCompare API](https://min-api.cryptocompare.com/)

**Data Types:**
- Hourly OHLCV (Open, High, Low, Close, Volume)  
- Real-time price and volume (current data)  
- Frequency: Hourly rolling window  

---

## 3. Predictive Model  
**Model Type:** Binary classification (Up vs Down/Neutral)  
**Framework:** XGBoost  

**Features:**
- Momentum, rolling volatility, SMA-24, SMA-168  
- Return over 1h, 3h, and 6h  
- **Target:** Next-hour close minus current close (i.e., `close[t+1] - close[t]`)  
- **Threshold:** Price increase > $50 → class 1 (up)  

---

## 4. Streamlit Application  
**App File:** `btc_predictor.py`  

**Features:**
- Current hour prediction with confidence  
- Last 10 hourly predictions with actual vs predicted  
- In-progress candle display  
- Live BTC price + 24h change  

---

## 5. Docker Deployment  
Dockerized for platform-independent deployment

```bash
docker build -t btc-app .
docker run -p 8501:8501 btc-app
