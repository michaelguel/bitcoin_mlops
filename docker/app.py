from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Load model
model = joblib.load("./btc_xgb_classifier.pkl")

# Define feature order (must match training)
FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume_btc",
    "volume_usd",
    "momentum",
    "volatility_24h",
    "sma_24",
    "sma_168",
    "return_1h",
    "return_3h",
    "return_6h",
]


# Function to fetch most recent hour
# @st.cache_data(show_spinner=False)
def fetch_latest_hour():
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    params = {
        "fsym": "BTC",
        "tsym": "USD",
        "limit": 190,  # increased to ensure rolling features are valid
    }
    response = requests.get(url, params=params)
    data = response.json()["Data"]["Data"]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s")
    df.rename(
        columns={"volumefrom": "volume_btc", "volumeto": "volume_usd"}, inplace=True
    )
    return df


def fetch_current_btc_price():
    url = "https://min-api.cryptocompare.com/data/pricemultifull"
    params = {"fsyms": "BTC", "tsyms": "USD"}
    r = requests.get(url, params=params)
    data = r.json()["RAW"]["BTC"]["USD"]
    return {
        "price": data["PRICE"],
        "change_24h": data["CHANGE24HOUR"],
        "volume_24h": data["TOTALVOLUME24H"],
    }


# Preprocessing for model input
def prepare_features(df):
    df["momentum"] = df["close"] - df["open"]
    df["volatility_24h"] = df["close"].rolling(window=24).std()
    df["sma_24"] = df["close"].rolling(window=24).mean()
    df["sma_168"] = df["close"].rolling(window=168).mean()
    df["return_1h"] = df["close"].pct_change(1)
    df["return_3h"] = df["close"].pct_change(3)
    df["return_6h"] = df["close"].pct_change(6)

    # Drop NaNs AFTER computing features
    df = df.dropna().copy()

    return df


# App UI
st.title("BTC Price Movement Predictor")  # 📈
st.markdown("Using XGBoost to predict the probability of BTC going up in the next hour")

# === Live BTC Price ===
st.subheader("Live BTC Price (Real-Time)")  # 💸
live_data = fetch_current_btc_price()

st.metric("Price (USD)", f"${live_data['price']:,.2f}")
st.metric("24h Change", f"{live_data['change_24h']:,.2f}")
st.metric("24h Volume", f"{live_data['volume_24h']:,.2f}")


# === Section: Refresh button to update data ===
# === Section: Refresh button ===
if st.button("Refresh Data"):  # 🔄
    # Refresh and store everything in session
    st.session_state["btc_hourly_data"] = fetch_latest_hour()
    st.session_state["btc_live_data"] = fetch_current_btc_price()


# Fetch or fallback (initial page load)
# On initial load if session isn't initialized
if "btc_hourly_data" not in st.session_state:
    st.session_state["btc_hourly_data"] = fetch_latest_hour()
if "btc_live_data" not in st.session_state:
    st.session_state["btc_live_data"] = fetch_current_btc_price()


# Load from session
df_raw = st.session_state["btc_hourly_data"]
live_data = st.session_state["btc_live_data"]


# Preprocess data
df_x = prepare_features(df_raw)

# Split data for prediction (complete) vs display (partial)
df = df_x.iloc[:-1].copy()  # last complete hour
df_partial = df_raw.iloc[-1:].copy()  # raw in-progress row (always shown)

# Calculate target and actual direction
df["target"] = df["close"].shift(-1) - df["close"]
threshold = 50
df["actual_direction"] = df["target"].apply(lambda x: 1 if x > threshold else 0)

# ✅ Always show current hour (in-progress)
st.subheader("Current Hour (Partial)")  # 🕒
st.dataframe(df_partial[["timestamp", "close", "volume_btc", "high", "low"]])
st.caption(
    "This row shows the current hour-in-progress. Predictions are made only on complete hourly candles."
)

# Proceed only if we have enough rows
if len(df) >= 11:
    df_recent = df.iloc[-11:].copy()
    df_hist = df_recent.iloc[:-1]
    df_current = df_recent.iloc[-1:]

    # Predict current hour
    X_current = df_current[FEATURES]
    proba = model.predict_proba(X_current)[0]
    label = model.predict(X_current)[0]

    st.subheader("Current Hour Prediction")  # 📍
    st.metric("Prediction", "↑ Up" if label == 1 else "↓ Down/Neutral")
    st.metric("Confidence", f"{proba[label]:.2f}%")
    st.dataframe(
        df_current[["timestamp", "close", "volume_btc", "momentum", "volatility_24h"]]
    )

    # Predict past 10 hours
    st.subheader("Last 10 Hourly Predictions")  # 📊
    X_hist = df_hist[FEATURES]
    hist_probs = model.predict_proba(X_hist)
    hist_preds = model.predict(X_hist)
    hist_confidence = [hist_probs[i][hist_preds[i]] for i in range(len(hist_preds))]

    hist_results = df_hist[["timestamp", "close", "actual_direction"]].copy()
    hist_results["predicted_direction"] = hist_preds
    hist_results["predicted_confidence"] = hist_confidence
    hist_results["predicted_label"] = hist_results["predicted_direction"].map(
        {1: "↑ Up", 0: "↓ Down/Neutral"}
    )
    hist_results["actual_label"] = hist_results["actual_direction"].map(
        {1: "↑ Up", 0: "↓ Down/Neutral"}
    )

    st.dataframe(
        hist_results[
            [
                "timestamp",
                "close",
                "actual_label",
                "predicted_label",
                "predicted_confidence",
            ]
        ].reset_index(drop=True)
    )
else:
    st.warning("Not enough valid data rows to make predictions.")
