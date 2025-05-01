import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# Load pre-trained XGBoost classifier
model = XGBClassifier()
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "btc_xgb_classifier.pkl"))
model = joblib.load(model_path)
# model = joblib.load("../docker/btc_xgb_classifier.pkl")


# ─── 2. Feature Definitions ─────────────────────────────────────────────────────
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


# ─── 3. PSI Helper ──────────────────────────────────────────────────────────────
def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) between two arrays.
    PSI > 0.25 indicates major drift; 0.1-0.25 moderate; <0.1 negligible.
    """
    # Determine common bin edges from the expected (baseline) distribution
    bin_edges = np.histogram_bin_edges(expected, bins=bins)
    exp_freq = np.histogram(expected, bins=bin_edges)[0] / len(expected)
    act_freq = np.histogram(actual, bins=bin_edges)[0] / len(actual)
    # Replace zeros to avoid log(0)
    exp_freq = np.where(exp_freq == 0, 1e-6, exp_freq)
    act_freq = np.where(act_freq == 0, 1e-6, act_freq)
    # Sum over bins
    return float(np.sum((exp_freq - act_freq) * np.log(exp_freq / act_freq)))


# ─── 4. Data Fetching ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_latest_hour(limit: int = 500) -> pd.DataFrame:
    """
    Fetch the last `limit` hours of BTC OHLCV data from CryptoCompare.
    """
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    params = {"fsym": "BTC", "tsym": "USD", "limit": limit}
    raw_data = requests.get(url, params=params).json()["Data"]["Data"]
    df = pd.DataFrame(raw_data)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s")
    # Rename columns to match FEATURES
    df.rename(
        columns={"volumefrom": "volume_btc", "volumeto": "volume_usd"}, inplace=True
    )
    return df


@st.cache_data
def fetch_current_btc_price() -> dict:
    """
    Fetch the current BTC price, 24h change, and 24h volume from CryptoCompare.
    """
    url = "https://min-api.cryptocompare.com/data/pricemultifull"
    params = {"fsyms": "BTC", "tsyms": "USD"}
    raw = requests.get(url, params=params).json()["RAW"]["BTC"]["USD"]
    return {
        "price": raw["PRICE"],
        "change_24h": raw["CHANGE24HOUR"],
        "volume_24h": raw["TOTALVOLUME24H"],
    }


@st.cache_data
def load_baseline_training_data(
    path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "btc_hourly_ohlc_volume_1year_cryptocompare.csv"))
) -> pd.DataFrame:
    """
    Load and preprocess the training data to use as baseline for drift analysis.
    Applies the same rolling features and log1p transforms as used in training.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])

    df["momentum"] = df["close"] - df["open"]
    df["volatility_24h"] = df["close"].rolling(window=24).std()
    df["sma_24"] = df["close"].rolling(window=24).mean()
    df["sma_168"] = df["close"].rolling(window=168).mean()
    df["return_1h"] = df["close"].pct_change(1)
    df["return_3h"] = df["close"].pct_change(3)
    df["return_6h"] = df["close"].pct_change(6)

    df = df.dropna().reset_index(drop=True)

    df["log_vol"] = np.log(df["volume_btc"].replace(0, np.nan)).fillna(method="bfill")
    df["Volume Change % (BTC)"] = df["volume_btc"].pct_change()

    df = df.copy()
    # 1‑bar log‑return for O & C
    df["log_ret_open"] = np.log(df["open"]).diff()
    df["log_ret_close"] = np.log(df["close"]).diff()

    # Range as % of close
    df["hl_spread_pct"] = (df["high"] - df["low"]) / df["close"]
    # Momentum (example: 24‑bar price diff) z‑scored over 30 bars
    df["mom_24"] = df["close"] - df["close"].shift(24)
    df["mom_24_z"] = (df["mom_24"] - df["mom_24"].rolling(24).mean()) / df[
        "mom_24"
    ].rolling(24).std()

    # Volatility rescaled
    df["vol24_rel"] = df["volatility_24h"] / df["volatility_24h"].rolling(24).median()

    # Price‑vs‑moving‑averages ratios
    df["ratio_sma_24"] = np.log(df["close"] / df["sma_24"])
    df["ratio_sma_168"] = np.log(df["close"] / df["sma_168"])

    for feat in FEATURES:
        df[feat] = df[feat].clip(lower=0)  # avoid log1p of negative
        df[feat] = np.log1p(df[feat])

    df.dropna(inplace=True)

    return df.tail(180 * 24)  # last 180 days (hourly)


# ─── 5. Feature Engineering ─────────────────────────────────────────────────────
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given raw OHLCV, compute rolling features and the binary direction label.
    """
    df = df.copy()
    # Momentum = close minus open
    df["momentum"] = df["close"] - df["open"]
    # 24h volatility (std), simple moving averages
    df["volatility_24h"] = df["close"].rolling(24).std()
    df["sma_24"] = df["close"].rolling(24).mean()
    df["sma_168"] = df["close"].rolling(168).mean()
    # Returns over various horizons
    df["return_1h"] = df["close"].pct_change(1)
    df["return_3h"] = df["close"].pct_change(3)
    df["return_6h"] = df["close"].pct_change(6)

    # Drop any rows with NaNs from rolling ops
    df = df.dropna().reset_index(drop=True)

    # Next-hour regression target and binary label
    df["next_delta"] = df["close"].shift(-1) - df["close"]
    df["actual_direction"] = (df["next_delta"] > 50).astype(int)

    return df


# ─── 6. Performance DataFrame ──────────────────────────────────────────────────
@st.cache_data
def load_performance_dataframe(limit: int = 500) -> pd.DataFrame:
    """
    Build a DataFrame of the last `limit` complete hours,
    with true labels and model predictions.
    """
    raw = fetch_latest_hour(limit + 1)
    features_df = compute_features(raw)
    df_recent = features_df.iloc[-(limit + 1) : -1].copy()

    # Model predictions
    proba = model.predict_proba(df_recent[FEATURES])[:, 1]
    df_recent["hist_probs"] = proba
    df_recent["hist_preds"] = (proba > 0.5).astype(int)

    return df_recent


# ─── 7. Streamlit App Layout ────────────────────────────────────────────────────
st.title("BTC Price Movement Predictor")
st.markdown("Using XGBoost to predict whether BTC will rise in the next hour")

# Create two main tabs
predict_tab, monitor_tab = st.tabs(["⬆⬇ Predictor", "Model Monitoring"])

# ─── 7a. Predictor Tab ─────────────────────────────────────────────────────────
with predict_tab:
    st.header("Live BTC Price & Next-Hour Signal")

    # Display current price metrics
    live_price = fetch_current_btc_price()
    st.metric("Price (USD)", f"${live_price['price']:,.2f}")
    st.metric("24h Δ (USD)", f"{live_price['change_24h']:,.2f}")
    st.metric("24h Volume", f"{live_price['volume_24h']:,.2f}")

    # Refresh button for OHLCV
    if st.button("Refresh OHLCV"):
        st.session_state["ohlcv_data"] = fetch_latest_hour()

    # Load or fetch into session state
    if "ohlcv_data" not in st.session_state:
        st.session_state["ohlcv_data"] = fetch_latest_hour()
    df_raw = st.session_state["ohlcv_data"]

    # Prepare features
    df_x = compute_features(df_raw)
    df = df_x.iloc[:-1]
    df_partial = df_raw.iloc[-1:]

    # Show in‑progress candle
    st.subheader("Current Hour (In-Progress)")
    st.dataframe(df_partial[["timestamp", "close", "volume_btc", "high", "low"]])
    st.caption("Predictions use only fully completed hourly candles.")

    # If we have ≥11 hours, make the next‑hour prediction
    if len(df) >= 11:
        df_recent = df.iloc[-11:]
        df_hist = df_recent.iloc[:-1]
        df_current = df_recent.iloc[-1:]

        # Next‑hour inference
        X_current = df_current[FEATURES]
        proba = model.predict_proba(X_current)[0, 1]
        label = model.predict(X_current)[0]

        st.subheader("Next-Hour Forecast")
        st.metric("Direction", "↑ Up" if label == 1 else "↓ Down/Neutral")
        st.metric("Confidence", f"{proba*100:.2f}%")

        # Show last 10 predictions for reference
        st.subheader("Recent 10-Hour Signals")
        X_hist = df_hist[FEATURES]
        probs_hist = model.predict_proba(X_hist)[:, 1]
        preds_hist = model.predict(X_hist)
        confidences = [probs_hist[i] * 100 for i in range(len(probs_hist))]

        display_df = df_hist[["timestamp", "close", "actual_direction"]].copy()
        display_df["hist_preds"] = preds_hist
        display_df["predicted_confidence"] = confidences
        display_df["Predicted Label"] = display_df["hist_preds"].map(
            {1: "↑ Up", 0: "↓ Down"}
        )
        display_df["Actual Label"] = display_df["actual_direction"].map(
            {1: "↑ Up", 0: "↓ Down"}
        )

        st.dataframe(
            display_df[
                [
                    "timestamp",
                    "close",
                    "Actual Label",
                    "Predicted Label",
                    "predicted_confidence",
                ]
            ].reset_index(drop=True)
        )
    else:
        st.warning("Not enough historical data (need at least 11 complete hours).")


# ─── 7b. Model Monitoring Tab ───────────────────────────────────────────────────
with monitor_tab:
    perf_df = load_performance_dataframe().set_index("timestamp")
    perf_subtab, drift_subtab, pred_subtab = st.tabs(
        ["Performance", "Data Drift", "Predictions"]
    )

    # — Performance —
    with perf_subtab:
        st.subheader("Performance Monitoring")
        st.markdown("Track classification metrics once ground-truth arrives.")

        # True vs. predicted directions
        y_true = perf_df["actual_direction"]
        y_pred = perf_df["hist_preds"]

        # Compute metrics
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred),
            "ROC-AUC": roc_auc_score(y_true, perf_df["hist_probs"]),
        }
        cols = st.columns(len(metrics))
        for col, (name, val) in zip(cols, metrics.items()):
            col.metric(name, f"{val:.2f}")

        # Rolling percentage of ups
        st.subheader("Rolling Percentage of Ups in the Last 24-hours")
        roll_rates = (
            perf_df[["hist_preds", "actual_direction"]].rolling("24h").mean().mul(100)
        )
        fig, ax = plt.subplots()
        sns.lineplot(data=roll_rates, ax=ax, dashes=False)
        ax.set_ylabel("% Up")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

        # Sample count
        num_samples = len(y_true)
        st.write(f"ℹ️ Evaluating on **{num_samples}** samples")

        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=["Down", "Up"],
            yticklabels=["Down", "Up"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    # — Data Drift Sub‑Tab —
    with drift_subtab:
        st.subheader("Data (Input) Monitoring")
        st.markdown("Detect feature distribution drift with PSI & KS test.")

        with st.expander("ℹ️ What are PSI & KS?"):
            st.latex(
                r"""
                \mathrm{PSI}
                = \sum_{i=1}^{k} \bigl(p_i - q_i\bigr)
                \ln\!\biggl(\frac{p_i}{q_i}\biggr)
                """
            )
            st.markdown(
                """
                **Population Stability Index (PSI)**
                - \(p_i\): proportion in bin *i* of the **baseline** data
                - \(q_i\): proportion in bin *i* of the **recent** data

                **Interpretation:**
                - **PSI < 0.10** → Negligible drift
                - **0.10 ≤ PSI ≤ 0.25** → Moderate drift
                - **PSI > 0.25** → Major drift
                """
            )

            st.markdown("**Kolmogorov-Smirnov (KS) Test**")
            st.latex(
                r"""
                D = \sup_x \bigl|F_{\mathrm{baseline}}(x) - F_{\mathrm{recent}}(x)\bigr|
                """
            )
            st.markdown(
                """
                - **\(D\)**: max distance between the two empirical CDFs
                - **p-value**: chance of seeing a \(D\) this large if distributions were the same

                **Interpretation:**
                - **p < 0.05** → reject “no drift” at the 5% level
                - Larger \(D\) means greater distributional difference
                """
            )

        # Split baseline vs. last 24 hrs
        # Load training-based baseline (180-day) and current live data
        baseline_df = load_baseline_training_data()
        recent_24h_df = perf_df[FEATURES].iloc[-168:].copy()

        # Apply log1p to match training transformation
        log_baseline = baseline_df.copy()
        log_recent = recent_24h_df.copy()

        log_recent["log_vol"] = np.log(
            log_recent["volume_btc"].replace(0, np.nan)
        ).fillna(method="bfill")
        log_recent["Volume Change % (BTC)"] = log_recent["volume_btc"].pct_change()

        log_recent = log_recent.copy()
        # 1‑bar log‑return for O & C
        log_recent["log_ret_open"] = np.log(log_recent["open"]).diff()
        log_recent["log_ret_close"] = np.log(log_recent["close"]).diff()

        # Range as % of close
        log_recent["hl_spread_pct"] = (
            log_recent["high"] - log_recent["low"]
        ) / log_recent["close"]
        # Momentum (example: 24‑bar price diff) z‑scored over 30 bars
        log_recent["mom_24"] = log_recent["close"] - log_recent["close"].shift(24)
        log_recent["mom_24_z"] = (
            log_recent["mom_24"] - log_recent["mom_24"].rolling(30).mean()
        ) / log_recent["mom_24"].rolling(24).std()

        # Volatility rescaled
        log_recent["vol24_rel"] = (
            log_recent["volatility_24h"]
            / log_recent["volatility_24h"].rolling(24).median()
        )

        # Price‑vs‑moving‑averages ratios
        log_recent["ratio_sma_24"] = np.log(log_recent["close"] / log_recent["sma_24"])
        log_recent["ratio_sma_168"] = np.log(
            log_recent["close"] / log_recent["sma_168"]
        )

        for feat in FEATURES:
            log_recent[feat] = log_recent[feat].clip(lower=0)
            log_recent[feat] = np.log1p(log_recent[feat])
        log_recent.dropna(inplace=True)

        # Compute PSI & KS, build summary table
        drift_summary = []
        FEATURES = [
            "momentum",
            "return_1h",
            "return_3h",
            "Volume Change % (BTC)",
            "log_ret_close",
            "log_ret_open",
        ]
        for feat in FEATURES:
            psi_val = psi(log_baseline[feat][60:].values, log_recent[feat][60:].values)
            ks_stat, ks_p = ks_2samp(log_baseline[feat][60:], log_recent[feat][60:])
            if psi_val > 0.25:
                status = "🔴 Major drift"
            elif psi_val > 0.10 or ks_p < 0.05:
                status = "🟡 Moderate drift"
            else:
                status = "🟢 OK"
            drift_summary.append(
                {
                    "Feature": feat,
                    "PSI": f"{psi_val:.3f}",
                    "KS p-value": f"{ks_p:.3f}",
                    "Status": status,
                }
            )

        st.table(pd.DataFrame(drift_summary).set_index("Feature"))

        # Overlayed distributions
        st.subheader("Feature Distributions: Baseline vs. Last 24 hours")
        for feat in FEATURES:
            fig, ax = plt.subplots()
            sns.histplot(
                log_baseline[feat],
                stat="density",
                element="step",
                label="Baseline",
                ax=ax,
            )
            sns.histplot(
                log_recent[feat],
                stat="density",
                element="step",
                label="Last 24-hrs",
                ax=ax,
            )
            ax.set_title(feat)
            ax.legend()
            st.pyplot(fig)

    # — Prediction —
    with pred_subtab:
        st.subheader("Prediction Confidence Distribution")

        # Confidence score distribution
        st.subheader("Confidence Score Histogram")
        fig, ax = plt.subplots()
        sns.histplot(
            perf_df["hist_probs"],
            bins=10,
            stat="count",
            ax=ax,
        )
        ax.set_xlabel("P(up)")
        ax.set_ylabel("Count")
        st.pyplot(fig)
