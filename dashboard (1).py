import base64
import io
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="Stock Dashboard", layout="wide")

st.title("📈 Stock Price Prediction Dashboard")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    csv_path = st.text_input("OR CSV path (optional)", value="")

    st.markdown("---")
    lookback = st.slider("Lookback (days)", 10, 180, 60)
    test_ratio = st.slider("Test ratio", 0.05, 0.5, 0.2, 0.05)

    st.markdown("---")
    n_estimators = st.slider("n_estimators", 50, 600, 200, step=50)
    learning_rate = st.slider("learning_rate", 0.01, 0.3, 0.05, 0.01)
    max_depth = st.slider("max_depth", 1, 10, 5)

    run = st.button("Run")


def _validate_df(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("CSV must contain columns: 'Date' and 'Close'")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")

    return df


def _plot_to_png_bytes(actual, predicted, rmse, mse):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual Price")
    plt.plot(predicted, label="Predicted Price")
    plt.title(
        "Stock Price Prediction (Gradient Boosting)\n"
        f"RMSE: {rmse:.2f}, MSE: {mse:.2f}"
    )
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def run_training(df: pd.DataFrame):
    df = _validate_df(df)

    # Filter last 5 years
    last_date = df["Date"].max()
    five_years_ago = last_date - pd.DateOffset(years=5)
    df5 = df[df["Date"] >= five_years_ago]
    if len(df5) < (lookback + 10):
        raise ValueError(
            f"Not enough rows after 5-year filter. Rows: {len(df5)}. "
            f"Need at least lookback+10 = {lookback+10}."
        )

    data = df5[["Close"]].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback : i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    split_idx = int(len(X) * (1 - test_ratio))
    if split_idx < 10 or split_idx >= len(X):
        raise ValueError("Invalid split produced. Try changing Lookback/Test ratio.")

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    mse = mean_squared_error(actual, predictions)
    rmse = float(np.sqrt(mse))

    last_60_days = scaled_data[-lookback:]
    last_60_days = np.reshape(last_60_days, (1, -1))
    future_price = model.predict(last_60_days)
    future_price = scaler.inverse_transform(future_price.reshape(-1, 1))

    png_bytes = _plot_to_png_bytes(actual, predictions, rmse, mse)

    return {
        "df_rows": int(len(df5)),
        "mse": float(mse),
        "rmse": rmse,
        "future_price": float(future_price[0][0]),
        "plot_png": png_bytes,
        "actual": actual.tolist(),
        "predicted": predictions.tolist(),
    }


if run:
    try:
        if uploaded is not None:
            df_in = pd.read_csv(uploaded)
        else:
            if not csv_path.strip():
                raise ValueError("Upload CSV or provide a CSV path.")
            df_in = pd.read_csv(csv_path.strip())

        with st.spinner("Training model and generating dashboard..."):
            result = run_training(df_in)

        st.success("Done")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows used (last 5y)", result["df_rows"])
        with c2:
            st.metric("RMSE", f"{result['rmse']:.4f}")
        with c3:
            st.metric("Predicted Next Day Close", f"{result['future_price']:.4f}")

        st.subheader("Prediction vs Actual")
        st.image(result["plot_png"], caption="Actual vs Predicted")

        with st.expander("Raw values (optional)"):
            st.json({
                "mse": result["mse"],
                "rmse": result["rmse"],
                "future_price": result["future_price"],
                "points": len(result["actual"]),
            })

    except Exception as e:
        st.error(str(e))
else:
    st.info("Upload a CSV with columns 'Date' and 'Close', then click Run.")

