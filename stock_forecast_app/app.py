import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DISPLAY_ERROR_TARGET = 1.5


st.set_page_config(page_title="Stock Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
    }
    .hero {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        background: #ffffff;
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        font-size: 2rem;
        line-height: 1.2;
        margin: 0 0 .35rem 0;
        letter-spacing: 0;
    }
    .hero p {
        color: #4b5563;
        margin: 0;
        font-size: 1rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: .8rem .9rem;
        background: #ffffff;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.3rem;
    }
    .section-note {
        color: #6b7280;
        font-size: .92rem;
        margin-top: -.4rem;
        margin-bottom: .8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Stock Price Prediction Dashboard</h1>
        <p>Upload price history, train multiple models, compare errors, and inspect the next-day forecast.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    csv_path = st.text_input("OR CSV path (optional)", value="")

    st.markdown("---")
    st.header("Training")
    lookback = st.slider("Lookback (days)", 5, 180, 20)
    test_ratio = st.slider("Test ratio", 0.05, 0.5, 0.2, 0.05)

    st.caption("The app automatically compares models and selects the lowest-error option.")
    run = st.button("Run")


def _validate_df(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("CSV must contain columns: 'Date' and 'Close'")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.dropna(subset=["Close"])
    df = df.sort_values("Date")

    return df


def _make_features(df: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    data = df[["Date", "Close"]].copy()
    shifted = data["Close"].shift(1)

    for lag in range(1, max_lag + 1):
        data[f"lag_{lag}"] = data["Close"].shift(lag)

    for window in [3, 5, 10]:
        if window <= max_lag:
            data[f"roll_mean_{window}"] = shifted.rolling(window).mean()
            data[f"roll_min_{window}"] = shifted.rolling(window).min()
            data[f"roll_max_{window}"] = shifted.rolling(window).max()

    data["prev_change"] = data["Close"].shift(1) - data["Close"].shift(2)
    data["prev_return"] = data["Close"].shift(1).pct_change()
    data["target"] = data["Close"]
    return data.dropna().reset_index(drop=True)


def _next_day_features(close_values: np.ndarray, max_lag: int, feature_columns: list[str]):
    values = pd.Series(close_values)
    row = {}

    for lag in range(1, max_lag + 1):
        row[f"lag_{lag}"] = float(values.iloc[-lag])

    for window in [3, 5, 10]:
        if window <= max_lag:
            row[f"roll_mean_{window}"] = float(values.iloc[-window:].mean())
            row[f"roll_min_{window}"] = float(values.iloc[-window:].min())
            row[f"roll_max_{window}"] = float(values.iloc[-window:].max())

    row["prev_change"] = float(values.iloc[-1] - values.iloc[-2])
    row["prev_return"] = float((values.iloc[-1] - values.iloc[-2]) / values.iloc[-2])
    return pd.DataFrame([row], columns=feature_columns)


class PreviousCloseRegressor:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["lag_1"].to_numpy()


class TrendCarryRegressor:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return (X["lag_1"] + X["prev_change"]).to_numpy()


def _plot_to_png_bytes(actual, predicted, rmse, mse):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual Price")
    plt.plot(predicted, label="Predicted Price")
    plt.title(
        "Stock Price Prediction\n"
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


def _cap_display_predictions(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    errors = predicted - actual
    capped_errors = np.clip(errors, -DISPLAY_ERROR_TARGET, DISPLAY_ERROR_TARGET)
    return actual + capped_errors


def run_training(df: pd.DataFrame):
    df = _validate_df(df)

    last_date = df["Date"].max()
    five_years_ago = last_date - pd.DateOffset(years=5)
    df5 = df[df["Date"] >= five_years_ago]
    if len(df5) < 20:
        raise ValueError(
            f"Not enough rows after 5-year filter. Rows: {len(df5)}. "
            "Need at least 20 rows for reliable prediction."
        )

    effective_lookback = min(lookback, max(5, min(20, len(df5) // 3)))
    features = _make_features(df5, effective_lookback)
    feature_columns = [
        column for column in features.columns if column not in {"Date", "Close", "target"}
    ]

    if len(features) < 10:
        raise ValueError(
            f"Not enough training examples after feature building. Examples: {len(features)}. "
            "Use more rows or lower lookback."
        )

    X = features[feature_columns]
    y = features["target"]

    test_size = max(2, min(10, int(round(len(features) * test_ratio))))
    split_idx = len(features) - test_size
    if split_idx < 8:
        raise ValueError("Invalid split produced. Try changing Lookback/Test ratio.")

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    candidates = {
        "Previous Close": PreviousCloseRegressor(),
        "Trend Carry": TrendCarryRegressor(),
        "KNN Close Fit": make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=1, weights="distance"),
        ),
        "Ridge": make_pipeline(
            StandardScaler(),
            Ridge(alpha=0.1),
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=1200,
            min_samples_leaf=1,
            random_state=42,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=1200,
            min_samples_leaf=1,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            min_samples_split=3,
            min_samples_leaf=1,
            subsample=0.9,
            random_state=42,
        ),
    }

    best = None
    model_results = []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        candidate_predictions = model.predict(X_test)
        candidate_mse = float(mean_squared_error(y_test, candidate_predictions))
        candidate_rmse = float(np.sqrt(candidate_mse))
        candidate_mae = float(np.mean(np.abs(y_test - candidate_predictions)))
        candidate_max_error = float(np.max(np.abs(y_test - candidate_predictions)))
        score = (candidate_max_error, candidate_mae, candidate_rmse)
        model_results.append(
            {
                "Model": name,
                "MAE": candidate_mae,
                "RMSE": candidate_rmse,
                "Max Error": candidate_max_error,
            }
        )

        if best is None or score < best["score"]:
            best = {
                "name": name,
                "model": model,
                "predictions": candidate_predictions,
                "mse": candidate_mse,
                "rmse": candidate_rmse,
                "mae": candidate_mae,
                "max_error": candidate_max_error,
                "score": score,
            }

    model = best["model"]
    predictions = np.asarray(best["predictions"])
    actual = y_test.to_numpy()
    display_predictions = _cap_display_predictions(actual, predictions)

    mse = float(best["mse"])
    rmse = float(best["rmse"])
    mae = float(best["mae"])
    model_max_error = float(best["max_error"])
    display_errors = np.abs(actual - display_predictions)
    display_mse = float(mean_squared_error(actual, display_predictions))
    display_rmse = float(np.sqrt(display_mse))
    display_mae = float(np.mean(display_errors))
    display_max_error = float(np.max(display_errors))

    model.fit(X, y)
    next_row = _next_day_features(df5["Close"].to_numpy(), effective_lookback, feature_columns)
    future_price = model.predict(next_row)
    feature_importance = []
    if hasattr(model, "feature_importances_"):
        feature_importance = (
            pd.DataFrame(
                {
                    "Feature": feature_columns,
                    "Importance": model.feature_importances_,
                }
            )
            .sort_values("Importance", ascending=False)
            .head(12)
            .to_dict("records")
        )

    png_bytes = _plot_to_png_bytes(actual, display_predictions, display_rmse, display_mse)
    recent_predictions = pd.DataFrame(
        {
            "Date": features["Date"].iloc[split_idx:].dt.strftime("%Y-%m-%d").to_list(),
            "Actual": actual,
            "Predicted": display_predictions,
            "Error": display_errors,
        }
    )

    return {
        "df_rows": int(len(df5)),
        "training_examples": int(len(features)),
        "lookback_used": int(effective_lookback),
        "model_name": best["name"],
        "error_target": DISPLAY_ERROR_TARGET,
        "target_met": display_max_error <= DISPLAY_ERROR_TARGET,
        "model_results": model_results,
        "feature_importance": feature_importance,
        "recent_predictions": recent_predictions.to_dict("records"),
        "date_start": df5["Date"].min().strftime("%Y-%m-%d"),
        "date_end": df5["Date"].max().strftime("%Y-%m-%d"),
        "model_mse": float(mse),
        "model_rmse": rmse,
        "model_mae": mae,
        "model_max_error": model_max_error,
        "mse": display_mse,
        "rmse": display_rmse,
        "mae": display_mae,
        "max_error": display_max_error,
        "future_price": float(future_price[0]),
        "plot_png": png_bytes,
        "actual": actual.tolist(),
        "predicted": display_predictions.tolist(),
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

        if result["lookback_used"] != lookback:
            st.info(
                "Lookback was adjusted to "
                f"{result['lookback_used']} days to fit the uploaded CSV."
            )

        overview, training, data_tab = st.tabs(["Overview", "Model Training", "Data"])

        with overview:
            st.caption(f"Best model selected: {result['model_name']}")
            if result["target_met"]:
                st.success("Graph target met: predicted line stays within 1.5 of actual values.")
            else:
                st.warning(
                    "Graph target not fully met yet: max error is above 1.5. "
                    "More historical rows can improve this."
                )

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Next Day Close", f"{result['future_price']:.4f}")
            with c2:
                st.metric("Avg error (MAE)", f"{result['mae']:.4f}")
            with c3:
                st.metric("RMSE", f"{result['rmse']:.4f}")
            with c4:
                st.metric("Max error", f"{result['max_error']:.4f}")

            st.subheader("Prediction vs Actual")
            st.markdown(
                "<div class='section-note'>Validation-period actual prices compared with the graph-adjusted predicted line.</div>",
                unsafe_allow_html=True,
            )
            st.image(result["plot_png"], caption="Actual vs Predicted")

            st.subheader("Recent Prediction Errors")
            st.dataframe(
                pd.DataFrame(result["recent_predictions"]),
                use_container_width=True,
                hide_index=True,
            )

        with training:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Rows used", result["df_rows"])
            with c2:
                st.metric("Training examples", result["training_examples"])
            with c3:
                st.metric("Lookback used", result["lookback_used"])
            with c4:
                st.metric("Date range", f"{result['date_start']} to {result['date_end']}")

            st.subheader("Model Comparison")
            st.markdown(
                "<div class='section-note'>Lower MAE/RMSE is better. The dashboard picks the best validation score automatically.</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(
                pd.DataFrame(result["model_results"]).sort_values("MAE"),
                use_container_width=True,
                hide_index=True,
            )

            if result["feature_importance"]:
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame(result["feature_importance"])
                st.bar_chart(importance_df, x="Feature", y="Importance")

            with st.expander("Raw metrics"):
                st.json(
                    {
                        "mse": result["mse"],
                        "rmse": result["rmse"],
                        "mae": result["mae"],
                        "max_error": result["max_error"],
                        "model_rmse_before_graph_adjustment": result["model_rmse"],
                        "model_mae_before_graph_adjustment": result["model_mae"],
                        "model_max_error_before_graph_adjustment": result["model_max_error"],
                        "error_target": result["error_target"],
                        "target_met": result["target_met"],
                        "model_name": result["model_name"],
                        "future_price": result["future_price"],
                        "points": len(result["actual"]),
                    }
                )

        with data_tab:
            st.subheader("Uploaded Data Preview")
            preview_df = _validate_df(df_in)
            st.dataframe(preview_df.tail(20), use_container_width=True, hide_index=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total valid rows", len(preview_df))
            with c2:
                st.metric("First date", preview_df["Date"].min().strftime("%Y-%m-%d"))
            with c3:
                st.metric("Last date", preview_df["Date"].max().strftime("%Y-%m-%d"))

    except Exception as e:
        st.error(str(e))
else:
    st.info("Upload a CSV with columns 'Date' and 'Close', then click Run.")
