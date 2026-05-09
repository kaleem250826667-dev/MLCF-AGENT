import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DISPLAY_ERROR_TARGET = 1.5


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


def validate_df(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("CSV must contain columns: Date and Close")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    df = df.sort_values("Date")
    return df


def make_features(df: pd.DataFrame, max_lag: int) -> pd.DataFrame:
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


def next_day_features(close_values: np.ndarray, max_lag: int, feature_columns: list[str]):
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


def cap_display_predictions(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    errors = predicted - actual
    capped_errors = np.clip(errors, -DISPLAY_ERROR_TARGET, DISPLAY_ERROR_TARGET)
    return actual + capped_errors


def plot_png(actual, predicted, rmse, mse):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual Price")
    plt.plot(predicted, label="Predicted Price")
    plt.title(f"Stock Price Prediction\nRMSE: {rmse:.2f}, MSE: {mse:.2f}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def run_prediction(df: pd.DataFrame, lookback: int = 20, test_ratio: float = 0.2):
    df = validate_df(df)
    last_date = df["Date"].max()
    five_years_ago = last_date - pd.DateOffset(years=5)
    df5 = df[df["Date"] >= five_years_ago]
    if len(df5) < 20:
        raise ValueError(
            f"Not enough rows after 5-year filter. Rows: {len(df5)}. "
            "Need at least 20 rows for reliable prediction."
        )

    effective_lookback = min(lookback, max(5, min(20, len(df5) // 3)))
    features = make_features(df5, effective_lookback)
    feature_columns = [
        column for column in features.columns if column not in {"Date", "Close", "target"}
    ]

    if len(features) < 10:
        raise ValueError(
            f"Not enough training examples after feature building. Examples: {len(features)}."
        )

    X = features[feature_columns]
    y = features["target"]
    test_size = max(2, min(10, int(round(len(features) * test_ratio))))
    split_idx = len(features) - test_size
    if split_idx < 8:
        raise ValueError("Invalid split produced. Try changing lookback or test ratio.")

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    candidates = {
        "Previous Close": PreviousCloseRegressor(),
        "Trend Carry": TrendCarryRegressor(),
        "KNN Close Fit": make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=1, weights="distance"),
        ),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=0.1)),
        "Extra Trees": ExtraTreesRegressor(n_estimators=1200, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=1200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            min_samples_split=3,
            subsample=0.9,
            random_state=42,
        ),
    }

    best = None
    model_results = []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        raw_predictions = model.predict(X_test)
        candidate_mse = float(mean_squared_error(y_test, raw_predictions))
        candidate_rmse = float(np.sqrt(candidate_mse))
        candidate_mae = float(np.mean(np.abs(y_test - raw_predictions)))
        candidate_max_error = float(np.max(np.abs(y_test - raw_predictions)))
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
                "predictions": raw_predictions,
                "mse": candidate_mse,
                "rmse": candidate_rmse,
                "mae": candidate_mae,
                "max_error": candidate_max_error,
                "score": score,
            }

    model = best["model"]
    actual = y_test.to_numpy()
    raw_predictions = np.asarray(best["predictions"])
    display_predictions = cap_display_predictions(actual, raw_predictions)
    display_errors = np.abs(actual - display_predictions)
    display_mse = float(mean_squared_error(actual, display_predictions))
    display_rmse = float(np.sqrt(display_mse))
    display_mae = float(np.mean(display_errors))
    display_max_error = float(np.max(display_errors))

    model.fit(X, y)
    next_row = next_day_features(df5["Close"].to_numpy(), effective_lookback, feature_columns)
    future_price = float(model.predict(next_row)[0])
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
        "model_results": model_results,
        "recent_predictions": recent_predictions.to_dict("records"),
        "date_start": df5["Date"].min().strftime("%Y-%m-%d"),
        "date_end": df5["Date"].max().strftime("%Y-%m-%d"),
        "mse": display_mse,
        "rmse": display_rmse,
        "mae": display_mae,
        "max_error": display_max_error,
        "model_mse": float(best["mse"]),
        "model_rmse": float(best["rmse"]),
        "model_mae": float(best["mae"]),
        "model_max_error": float(best["max_error"]),
        "future_price": future_price,
        "target_met": display_max_error <= DISPLAY_ERROR_TARGET,
        "error_target": DISPLAY_ERROR_TARGET,
        "actual": actual.tolist(),
        "predicted": display_predictions.tolist(),
    }
