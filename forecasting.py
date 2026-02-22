#!/usr/bin/env python3
"""
forecasting.py - fixed version

Usage:
  python forecasting.py --generate-sample --output outputs/forecast --periods 6
  python forecasting.py --input data/financial_data.csv --output outputs/forecast --periods 6

Dependencies:
  pip install pandas numpy prophet pmdarima scikit-learn joblib matplotlib
"""

from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np
from prophet import Prophet
import pmdarima as pm
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def generate_sample_data(start_date: str = "2018-01-01", periods: int = 72, freq: str = "MS"):
    rng = pd.date_range(start=start_date, periods=periods, freq=freq)
    t = np.arange(periods)
    revenue_trend = 10000 + 150 * t
    revenue_season = 2000 * np.sin(2 * np.pi * (t % 12) / 12)
    revenue_noise = np.random.normal(scale=2000, size=periods)
    revenue = revenue_trend + revenue_season + revenue_noise
    revenue = np.where(revenue < 100, 100, revenue)
    expenses = revenue * (0.6 + 0.05 * np.sin(2 * np.pi * (t % 12) / 12)) + np.random.normal(scale=1000, size=periods)
    expenses = np.where(expenses < 50, 50, expenses)
    df = pd.DataFrame({
        "date": rng,
        "revenue": revenue.round(2),
        "expenses": expenses.round(2),
        "region": ["Global"] * periods,
        "department": ["General"] * periods
    })
    return df


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    if "date" not in df.columns:
        raise ValueError("Input CSV must contain a 'date' column.")
    for col in ["revenue", "expenses"]:
        if col not in df.columns:
            logging.warning(f"Column '{col}' not found in input; creating with zeros.")
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["revenue"] = df["revenue"].fillna(method="ffill").fillna(0)
    df["expenses"] = df["expenses"].fillna(method="ffill").fillna(0)
    df["month_start"] = df["date"].dt.to_period("M").dt.to_timestamp()
    agg = df.groupby("month_start").agg({"revenue": "sum", "expenses": "sum"}).reset_index().rename(columns={"month_start": "date"})
    agg["profit"] = agg["revenue"] - agg["expenses"]
    agg = agg.sort_values("date").reset_index(drop=True)
    logging.info(f"Loaded and aggregated data: {len(agg)} monthly rows.")
    return agg


def prophet_forecast(df: pd.DataFrame, value_col: str = "revenue", periods: int = 6):
    # prepare
    df_prophet = df[["date", value_col]].rename(columns={"date": "ds", value_col: "y"})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=periods, freq="MS")
    forecast = m.predict(future)
    forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={
        "ds": "date", "yhat": "predicted_revenue", "yhat_lower": "pred_lower", "yhat_upper": "pred_upper"
    })
    # create actuals frame aligned for merge
    df_actual = df_prophet.rename(columns={"ds": "date", "y": "actual_revenue"})
    combined = pd.merge(df_actual, forecast, on="date", how="outer")
    combined["model"] = "Prophet"
    # ensure date sorted
    combined = combined.sort_values("date").reset_index(drop=True)
    return m, combined


def arima_forecast(df: pd.DataFrame, value_col: str = "revenue", periods: int = 6):
    ts = df.set_index("date")[value_col].asfreq("MS")
    if ts.isnull().any():
        ts = ts.fillna(method="ffill").fillna(0)
    model = pm.auto_arima(ts, seasonal=True, m=12, stepwise=True, suppress_warnings=True, error_action="ignore")
    fc, conf_int = model.predict(n_periods=periods, return_conf_int=True)
    start = ts.index[-1] + pd.offsets.MonthBegin(1)
    future_index = pd.date_range(start=start, periods=periods, freq="MS")
    forecast_df = pd.DataFrame({
        "date": future_index,
        "predicted_revenue": fc,
        "pred_lower": conf_int[:, 0],
        "pred_upper": conf_int[:, 1]
    })
    # create combined with actuals (historical actual_revenue only)
    hist_actual = df[["date", "revenue"]].rename(columns={"revenue": "actual_revenue"})
    combined = pd.concat([
        hist_actual.assign(predicted_revenue=pd.NA, pred_lower=pd.NA, pred_upper=pd.NA, model=pd.NA),
        forecast_df.assign(model="ARIMA")
    ], ignore_index=True, sort=False)
    combined = combined.sort_values("date").reset_index(drop=True)
    return model, combined


def evaluate(actual: pd.DataFrame, predicted: pd.DataFrame, actual_col: str = "actual_revenue", pred_col: str = "predicted_revenue") -> dict:
    # safety checks
    if "date" not in actual.columns or "date" not in predicted.columns:
        logging.warning("Missing 'date' in actual or predicted for evaluation.")
        return {"mape": None, "rmse": None}
    if actual_col not in actual.columns:
        logging.warning(f"Actual column '{actual_col}' missing.")
        return {"mape": None, "rmse": None}
    if pred_col not in predicted.columns:
        logging.warning(f"Predicted column '{pred_col}' missing.")
        return {"mape": None, "rmse": None}
    merged = actual.merge(predicted[["date", pred_col]], on="date", how="inner")
    if merged.empty:
        logging.warning("No overlapping dates for evaluation.")
        return {"mape": None, "rmse": None}
    try:
        mape = mean_absolute_percentage_error(merged[actual_col], merged[pred_col])
        mse = mean_squared_error(merged[actual_col], merged[pred_col])
        rmse = float(mse ** 0.5)
        return {"mape": float(mape), "rmse": rmse}
    except Exception as e:
        logging.error(f"Evaluation failure: {e}")
        return {"mape": None, "rmse": None}


def save_outputs(output_prefix: str, hist: pd.DataFrame, model_name: str, model_obj, forecast_df: pd.DataFrame):
    from pathlib import Path
    import joblib
    import logging
    import pandas as pd

    out_dir = Path(output_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_file = f"{output_prefix}_{model_name}_model.pkl"
    joblib.dump(model_obj, model_file)
    logging.info(f"✅ Saved model to {model_file}")

    fc = forecast_df.copy()

    # --- 🔧 Handle ARIMA-style outputs ---
    # Sometimes ARIMA forecast returns columns like 'forecast', 'lower', 'upper' instead of yhat
    rename_map = {}
    if "yhat" in fc.columns:
        rename_map["yhat"] = "predicted_revenue"
    elif "forecast" in fc.columns:
        rename_map["forecast"] = "predicted_revenue"

    if "yhat_lower" in fc.columns:
        rename_map["yhat_lower"] = "pred_lower"
    elif "lower" in fc.columns:
        rename_map["lower"] = "pred_lower"

    if "yhat_upper" in fc.columns:
        rename_map["yhat_upper"] = "pred_upper"
    elif "upper" in fc.columns:
        rename_map["upper"] = "pred_upper"

    if rename_map:
        fc = fc.rename(columns=rename_map)

    # --- 🗓️ Save Forecast File ---
    forecast_file = f"{output_prefix}_{model_name}_forecast.csv"
    fc.to_csv(forecast_file, index=False, date_format="%Y-%m-%d")
    logging.info(f"✅ Saved forecast to {forecast_file}")

    # --- 📊 Prepare Combined File for Power BI ---
    hist_cols = hist[["date", "revenue"]].rename(columns={"revenue": "actual_revenue"})

    # Identify future rows (dates beyond last actual)
    future_only = fc[~fc["date"].isin(hist_cols["date"])]

    combined = pd.concat([
        hist_cols.assign(predicted_revenue=pd.NA, pred_lower=pd.NA, pred_upper=pd.NA, model="Actual"),
        future_only.assign(actual_revenue=pd.NA, model=model_name)
    ], ignore_index=True, sort=False)

    combined_file = f"{output_prefix}_{model_name}_combined.csv"
    combined.to_csv(combined_file, index=False, date_format="%Y-%m-%d")
    logging.info(f"✅ Saved combined CSV to {combined_file}")

def plot_forecast(hist: pd.DataFrame, forecast_df: pd.DataFrame, value_col="revenue", model_name="model", outpath: str | None = None):
    # ensure forecast_df has predicted_revenue column or yhat fallback
    fc = forecast_df.copy()
    if "predicted_revenue" not in fc.columns and "yhat" in fc.columns:
        fc = fc.rename(columns={"yhat": "predicted_revenue", "yhat_lower": "pred_lower", "yhat_upper": "pred_upper"})
    plt.figure(figsize=(10, 5))
    plt.plot(hist["date"], hist[value_col], label="Actual")
    # plot forecast predicted_revenue where present
    if "predicted_revenue" in fc.columns:
        plt.plot(fc["date"], fc["predicted_revenue"], label=f"Forecast ({model_name})", linestyle="--")
        if "pred_lower" in fc.columns and "pred_upper" in fc.columns:
            plt.fill_between(fc["date"], fc["pred_lower"].astype(float), fc["pred_upper"].astype(float), alpha=0.2)
    plt.title(f"Actual vs Forecast - {model_name}")
    plt.xlabel("Date")
    plt.ylabel(value_col.capitalize())
    plt.legend()
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
        logging.info(f"Saved forecast plot to {outpath}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Financial forecasting script (Prophet + ARIMA).")
    parser.add_argument("--input", help="CSV input path with at least 'date' and 'revenue' columns.")
    parser.add_argument("--output", required=True, help="Output file prefix (e.g., outputs/forecast). Directory will be created.")
    parser.add_argument("--periods", type=int, default=6, help="Forecast horizon in months.")
    parser.add_argument("--generate-sample", action="store_true", help="Generate synthetic sample data instead of reading input.")
    parser.add_argument("--sample-start", default="2018-01-01", help="Start date for sample data (if --generate-sample).")
    parser.add_argument("--sample-months", type=int, default=60, help="Number of months for sample data.")
    args = parser.parse_args()

    output_prefix = args.output.rstrip("/")

    if args.generate_sample:
        logging.info("Generating synthetic sample data...")
        df_raw = generate_sample_data(start_date=args.sample_start, periods=args.sample_months, freq="MS")
    else:
        if not args.input:
            raise ValueError("Either --input must be provided or use --generate-sample.")
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")
        df_raw = pd.read_csv(args.input, parse_dates=["date"])

    raw_out = f"{output_prefix}_raw.csv"
    Path(raw_out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(df_raw).to_csv(raw_out, index=False, date_format="%Y-%m-%d")
    logging.info(f"Wrote raw data to {raw_out}")

    hist = load_and_clean(raw_out)

    logging.info("Fitting Prophet for revenue...")
    prop_model, prop_combined = prophet_forecast(hist, value_col="revenue", periods=args.periods)

    logging.info("Fitting ARIMA for revenue...")
    arima_model, arima_combined = arima_forecast(hist, value_col="revenue", periods=args.periods)

    # Evaluation: align columns for evaluation
    logging.info("Evaluating models...")
    hist_eval = hist[["date", "revenue"]].rename(columns={"revenue": "actual_revenue"})
    # Prophet eval (prop_combined has predicted_revenue)
    prop_pred_eval = prop_combined[["date", "predicted_revenue"]].dropna()
    eval_prop = evaluate(hist_eval, prop_pred_eval, actual_col="actual_revenue", pred_col="predicted_revenue")
    # ARIMA eval (arima_combined contains historical then forecast rows; select forecast rows only)
    arima_pred_eval = arima_combined[arima_combined["model"] == "ARIMA"][["date", "predicted_revenue"]]
    eval_arima = evaluate(hist_eval, arima_pred_eval, actual_col="actual_revenue", pred_col="predicted_revenue")

    logging.info(f"Prophet eval — MAPE={eval_prop['mape']} RMSE={eval_prop['rmse']}")
    logging.info(f"ARIMA eval  — MAPE={eval_arima['mape']} RMSE={eval_arima['rmse']}")

    # Save outputs (models, forecasts, combined csvs)
    save_outputs(output_prefix, hist, "prophet", prop_model, prop_combined)
    save_outputs(output_prefix, hist, "arima", arima_model, arima_combined)

    # Plots
    plot_forecast(hist, prop_combined, value_col="revenue", model_name="Prophet", outpath=f"{output_prefix}_prophet_plot.png")
    plot_forecast(hist, arima_combined, value_col="revenue", model_name="ARIMA", outpath=f"{output_prefix}_arima_plot.png")

    logging.info("Done. Check outputs/ for CSVs, models and plots.")


if __name__ == "__main__":
    main()
