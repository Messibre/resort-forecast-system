import os
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from huggingface_hub import HfApi
from prophet import Prophet
from supabase import create_client


def require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {var_name}")
    return value


def get_supabase_client():
    url = require_env("SUPABASE_URL")
    key = require_env("SUPABASE_ANON_KEY")
    return create_client(url, key)


def fetch_daily_occupancy(client) -> pd.DataFrame:
    response = (
        client.table("daily_occupancy")
        .select("date,occupied_rooms")
        .order("date", desc=False)
        .execute()
    )

    rows = response.data or []
    if not rows:
        raise RuntimeError("No records found in daily_occupancy table")

    df = pd.DataFrame(rows)
    if "date" not in df.columns or "occupied_rooms" not in df.columns:
        raise RuntimeError(
            "daily_occupancy must provide 'date' and 'occupied_rooms' columns"
        )

    df = df.rename(columns={"date": "ds", "occupied_rooms": "y"})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).sort_values("ds")

    if df.empty:
        raise RuntimeError("daily_occupancy had no valid rows after cleaning")

    return df[["ds", "y"]]


def fetch_events(client) -> pd.DataFrame:
    response = client.table("events").select("date,name").execute()
    rows = response.data or []

    if not rows:
        return pd.DataFrame(columns=["ds", "holiday", "lower_window", "upper_window"])

    events = pd.DataFrame(rows)
    events["ds"] = pd.to_datetime(events.get("date"), errors="coerce")
    events["holiday"] = (
        events.get("name")
        .fillna("event")
        .astype(str)
        .str.strip()
        .replace("", "event")
        .apply(lambda x: f"event_{x.lower().replace(' ', '_')}")
    )
    events["lower_window"] = 0
    events["upper_window"] = 0

    events = events.dropna(subset=["ds", "holiday"])
    return events[["ds", "holiday", "lower_window", "upper_window"]]


def load_static_holidays() -> pd.DataFrame:
    csv_path = Path(__file__).resolve().parents[1] / "data" / "ethiopian_holidays.csv"
    if not csv_path.exists():
        raise RuntimeError(f"Holidays CSV not found: {csv_path}")

    holidays = pd.read_csv(csv_path)
    required_cols = {"ds", "holiday"}
    if not required_cols.issubset(set(holidays.columns)):
        raise RuntimeError("ethiopian_holidays.csv must contain ds and holiday columns")

    holidays["ds"] = pd.to_datetime(holidays["ds"], errors="coerce")
    holidays["lower_window"] = pd.to_numeric(
        holidays.get("lower_window", 0), errors="coerce"
    ).fillna(0).astype(int)
    holidays["upper_window"] = pd.to_numeric(
        holidays.get("upper_window", 0), errors="coerce"
    ).fillna(0).astype(int)

    holidays = holidays.dropna(subset=["ds", "holiday"])
    return holidays[["ds", "holiday", "lower_window", "upper_window"]]


def build_holidays(static_holidays: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    all_holidays = pd.concat([static_holidays, events], ignore_index=True)
    all_holidays = all_holidays.drop_duplicates(subset=["ds", "holiday"])
    return all_holidays.sort_values("ds")


def train_model(train_df: pd.DataFrame, holidays_df: pd.DataFrame) -> Prophet:
    model = Prophet(
        holidays=holidays_df if not holidays_df.empty else None,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.8,
    )
    model.fit(train_df)
    return model


def upload_model(model: Prophet) -> None:
    hf_token = require_env("HF_TOKEN")
    repo_id = os.getenv("HF_REPO_ID") or os.getenv("HUGGINGFACE_REPO_ID") or "messibre/resort-demand-forecast"
    model_file = os.getenv("HF_MODEL_FILE", "model.joblib")

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
        temp_path = Path(tmp_file.name)

    try:
        joblib.dump(model, temp_path)
        api.upload_file(
            path_or_fileobj=str(temp_path),
            path_in_repo=model_file,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
        )
        print(f"Uploaded model to Hugging Face repo: {repo_id} as {model_file}")
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> int:
    print("Starting model training job...")
    client = get_supabase_client()

    train_df = fetch_daily_occupancy(client)
    print(f"Loaded {len(train_df)} occupancy rows")

    static_holidays = load_static_holidays()
    print(f"Loaded {len(static_holidays)} static holiday rows")

    events = fetch_events(client)
    print(f"Loaded {len(events)} event rows")

    holidays_df = build_holidays(static_holidays, events)
    print(f"Using {len(holidays_df)} total holiday/event rows for Prophet")

    model = train_model(train_df, holidays_df)
    print("Model training complete")

    upload_model(model)
    print("Training workflow finished successfully")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Training job failed: {exc}")
        raise
