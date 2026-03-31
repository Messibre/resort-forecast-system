import os
import sys
import tempfile
from pathlib import Path
import re

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
    table_name = os.getenv("SUPABASE_OCCUPANCY_TABLE", "daily_occupancy")
    date_col_override = os.getenv("SUPABASE_OCCUPANCY_DATE_COLUMN")
    target_col_override = os.getenv("SUPABASE_OCCUPANCY_TARGET_COLUMN")

    response = (
        client.table(table_name)
        .select("*")
        .execute()
    )

    rows = response.data or []
    if not rows:
        raise RuntimeError(f"No records found in {table_name} table")

    df = pd.DataFrame(rows)

    date_candidates = [
        date_col_override,
        "date",
        "stay_date",
        "booking_date",
        "ds",
        "created_at",
    ]
    target_candidates = [
        target_col_override,
        "rooms_sold",
        "occupancy",
        "demand",
        "y",
    ]

    date_col = next((col for col in date_candidates if col and col in df.columns), None)
    target_col = next(
        (col for col in target_candidates if col and col in df.columns),
        None,
    )

    if not date_col or not target_col:
        raise RuntimeError(
            f"Could not infer date/target columns in {table_name}. "
            f"Available columns: {sorted(df.columns.tolist())}. "
            "Set SUPABASE_OCCUPANCY_DATE_COLUMN and SUPABASE_OCCUPANCY_TARGET_COLUMN."
        )

    df = df.rename(columns={date_col: "ds", target_col: "y"})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).sort_values("ds")

    if df.empty:
        raise RuntimeError(f"{table_name} had no valid rows after cleaning")

    return df[["ds", "y"]]


def fetch_events(client) -> pd.DataFrame:
    table_name = os.getenv("SUPABASE_EVENTS_TABLE", "events")
    date_col_override = os.getenv("SUPABASE_EVENTS_DATE_COLUMN")
    name_col_override = os.getenv("SUPABASE_EVENTS_NAME_COLUMN")
    type_col_override = os.getenv("SUPABASE_EVENTS_TYPE_COLUMN")
    impact_col_override = os.getenv("SUPABASE_EVENTS_IMPACT_COLUMN")

    try:
        response = client.table(table_name).select("*").execute()
    except Exception as exc:
        print(f"Warning: failed to fetch events from {table_name}: {exc}")
        return pd.DataFrame(columns=["ds", "holiday", "lower_window", "upper_window"])

    rows = response.data or []
    if not rows:
        return pd.DataFrame(columns=["ds", "holiday", "lower_window", "upper_window"])

    events = pd.DataFrame(rows)

    date_candidates = [date_col_override, "date", "ds", "event_date", "created_at"]
    name_candidates = [name_col_override, "event_name", "name", "title"]
    type_candidates = [type_col_override, "event_type", "type", "category"]
    impact_candidates = [impact_col_override, "expected_impact", "impact", "priority"]

    date_col = next((col for col in date_candidates if col and col in events.columns), None)
    name_col = next((col for col in name_candidates if col and col in events.columns), None)
    type_col = next((col for col in type_candidates if col and col in events.columns), None)
    impact_col = next((col for col in impact_candidates if col and col in events.columns), None)

    if not date_col or not name_col:
        print(
            f"Warning: could not infer event date/name columns in {table_name}. "
            f"Available columns: {sorted(events.columns.tolist())}. Skipping events."
        )
        return pd.DataFrame(columns=["ds", "holiday", "lower_window", "upper_window"])

    def slug(value: object) -> str:
        text = str(value).strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        return text.strip("_") or "event"

    events["ds"] = pd.to_datetime(events.get(date_col), errors="coerce")
    event_name_series = events.get(name_col).fillna("event")
    if type_col:
        event_type_series = events.get(type_col).fillna("general")
        events["holiday"] = [
            f"event_{slug(t)}_{slug(n)}" for t, n in zip(event_type_series, event_name_series)
        ]
    else:
        events["holiday"] = [f"event_{slug(n)}" for n in event_name_series]

    # Wider windows for bigger expected impact to let Prophet learn spillover effects.
    if impact_col:
        impact_map = {
            "low": 0,
            "medium": 1,
            "high": 2,
        }
        impact_labels = events.get(impact_col).fillna("low").astype(str).str.strip().str.lower()
        window_sizes = impact_labels.map(impact_map).fillna(0).astype(int)
        events["lower_window"] = -window_sizes
        events["upper_window"] = window_sizes
    else:
        events["lower_window"] = 0
        events["upper_window"] = 0

    events = events.dropna(subset=["ds", "holiday"])
    return events[["ds", "holiday", "lower_window", "upper_window"]]


def fetch_feedback_actuals(client) -> pd.DataFrame:
    table_name = os.getenv("SUPABASE_FEEDBACK_TABLE", "actual_vs_predicted")
    date_col_override = os.getenv("SUPABASE_FEEDBACK_DATE_COLUMN")
    actual_col_override = os.getenv("SUPABASE_FEEDBACK_ACTUAL_COLUMN")

    try:
        response = client.table(table_name).select("*").execute()
    except Exception as exc:
        print(f"Warning: failed to fetch feedback data from {table_name}: {exc}")
        return pd.DataFrame(columns=["ds", "y"])

    rows = response.data or []
    if not rows:
        return pd.DataFrame(columns=["ds", "y"])

    feedback = pd.DataFrame(rows)
    date_candidates = [date_col_override, "date", "ds", "created_at"]
    actual_candidates = [
        actual_col_override,
        "actual_rooms_sold",
        "actual",
        "rooms_sold",
        "y",
    ]

    date_col = next((col for col in date_candidates if col and col in feedback.columns), None)
    actual_col = next((col for col in actual_candidates if col and col in feedback.columns), None)

    if not date_col or not actual_col:
        print(
            f"Warning: could not infer feedback date/actual columns in {table_name}. "
            f"Available columns: {sorted(feedback.columns.tolist())}. Skipping feedback data."
        )
        return pd.DataFrame(columns=["ds", "y"])

    feedback = feedback.rename(columns={date_col: "ds", actual_col: "y"})
    feedback["ds"] = pd.to_datetime(feedback["ds"], errors="coerce")
    feedback["y"] = pd.to_numeric(feedback["y"], errors="coerce")
    feedback = feedback.dropna(subset=["ds", "y"])
    feedback = feedback.sort_values("ds")
    return feedback[["ds", "y"]]


def apply_feedback_adjustments(train_df: pd.DataFrame, feedback_df: pd.DataFrame) -> pd.DataFrame:
    if feedback_df.empty:
        return train_df

    base = train_df.copy()
    base["source_rank"] = 0

    feedback = feedback_df.copy()
    feedback["source_rank"] = 1

    combined = pd.concat([base, feedback], ignore_index=True)
    combined["ds"] = pd.to_datetime(combined["ds"], errors="coerce")
    combined["y"] = pd.to_numeric(combined["y"], errors="coerce")
    combined = combined.dropna(subset=["ds", "y"])
    combined = combined.sort_values(["ds", "source_rank"])
    combined = combined.drop_duplicates(subset=["ds"], keep="last")

    return combined[["ds", "y"]].sort_values("ds")


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

    feedback_df = fetch_feedback_actuals(client)
    print(f"Loaded {len(feedback_df)} feedback rows")

    train_df = apply_feedback_adjustments(train_df, feedback_df)
    print(f"Training rows after feedback adjustment: {len(train_df)}")

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
