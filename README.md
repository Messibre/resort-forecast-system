# Resort Forecast System

## Overview

This repository contains a demand forecasting platform for resort operations.

Core components:

- Prediction API in prediction-api using FastAPI
- Automated model training in scripts/train.py
- Holiday and event-aware modeling for Prophet
- Model artifact storage on Hugging Face
- Data and operational tables in Supabase
- Daily retraining via GitHub Actions

High-level flow:

1. Training job reads occupancy history, events, holidays, and feedback data.
2. Prophet model is trained and uploaded to Hugging Face.
3. API loads the latest model on startup and can reload on demand.
4. Managers can submit overrides and feedback to improve future retraining.

## How To Add Events

Events are stored in the Supabase events table and are used during training and forecast adjustment.

Recommended columns:

- date
- event_name
- event_type
- expected_impact
- source

Example SQL:

```sql
insert into events (date, event_name, event_type, expected_impact, source)
values
  ('2026-04-10', 'City Marathon', 'sports', 'high', 'admin'),
  ('2026-04-14', 'Tech Conference', 'conference', 'medium', 'admin');
```

Notes:

- date should be in YYYY-MM-DD format.
- expected_impact is typically low, medium, or high.

## How To Adjust Staffing Rules

Staffing recommendations are read from the Supabase staffing_rules table.

Required columns:

- department
- guest_ratio
- min_staff
- max_staff
- hourly_rate

Default departments in this project:

- housekeeping
- front_desk
- f_and_b
- maintenance

Example SQL update:

```sql
update staffing_rules
set guest_ratio = 12,
    min_staff = 4,
    hourly_rate = 17
where department = 'housekeeping';
```

Example SQL insert:

```sql
insert into staffing_rules (department, guest_ratio, min_staff, max_staff, hourly_rate)
values ('security', 60, 1, null, 15.5);
```

## How To Trigger Manual Retraining

Retraining is defined in GitHub Actions workflow .github/workflows/train_model.yml.

Manual trigger steps:

1. Open the GitHub repository.
2. Go to Actions.
3. Select Daily Model Training.
4. Click Run workflow.

Required GitHub secrets:

- SUPABASE_URL
- SUPABASE_ANON_KEY
- HF_TOKEN

Recommended GitHub variables:

- HF_REPO_ID
- HF_MODEL_FILE
- SUPABASE_OCCUPANCY_TABLE
- SUPABASE_OCCUPANCY_DATE_COLUMN
- SUPABASE_OCCUPANCY_TARGET_COLUMN
- SUPABASE_EVENTS_TABLE
- SUPABASE_EVENTS_DATE_COLUMN
- SUPABASE_EVENTS_NAME_COLUMN
- SUPABASE_EVENTS_TYPE_COLUMN
- SUPABASE_EVENTS_IMPACT_COLUMN
- SUPABASE_FEEDBACK_TABLE
- SUPABASE_FEEDBACK_DATE_COLUMN
- SUPABASE_FEEDBACK_ACTUAL_COLUMN

## Where To Find API Documentation

When the API is running, FastAPI docs are available at:

- /docs (Swagger UI)
- /redoc (ReDoc)

If deployed on Render, use your base URL, for example:

- https://your-service.onrender.com/docs
- https://your-service.onrender.com/redoc

## API Endpoints

### GET /

Returns a welcome message.

Response example:

```json
{
  "message": "Welcome to the prediction API"
}
```

### GET /health

Returns health status when model is loaded.

Response example:

```json
{
  "status": "ok"
}
```

### POST /reload

Reloads latest model from Hugging Face into memory.

Headers:

- X-API-Token: must match RELOAD_TOKEN

Response example:

```json
{
  "status": "reloaded"
}
```

### POST /forecast

Generates future forecasts.

Request body:

```json
{
  "horizon_days": 90,
  "include_staffing": true,
  "total_rooms": 60
}
```

Response includes:

- date-level predictions
- confidence bounds
- demand class
- occupancy percentage
- event markers
- override flags
- staffing recommendation and labor cost when enabled

### GET /forecast/today

Quick lookup for today prediction.

Query parameters:

- include_staffing (default true)
- total_rooms (default 60)

Response example:

```json
{
  "date": "2026-04-01",
  "prediction": {
    "date": "2026-04-01",
    "predicted_rooms": 42.1,
    "lower_bound": 37.2,
    "upper_bound": 48.4,
    "demand_class": "medium",
    "occupancy_percentage": 70.17,
    "events": [],
    "is_overridden": false
  },
  "event_adjustment_applied": false,
  "event_note": null
}
```

### POST /override

Creates or updates manager override for a date.

Request body:

```json
{
  "date": "2026-04-05",
  "new_prediction": 55,
  "reason": "Large confirmed booking",
  "created_by": "manager_1",
  "include_staffing": true,
  "total_rooms": 60
}
```

Stored fields in forecast_overrides:

- date
- original_prediction
- new_prediction
- reason
- created_by
- created_at (database-generated)

### POST /feedback

Stores actual vs predicted outcome for learning loop.

Request body:

```json
{
  "date": "2026-04-01",
  "actual_rooms_sold": 46
}
```

Stored fields in actual_vs_predicted:

- date
- predicted
- actual
- error
- created_at (database-generated)

## Deployment Notes

- Render serves the API from prediction-api.
- Keep secrets in Render environment variables and GitHub Actions secrets.
- Do not commit local .env files or virtual environments.
