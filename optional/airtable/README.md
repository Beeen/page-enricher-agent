Optional Airtable Integration

- File: `optional/airtable/airtable_integration.py`
- Install: `pip install pyairtable`
- Env (add to `backend/.env`):
  - `AIRTABLE_API_KEY=...`
  - `AIRTABLE_BASE_ID=...`
  - `AIRTABLE_TABLE_NAME=trip_planner_traces` (optional)

Usage
- Import and use in a custom route or workflow to log requests/responses for manual labeling.
- Not required for core trip planning; safe to ignore in production.

