# Streamlit Web Runner (GSR_GPS_Shimmer)

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

## Deploy (keep code private)
- Make the GitHub repo **private**
- Use Streamlit Community Cloud (or Streamlit Teams) to deploy `streamlit_app/app.py`
- Users will only see the app, not the repository.

## Inputs
- GSR CSV (Shimmer)
- SQLite DB (eDiary)
- optional GPX track
- optional config JSON in sidebar (`gsr_gps_config.json`)

## Outputs
- merged CSV
- peaks CSV
- events CSV
- PNG plot
- KML + KMZ
- one ZIP containing all outputs
