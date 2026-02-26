import streamlit as st
import tempfile
from pathlib import Path
import json
import sys
from pathlib import Path

# Repo-Root in den Python-Pfad aufnehmen (damit "import src" funktioniert)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_processing.gsr_gps_core_runner import (
    PipelineConfig,
    detect_shimmer_columns,
    list_runs,
    build_run_items,
    run_pipeline,
)

st.set_page_config(page_title="GSR/GPS/Shimmer – Runner", layout="wide")

st.title("GSR + GPS + Feedback – Web Runner")
st.write("Uploads: **GSR CSV**, **SQLite DB**, optional **GPX**. Dann Run auswählen und Pipeline starten.")

with st.sidebar:
    st.header("Konfiguration")
    cfg_upload = st.file_uploader("Optional: gsr_gps_config.json", type=["json"])
    if cfg_upload:
        cfg_dict = json.loads(cfg_upload.getvalue().decode("utf-8"))
        cfg = PipelineConfig.from_dict(cfg_dict)
        st.success("Config geladen.")
    else:
        cfg = PipelineConfig()
        st.info("Default-Config aktiv.")
    cfg.output_prefix = st.text_input("Output Prefix", value=cfg.output_prefix)

col1, col2, col3 = st.columns(3)
with col1:
    gsr_csv = st.file_uploader("1) GSR CSV (Shimmer)", type=["csv"])
with col2:
    db_file = st.file_uploader("2) SQLite DB (eDiary)", type=["db"])
with col3:
    gpx_file = st.file_uploader("3) GPX (optional)", type=["gpx"])

if not (gsr_csv and db_file):
    st.warning("Bitte mindestens **CSV** und **DB** hochladen.")
    st.stop()

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    gsr_path = tmpdir / gsr_csv.name
    db_path = tmpdir / db_file.name
    gsr_path.write_bytes(gsr_csv.getbuffer())
    db_path.write_bytes(db_file.getbuffer())

    gpx_path = None
    if gpx_file is not None:
        gpx_path = tmpdir / gpx_file.name
        gpx_path.write_bytes(gpx_file.getbuffer())

    # Shimmer column selection
    ts_cands, gsr_cands = detect_shimmer_columns(str(gsr_path))
    st.subheader("Shimmer Spaltenauswahl")
    if not ts_cands or not gsr_cands:
        st.error("Konnte Timestamp/GSR-Spalten nicht automatisch erkennen. Bitte prüfe das CSV-Format.")
        st.stop()

    cA, cB = st.columns(2)
    with cA:
        ts_col = st.selectbox("Timestamp-Spalte", options=ts_cands, index=0)
    with cB:
        gsr_col = st.selectbox("GSR/Conductance-Spalte", options=gsr_cands, index=0)

    # Run selection
    st.subheader("Run Auswahl (aus DB)")
    try:
        # Need GSR range for overlap sorting: quick-read timestamps only
        import pandas as pd
        df_preview = pd.read_csv(str(gsr_path), sep="\\t", header=1, skiprows=[2], usecols=[ts_col])
        gsr_min = pd.to_datetime(df_preview[ts_col], unit="ms", errors="coerce").min()
        gsr_max = pd.to_datetime(df_preview[ts_col], unit="ms", errors="coerce").max()
    except Exception:
        gsr_min = None
        gsr_max = None

    runs_df = list_runs(str(db_path))
    run_items = build_run_items(runs_df, gsr_min, gsr_max) if (gsr_min is not None and gsr_max is not None) else []

    if run_items:
        options = {it["display"]: it["run_id"] for it in run_items}
        run_display = st.selectbox("Run (sortiert nach Overlap mit GSR)", options=list(options.keys()), index=0)
        run_id = options[run_display]
    else:
        st.info("Keine Runs gefunden oder Spalten nicht erkannt. Es wird ohne Run-Cutoff gearbeitet.")
        run_id = None

    st.divider()

    run_btn = st.button("▶ Pipeline starten", type="primary")
    if run_btn:
        out_dir = tmpdir / "results"
        out_dir.mkdir(parents=True, exist_ok=True)

        with st.spinner("Pipeline läuft…"):
            res = run_pipeline(
                gsr_csv_path=str(gsr_path),
                db_path=str(db_path),
                gpx_path=str(gpx_path) if gpx_path else None,
                out_dir=str(out_dir),
                cfg=cfg,
                run_id=run_id,
                shimmer_ts_col=ts_col,
                shimmer_gsr_col=gsr_col,
            )

        st.success("✅ Fertig!")
        st.write(f"GPS-Quelle: **{res['gps_used']}**")

        outputs = res["outputs"]
        st.subheader("Download")
        zip_bytes = Path(outputs["zip"]).read_bytes()
        st.download_button("⬇️ Alle Outputs als ZIP", data=zip_bytes, file_name=Path(outputs["zip"]).name, mime="application/zip")

        st.caption("Einzeldateien:")
        cols = st.columns(3)
        def dl(i, label, key, mime):
            p = Path(outputs[key])
            cols[i].download_button(label, data=p.read_bytes(), file_name=p.name, mime=mime)

        dl(0, "CSV merged", "csv_all", "text/csv")
        dl(0, "CSV peaks", "csv_peaks", "text/csv")
        dl(1, "CSV events", "csv_events", "text/csv")
        dl(1, "Plot PNG", "plot_png", "image/png")
        dl(2, "KML", "kml", "application/vnd.google-earth.kml+xml")
        dl(2, "KMZ", "kmz", "application/vnd.google-earth.kmz")

        st.image(outputs["plot_png"], caption="Plot", use_container_width=True)
