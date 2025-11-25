# -*- coding: utf-8 -*-
"""
GSR + GPS + Feedback + RUN-Metadaten (v5.2, modularisiert)
---------------------------------------------------------
- Pipeline-Funktion statt Monolith-Skript
- Geeignet für Nutzung aus main.py oder Jupyter-Notebooks
"""

import os
import re
import sqlite3
from datetime import datetime, timedelta
from zipfile import ZipFile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
from scipy.signal import find_peaks, butter, filtfilt


# ======================= Settings =========================

@dataclass
class GsrGpsConfig:
    ts_col: str = "Shimmer_C078_Timestamp_Unix_CAL"
    gsr_col: str = "Shimmer_C078_GSR_Skin_Conductance_CAL"

    gps_table: str = "locationEventData"
    feedback_table: str = "feedbackEventData"
    run_table: str = "run"

    scr_offset_us: float = 0.075
    min_peak_distance_s: float = 10
    min_peak_prominence_us: float = 0.1
    stimulus_window_s: float = 10
    resample_hz: float = 1
    merge_tolerance_ms: int = 4000

    scl_baseline_cutoff_hz: float = 0.01
    scl_global_cutoff_hz: float = 0.02

    scl_window_s: int = 10  # Fenster für lokalen SCL (Sekunden)


# ==================== Helper Functions =====================

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    return filtfilt(b, a, data)


def lowpass(series, cutoff_hz, fs, order=4):
    x = series.astype(float).interpolate().bfill().ffill().to_numpy()
    y = butter_lowpass_filter(x, cutoff_hz, fs, order)
    return pd.Series(y, index=series.index)


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def sanitize_for_filename(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "NA"
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_")
    return s if s else "NA"


def map_gender(g):
    try:
        if pd.isna(g):
            return "NA"
        if isinstance(g, (int, np.integer, float, np.floating)):
            gi = int(g)
            if gi == 1:
                return "male"
            if gi == 0:
                return "female"
            return str(gi)
        s = str(g).strip().lower()
        if s in ["1", "m", "male", "mann"]:
            return "male"
        if s in ["0", "f", "female", "frau"]:
            return "female"
        return s
    except Exception:
        return "NA"


# ==================== 1) GSR ===============================

def load_gsr(csv_path: str, cfg: GsrGpsConfig) -> pd.DataFrame:
    print("📥 Lade GSR…")
    df_gsr = pd.read_csv(csv_path, sep="\t", header=1, skiprows=[2])
    df_gsr = df_gsr.drop(columns=[c for c in df_gsr.columns if c.startswith("Unnamed")],
                         errors="ignore")
    df_gsr["Timestamp"] = pd.to_datetime(df_gsr[cfg.ts_col], unit="ms", errors="coerce")
    df_gsr = df_gsr[["Timestamp", cfg.gsr_col]].rename(columns={cfg.gsr_col: "Conductance"})
    df_gsr = (
        df_gsr
        .dropna()
        .set_index("Timestamp")
        .resample("1s")
        .mean()
        .interpolate()
        .reset_index()
    )
    t_min, t_max = df_gsr["Timestamp"].min(), df_gsr["Timestamp"].max()
    print(f"⏱ Zeitraum GSR: {t_min} → {t_max}")
    return df_gsr


# ==================== 2) RUN-Metadaten =====================

def get_run_metadata(conn: sqlite3.Connection,
                     t_min: pd.Timestamp,
                     t_max: pd.Timestamp,
                     cfg: GsrGpsConfig) -> dict:
    try:
        runs = pd.read_sql_query(f"SELECT * FROM {cfg.run_table}", conn)
        print(f"📘 Run-Tabelle gefunden: {cfg.run_table}")
    except Exception:
        print(f"⚠️ Tabelle '{cfg.run_table}' nicht gefunden. Suche nach alternativen Run-Tabellen…")
        cursor = conn.cursor()
        table_names = [r[0] for r in cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        run_like = [t for t in table_names if "run" in t.lower()]
        if not run_like:
            print("⚠️ Keine Run-Tabelle gefunden. Fahre ohne Run-Metadaten fort.")
            return dict(runId="NA", study="NA", name="NA", birthYear="NA", gender="NA")
        else:
            cfg.run_table = run_like[0]
            print(f"📘 Verwende Run-Tabelle: {cfg.run_table}")
            runs = pd.read_sql_query(f"SELECT * FROM {cfg.run_table}", conn)

    if runs.empty:
        return dict(runId="NA", study="NA", name="NA", birthYear="NA", gender="NA")

    col_id = pick_col(runs, ["id", "runId"])
    col_start = pick_col(runs, ["start"])
    col_end = pick_col(runs, ["end"])
    col_study = pick_col(runs, ["study"])
    col_name = pick_col(runs, ["name"])
    col_birthY = pick_col(runs, ["birthYear"])
    col_gender = pick_col(runs, ["gender"])

    runs["start_dt"] = pd.to_datetime(runs[col_start], unit="ms", errors="coerce")
    runs["end_dt"] = pd.to_datetime(runs[col_end], unit="ms", errors="coerce")

    def overlap_seconds(a_start, a_end, b_start, b_end):
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        return max(0.0, (end - start).total_seconds())

    runs["overlap_s"] = runs.apply(
        lambda r: overlap_seconds(t_min, t_max, r["start_dt"], r["end_dt"]), axis=1
    )
    runs_valid = runs[runs["overlap_s"] > 0].copy()

    if runs_valid.empty:
        print("⚠️ Kein überlappender Run gefunden – keine Metadaten.")
        return dict(runId="NA", study="NA", name="NA", birthYear="NA", gender="NA")

    best_run = runs_valid.sort_values("overlap_s", ascending=False).iloc[0]
    run_meta = dict(
        runId=best_run[col_id],
        study=best_run.get(col_study, "NA"),
        name=best_run.get(col_name, "NA"),
        birthYear=best_run.get(col_birthY, "NA"),
        gender=map_gender(best_run.get(col_gender, "NA"))
    )
    print(f"✅ Gewählter Run: id={run_meta['runId']} | study={run_meta['study']}")
    return run_meta


# ==================== 3) SCR Detection =====================

def detect_scr(df_gsr: pd.DataFrame, cfg: GsrGpsConfig) -> pd.DataFrame:
    print("🔎 SCR-Detektion…")

    # Lokaler SCL
    df_gsr["SCL_local"] = df_gsr["Conductance"].rolling(
        window=cfg.scl_window_s,
        min_periods=1
    ).mean()
    df_gsr["Threshold"] = df_gsr["SCL_local"] + cfg.scr_offset_us

    sig = df_gsr["Conductance"].astype(float).to_numpy()
    thr = df_gsr["Threshold"].astype(float).to_numpy()

    scr_indices, props = find_peaks(
        sig, height=thr,
        distance=int(cfg.min_peak_distance_s * cfg.resample_hz),
        prominence=cfg.min_peak_prominence_us
    )

    df_gsr["SCR_Peak"] = 0
    df_gsr.loc[scr_indices, "SCR_Peak"] = 1
    print(f"✅ SCR-Peaks erkannt: {int(df_gsr['SCR_Peak'].sum())}")

    df_gsr["Trigger"] = 0
    df_gsr["SCR_Latency_s"] = np.nan
    last_trigger_time = None

    for idx in scr_indices:
        peak_time = df_gsr.loc[idx, "Timestamp"]
        pre_idx = idx
        while pre_idx > 0 and df_gsr.loc[pre_idx, "Conductance"] > df_gsr.loc[pre_idx, "SCL_local"]:
            pre_idx -= 1
        stim_time = df_gsr.loc[pre_idx, "Timestamp"]
        if (last_trigger_time is None) or ((stim_time - last_trigger_time).total_seconds() > cfg.stimulus_window_s):
            df_gsr.loc[pre_idx, "Trigger"] = 1
            df_gsr.loc[idx, "SCR_Latency_s"] = (peak_time - stim_time).total_seconds()
            last_trigger_time = stim_time

    # SCL Lowpass
    df_gsr = df_gsr.set_index("Timestamp")
    df_gsr["SCL_Baseline"] = lowpass(df_gsr["Conductance"], cfg.scl_baseline_cutoff_hz, cfg.resample_hz)
    df_gsr["SCL_Global"] = lowpass(df_gsr["Conductance"], cfg.scl_global_cutoff_hz, cfg.resample_hz)
    df_gsr = df_gsr.reset_index()

    return df_gsr


# ==================== 4) GPS & Feedback ====================

def load_gps_feedback(conn: sqlite3.Connection,
                      run_meta: dict,
                      cfg: GsrGpsConfig,
                      merge_tolerance_ms: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gps = pd.read_sql_query(f"SELECT * FROM {cfg.gps_table}", conn)
    fb = pd.read_sql_query(f"SELECT * FROM {cfg.feedback_table}", conn)

    gps_run_col = pick_col(gps, ["runId"])
    fb_run_col = pick_col(fb, ["runId"])

    # Nur auf Run filtern, wenn wir einen haben und die Spalte existiert
    if str(run_meta["runId"]) != "NA" and gps_run_col is not None:
        gps = gps[gps[gps_run_col] == run_meta["runId"]]
    if str(run_meta["runId"]) != "NA" and fb_run_col is not None:
        fb = fb[fb[fb_run_col] == run_meta["runId"]]

    gps["Timestamp"] = pd.to_datetime(gps["timestamp"], unit="ms", errors="coerce")
    gps = gps[["Timestamp", "latitude", "longitude"]].dropna()
    gps = gps.set_index("Timestamp").resample("1s").nearest().reset_index()

    col_feeling = pick_col(fb, ["feelingDescription", "feeling", "emotion"])
    col_note = pick_col(fb, ["note", "comment", "description"])
    if col_feeling is None:
        fb["feeling"] = ""
    else:
        fb = fb.rename(columns={col_feeling: "feeling"})
    if col_note is None:
        fb["note"] = ""
    else:
        fb = fb.rename(columns={col_note: "note"})

    fb["Timestamp"] = pd.to_datetime(fb["timestamp"], unit="ms", errors="coerce")
    fb = fb.dropna(subset=["Timestamp"])[["Timestamp", "feeling", "note"]]

    feedback_geo = pd.merge_asof(
        fb.sort_values("Timestamp"),
        gps.sort_values("Timestamp"),
        on="Timestamp", direction="nearest",
        tolerance=pd.Timedelta(milliseconds=merge_tolerance_ms)
    ).dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    return gps, fb, feedback_geo


# ==================== 5) Merge & Export ====================

def merge_streams(df_gsr: pd.DataFrame,
                  gps: pd.DataFrame,
                  run_meta: dict,
                  merge_tolerance_ms: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = pd.merge_asof(
        df_gsr.sort_values("Timestamp"),
        gps.sort_values("Timestamp"),
        on="Timestamp", direction="nearest",
        tolerance=pd.Timedelta(milliseconds=merge_tolerance_ms)
    ).dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    for k, v in run_meta.items():
        merged[k] = v
        df_gsr[k] = v

    peaks_geo = merged[merged["SCR_Peak"] == 1].copy()
    trigs_geo = merged[merged["Trigger"] == 1].copy()

    return merged, peaks_geo, trigs_geo


def export_csvs(merged: pd.DataFrame,
                peaks_geo: pd.DataFrame,
                feedback_geo: pd.DataFrame,
                run_meta: dict,
                out_dir: str) -> dict:
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{sanitize_for_filename(run_meta.get('study'))}_run{sanitize_for_filename(run_meta.get('runId'))}_" \
           f"{sanitize_for_filename(run_meta.get('gender'))}_{sanitize_for_filename(run_meta.get('birthYear'))}_" \
           f"{time_str}"

    os.makedirs(out_dir, exist_ok=True)

    out_all = os.path.join(out_dir, f"output_GSR_GPS_{base}.csv")
    out_peaks = os.path.join(out_dir, f"output_GSR_GPS_SCRonly_{base}.csv")
    out_events = os.path.join(out_dir, f"output_Feedback_to_SCR_{base}.csv")

    merged.to_csv(out_all, index=False)
    peaks_geo.to_csv(out_peaks, index=False)
    feedback_geo.to_csv(out_events, index=False)

    print("💾 CSV:", out_all)
    print("💾 Peaks:", out_peaks)
    print("💾 Events:", out_events)

    return dict(all=out_all, peaks=out_peaks, events=out_events, base=base)


# ==================== 6) KML / KMZ =========================

def export_kml_kmz(merged: pd.DataFrame,
                   base_label: str,
                   out_dir: str) -> dict:
    def safe_coord(v):
        try:
            if pd.isna(v):
                return None
            return float(v)
        except Exception:
            return None

    conductance_min = merged["Conductance"].min()
    conductance_max = merged["Conductance"].max()
    height_scale_m = 100
    alpha = 128  # 50% Transparenz

    def color_for_value(v):
        norm = (v - conductance_min) / (conductance_max - conductance_min)
        norm = min(max(norm, 0), 1)
        r = int(255 * norm)
        g = int(128 * (1 - abs(0.5 - norm) * 2))
        b = int(255 * (1 - norm))
        return f"{alpha:02x}{b:02x}{g:02x}{r:02x}"  # AABBGGRR

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(out_dir, exist_ok=True)

    kml_filename = os.path.join(out_dir, f"output_GSR_3DTrack_{base_label}_{time_str}.kml")
    kmz_filename = os.path.join(out_dir, f"output_GSR_3DTrack_{base_label}_{time_str}.kmz")

    kml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '  <Document>',
        f'    <name>GSR_3D_Track_{time_str}</name>',
        '    <open>1</open>'
    ]

    # Blauer Pfad
    kml_lines.append('    <Placemark>')
    kml_lines.append('      <name>GPS Track</name>')
    kml_lines.append('      <Style><LineStyle>')
    kml_lines.append('        <color>800000ff</color>')
    kml_lines.append('        <width>4</width>')
    kml_lines.append('      </LineStyle></Style>')
    kml_lines.append('      <LineString>')
    kml_lines.append('        <tessellate>1</tessellate>')
    kml_lines.append('        <altitudeMode>relativeToGround</altitudeMode>')
    kml_lines.append('        <coordinates>')

    for _, r in merged.sort_values("Timestamp").iterrows():
        lon, lat = safe_coord(r["longitude"]), safe_coord(r["latitude"])
        if lon is None or lat is None:
            continue
        height = (r["Conductance"] - conductance_min) / (conductance_max - conductance_min) * height_scale_m
        kml_lines.append(f'          {lon},{lat},{height}')

    kml_lines.append('        </coordinates>')
    kml_lines.append('      </LineString>')
    kml_lines.append('    </Placemark>')

    # Vertikale Stäbe
    kml_lines.append('    <Folder>')
    kml_lines.append('      <name>Conductance Vertical Bars</name>')
    for _, r in merged.iterrows():
        lon, lat = safe_coord(r["longitude"]), safe_coord(r["latitude"])
        if lon is None or lat is None:
            continue
        height = (r["Conductance"] - conductance_min) / (conductance_max - conductance_min) * height_scale_m
        color = color_for_value(r["Conductance"])

        kml_lines.append('      <Placemark>')
        kml_lines.append('        <Style><LineStyle>')
        kml_lines.append(f'          <color>{color}</color>')
        kml_lines.append('          <width>3</width>')
        kml_lines.append('        </LineStyle></Style>')
        kml_lines.append('        <LineString>')
        kml_lines.append('          <altitudeMode>relativeToGround</altitudeMode>')
        kml_lines.append('          <coordinates>')
        kml_lines.append(f'            {lon},{lat},0')
        kml_lines.append(f'            {lon},{lat},{height}')
        kml_lines.append('          </coordinates>')
        kml_lines.append('        </LineString>')
        kml_lines.append('      </Placemark>')
    kml_lines.append('    </Folder>')

    kml_lines.append('  </Document>')
    kml_lines.append('</kml>')

    with open(kml_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(kml_lines))

    with ZipFile(kmz_filename, "w") as kmz:
        kmz.write(kml_filename, arcname=os.path.basename(kml_filename))

    print(f"🌍 KML gespeichert: {kml_filename}")
    print(f"📦 KMZ gespeichert: {kmz_filename}")

    return dict(kml=kml_filename, kmz=kmz_filename)


# ==================== 7) Plot ==============================

def plot_gsr_gps(df_gsr: pd.DataFrame,
                 merged: pd.DataFrame,
                 peaks_geo: pd.DataFrame,
                 trigs_geo: pd.DataFrame,
                 fb: pd.DataFrame,
                 feedback_geo: pd.DataFrame,
                 run_meta: dict,
                 out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 0.6])
    ax1, ax2, ax3 = gs.subplots()

    fig.suptitle(f"GSR + GPS (v5.2) – {run_meta.get('study')} / Run {run_meta.get('runId')}", fontsize=14)

    # Signal
    ax1.plot(df_gsr["Timestamp"], df_gsr["Conductance"], color="steelblue", label="Conductance")
    ax1.plot(df_gsr["Timestamp"], df_gsr["SCL_Global"], color="#ffb6c1", linewidth=1.6, label="SCL 0.02 Hz")
    ax1.plot(df_gsr["Timestamp"], df_gsr["SCL_Baseline"], color="green", linewidth=1.6, label="Baseline 0.01 Hz")

    # Trigger
    ax1.scatter(
        df_gsr.loc[df_gsr["Trigger"] == 1, "Timestamp"],
        df_gsr.loc[df_gsr["Trigger"] == 1, "Conductance"],
        color="red", marker="*", s=130, label="Trigger"
    )

    # Peaks
    ax1.scatter(
        df_gsr.loc[df_gsr["SCR_Peak"] == 1, "Timestamp"],
        df_gsr.loc[df_gsr["SCR_Peak"] == 1, "Conductance"],
        color="orange", marker="^", s=100, label="SCR Peak"
    )

    # Verbindung Trigger→Peak
    for idx, row in df_gsr[df_gsr["SCR_Peak"] == 1].iterrows():
        lat = row["SCR_Latency_s"]
        if pd.notna(lat):
            t_peak = row["Timestamp"]
            t_trig = t_peak - timedelta(seconds=float(lat))
            y_trig = df_gsr.loc[df_gsr["Timestamp"] == t_trig, "Conductance"]
            if len(y_trig) > 0:
                ax1.plot([t_trig, t_peak], [y_trig.values[0], row["Conductance"]],
                         color="gray", linestyle="--", alpha=0.5)

    # Feeling-Annotations
    for i, ev in fb.iterrows():
        ax1.axvline(ev["Timestamp"], color="gold", linestyle="--", alpha=0.25)
        ax1.scatter(ev["Timestamp"], df_gsr["Conductance"].max() * 0.9,
                    color="gold", s=80, label="Event" if i == 0 else "")
        txt = f"#{i + 1} {ev['feeling'] or ''}\n{ev['note'] or ''}"
        ax1.text(ev["Timestamp"], df_gsr["Conductance"].max() * 0.93, txt,
                 rotation=90, fontsize=8, ha="right", va="center")

    ax1.set_ylabel("Conductance (µS)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Karte
    ax2.set_title("GPS-Track (● Events, ★ Trigger, ▲ Peaks)")
    ax2.scatter(merged["longitude"], merged["latitude"], s=8, c="blue", label="Pfad")

    if not peaks_geo.empty:
        ax2.scatter(
            peaks_geo["longitude"], peaks_geo["latitude"],
            color="orange", marker="^", s=70, label="SCR Peaks"
        )

    if not trigs_geo.empty:
        ax2.scatter(
            trigs_geo["longitude"], trigs_geo["latitude"],
            color="red", marker="*", s=80, label="Trigger"
        )

    for i, fbrow in feedback_geo.iterrows():
        ax2.scatter(
            fbrow["longitude"], fbrow["latitude"],
            color="gold", s=70, label="Event" if i == 0 else ""
        )

    try:
        ctx.add_basemap(ax2, crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.8)
    except Exception as e:
        print(f"⚠️ OSM-Fehler: {e}")

    ax2.legend(fontsize=9)
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")

    # Histogramm Latenzen
    latencies = df_gsr["SCR_Latency_s"].dropna()
    ax3.hist(latencies, bins=np.arange(0, 10, 0.5), color="lightcoral", edgecolor="k")
    ax3.set_title("Histogramm der SCR-Latenzen (Stimulus → Peak)")
    ax3.set_xlabel("Latenz (Sekunden)")
    ax3.set_ylabel("Anzahl")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_plotpng = os.path.join(out_dir, f"output_GSR_GPS_v5_2_{time_str}.png")
    plt.savefig(out_plotpng, dpi=200)
    plt.close()

    print(f"🖼️ Plot gespeichert: {out_plotpng}")
    return out_plotpng


# ==================== 8) High-Level Pipeline ===============

def run_gsr_gps_pipeline(
    gsr_csv_path: str,
    db_path: str,
    out_dir: str = "results",
    cfg: GsrGpsConfig | None = None
) -> dict:
    """
    Führt die komplette GSR+GPS+Feedback Pipeline aus.
    Gibt Pfade zu den wichtigsten Outputs zurück.
    """
    if cfg is None:
        cfg = GsrGpsConfig()

    # 1) GSR
    df_gsr = load_gsr(gsr_csv_path, cfg)
    t_min, t_max = df_gsr["Timestamp"].min(), df_gsr["Timestamp"].max()

    # 2) DB & Run
    conn = sqlite3.connect(db_path)
    run_meta = get_run_metadata(conn, t_min, t_max, cfg)

    # 3) SCR
    df_gsr = detect_scr(df_gsr, cfg)

    # 4) GPS & Feedback
    gps, fb, feedback_geo = load_gps_feedback(conn, run_meta, cfg, cfg.merge_tolerance_ms)
    conn.close()

    # 5) Merge
    merged, peaks_geo, trigs_geo = merge_streams(df_gsr, gps, run_meta, cfg.merge_tolerance_ms)

    # 6) CSV & KML/KMZ
    csv_paths = export_csvs(merged, peaks_geo, feedback_geo, run_meta, out_dir=os.path.join(out_dir, "csv"))
    kml_paths = export_kml_kmz(merged, csv_paths["base"], out_dir=os.path.join(out_dir, "kml"))

    # 7) Plot
    plot_path = plot_gsr_gps(
        df_gsr=df_gsr,
        merged=merged,
        peaks_geo=peaks_geo,
        trigs_geo=trigs_geo,
        fb=fb,
        feedback_geo=feedback_geo,
        run_meta=run_meta,
        out_dir=os.path.join(out_dir, "plots")
    )

    print("✅ Pipeline erfolgreich ausgeführt.")
    return {
        "csv": csv_paths,
        "kml": kml_paths,
        "plot": plot_path,
        "run_meta": run_meta,
    }
