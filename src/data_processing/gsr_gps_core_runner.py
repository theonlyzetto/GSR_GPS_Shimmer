# -*- coding: utf-8 -*-
"""
Core runner for GSR/GPS/Feedback pipeline (Streamlit-ready).

Design goals:
- No tkinter / no dialogs
- Pure functions: pass paths + config + selections
- GPX optional (preferred GPS source)
- Run selection provided by caller (e.g., Streamlit dropdown)
"""

from __future__ import annotations

import os
import re
import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from zipfile import ZipFile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt


GPS_TABLE_NAME = "locationEventData"
FEEDBACK_TABLE = "feedbackEventData"
RUN_TABLE = "run"


# -------------------- Config --------------------

@dataclass
class PipelineConfig:
    sampling_rate_hz: int = 4
    scr_offset: float = 0.08
    min_peak_distance_s: int = 5
    min_peak_prominence: float = 0.1  # fallback if config doesn't provide
    stimulus_window_s: int = 10
    merge_tolerance_ms: int = 4000
    initial_cut_s: int = 0

    eda_lowpass_enable: bool = True
    eda_lowpass_cutoff_hz: float = 1.0
    eda_lowpass_order: int = 2

    scl_baseline_cutoff_hz: float = 0.0005
    scl_global_cutoff_hz: float = 0.002

    output_prefix: str = "Run"

    @staticmethod
    def from_dict(d: dict) -> "PipelineConfig":
        cfg = PipelineConfig()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


# -------------------- Helpers --------------------

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def try_pick_by_substring(columns: list[str], must_contain: list[str]) -> list[str]:
    out = []
    cols_lower = [(c, c.lower()) for c in columns]
    for c, cl in cols_lower:
        ok = True
        for m in must_contain:
            if m.lower() not in cl:
                ok = False
                break
        if ok:
            out.append(c)
    return out

def safe_to_dt(x):
    try:
        # numeric ms epoch
        if isinstance(x, (int, float, np.integer, np.floating)) and not pd.isna(x):
            return pd.to_datetime(int(x), unit="ms", errors="coerce")
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return pd.NaT

def _safe_slug(s: str) -> str:
    s = "" if s is None else str(s)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_")
    return s if s else "NA"

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    return filtfilt(b, a, data)

def lowpass(series: pd.Series, cutoff_hz: float, fs: float, order=4) -> pd.Series:
    x = series.astype(float).interpolate().bfill().ffill().to_numpy()
    y = butter_lowpass_filter(x, cutoff_hz, fs, order)
    return pd.Series(y, index=series.index)

# -------------------- GPX --------------------

def read_gpx_file(gpx_path: str) -> pd.DataFrame:
    """
    Minimal GPX parser using xml.etree (no extra deps).
    Returns columns: Timestamp (datetime64[ns]), latitude, longitude
    """
    tree = ET.parse(gpx_path)
    root = tree.getroot()

    # namespaces can vary; detect 'trkpt' regardless of ns
    trkpts = root.findall(".//{*}trkpt")
    rows = []
    for pt in trkpts:
        lat = pt.attrib.get("lat", None)
        lon = pt.attrib.get("lon", None)
        t_el = pt.find("{*}time")
        if lat is None or lon is None or t_el is None or t_el.text is None:
            continue
        ts = pd.to_datetime(t_el.text, errors="coerce", utc=True)
        if pd.isna(ts):
            continue
        rows.append((ts.tz_convert(None), float(lat), float(lon)))

    gps = pd.DataFrame(rows, columns=["Timestamp", "latitude", "longitude"])
    gps = gps.dropna().sort_values("Timestamp")
    return gps

def normalize_gps_1hz(gps: pd.DataFrame) -> pd.DataFrame:
    gps = gps.dropna(subset=["Timestamp","latitude","longitude"]).sort_values("Timestamp")
    gps = gps.set_index("Timestamp").resample("1s").nearest().reset_index()
    return gps

def best_gpx_offset_hours(gps: pd.DataFrame, run_start: pd.Timestamp, run_end: pd.Timestamp, max_abs_h: int = 3) -> int:
    """
    Try offsets in [-max_abs_h..+max_abs_h] hours and choose the one that yields most GPS points inside run window.
    """
    if gps.empty or pd.isna(run_start) or pd.isna(run_end):
        return 0
    best_h = 0
    best_cnt = -1
    for h in range(-max_abs_h, max_abs_h + 1):
        t = gps["Timestamp"] + pd.Timedelta(hours=h)
        cnt = ((t >= run_start) & (t <= run_end)).sum()
        if cnt > best_cnt:
            best_cnt = cnt
            best_h = h
    return best_h


# -------------------- DB: runs / gps / feedback --------------------

def list_runs(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        runs = pd.read_sql_query(f"SELECT * FROM {RUN_TABLE}", conn)
    finally:
        conn.close()
    return runs

def build_run_items(runs: pd.DataFrame, gsr_min: pd.Timestamp, gsr_max: pd.Timestamp) -> list[dict]:
    if runs is None or runs.empty:
        return []
    col_id = pick_col(runs, ["id", "runId", "run_id"])
    col_start = pick_col(runs, ["start", "startTime", "start_ts"])
    col_end = pick_col(runs, ["end", "endTime", "end_ts"])
    if not (col_id and col_start and col_end):
        return []

    rs = runs.copy()
    rs["start_dt"] = rs[col_start].apply(safe_to_dt)
    rs["end_dt"] = rs[col_end].apply(safe_to_dt)

    def overlap_s(a_start, a_end, b_start, b_end):
        if pd.isna(b_start) or pd.isna(b_end):
            return 0.0
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        return max(0.0, (end - start).total_seconds())

    items = []
    for _, r in rs.iterrows():
        it = {
            "run_id": str(r[col_id]),
            "start": r["start_dt"],
            "end": r["end_dt"],
            "study": str(r.get("study", "")).strip(),
            "name": str(r.get("name", "")).strip(),
        }
        it["overlap_s"] = overlap_s(gsr_min, gsr_max, it["start"], it["end"])
        it["display"] = f"{it['study']} | {it['name']} | run_id={it['run_id']} | {it['start']} -> {it['end']} | overlap={int(it['overlap_s'])}s".strip(" |")
        items.append(it)

    items = sorted(items, key=lambda x: x["overlap_s"], reverse=True)
    return items

def read_feedback(db_path: str, run_id: str | None = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        fb = pd.read_sql_query(f"SELECT * FROM {FEEDBACK_TABLE}", conn)
    finally:
        conn.close()

    run_col = pick_col(fb, ["runId", "run_id"])
    if run_id and run_col:
        fb = fb[fb[run_col].astype(str) == str(run_id)].copy()

    fb["Timestamp"] = pd.to_datetime(fb.get("timestamp"), unit="ms", errors="coerce")
    col_feeling = pick_col(fb, ["feelingDescription", "feeling", "emotion"])
    col_note = pick_col(fb, ["note", "comment", "description"])
    if col_feeling and col_feeling != "feeling":
        fb = fb.rename(columns={col_feeling: "feeling"})
    if "feeling" not in fb.columns:
        fb["feeling"] = ""
    if col_note and col_note != "note":
        fb = fb.rename(columns={col_note: "note"})
    if "note" not in fb.columns:
        fb["note"] = ""

    fb = fb.dropna(subset=["Timestamp"])[["Timestamp", "feeling", "note"]].sort_values("Timestamp")
    return fb

def read_gps_from_db(db_path: str, run_id: str | None = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        gps = pd.read_sql_query(f"SELECT * FROM {GPS_TABLE_NAME}", conn)
    finally:
        conn.close()

    run_col = pick_col(gps, ["runId", "run_id"])
    if run_id and run_col:
        gps = gps[gps[run_col].astype(str) == str(run_id)].copy()

    gps["Timestamp"] = pd.to_datetime(gps.get("timestamp"), unit="ms", errors="coerce")
    gps = gps[["Timestamp", "latitude", "longitude"]].dropna().sort_values("Timestamp")
    gps = normalize_gps_1hz(gps)
    return gps


# -------------------- GSR --------------------

def read_gsr_csv(csv_path: str, ts_col: str, gsr_col: str, cfg: PipelineConfig) -> pd.DataFrame:
    # Shimmer files are often TSV with header offset; be robust:
    try:
        df = pd.read_csv(csv_path, sep="\t", header=1, skiprows=[2])
    except Exception:
        df = pd.read_csv(csv_path)

    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")
    if ts_col not in df.columns or gsr_col not in df.columns:
        raise FileNotFoundError(f"Spalten nicht gefunden. Timestamp='{ts_col}', GSR='{gsr_col}'.")
    df["Timestamp"] = pd.to_datetime(df[ts_col], unit="ms", errors="coerce")
    df = df[["Timestamp", gsr_col]].rename(columns={gsr_col: "Conductance"})
    df = df.dropna(subset=["Timestamp", "Conductance"]).sort_values("Timestamp")

    # optional initial cut
    if cfg.initial_cut_s and cfg.initial_cut_s > 0:
        t0 = df["Timestamp"].min() + pd.Timedelta(seconds=int(cfg.initial_cut_s))
        df = df[df["Timestamp"] >= t0].copy()

    # view at 1 Hz for merge/map
    df_1hz = df.set_index("Timestamp").resample("1s").mean().interpolate().reset_index()

    return df_1hz

def detect_scr(df_1hz: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    # local baseline as rolling mean window (10s)
    scl_window_s = 10
    df = df_1hz.copy()
    df["SCL_local"] = df["Conductance"].rolling(window=scl_window_s, min_periods=1).mean()
    df["Threshold"] = df["SCL_local"] + cfg.scr_offset

    sig = df["Conductance"].astype(float).to_numpy()
    thr = df["Threshold"].astype(float).to_numpy()

    scr_indices, props = find_peaks(
        sig,
        height=thr,
        distance=int(cfg.min_peak_distance_s * 1),
        prominence=cfg.min_peak_prominence
    )

    df["SCR_Peak"] = 0
    df.loc[scr_indices, "SCR_Peak"] = 1

    df["Trigger"] = 0
    df["SCR_Latency_s"] = np.nan
    last_trigger_time = None

    for idx in scr_indices:
        peak_time = df.loc[idx, "Timestamp"]
        pre_idx = idx
        while pre_idx > 0 and df.loc[pre_idx, "Conductance"] > df.loc[pre_idx, "SCL_local"]:
            pre_idx -= 1
        stim_time = df.loc[pre_idx, "Timestamp"]
        if (last_trigger_time is None) or ((stim_time - last_trigger_time).total_seconds() > cfg.stimulus_window_s):
            df.loc[pre_idx, "Trigger"] = 1
            df.loc[idx, "SCR_Latency_s"] = (peak_time - stim_time).total_seconds()
            last_trigger_time = stim_time

    # global SCLs
    df_lp = df.set_index("Timestamp")
    df_lp["SCL_Baseline"] = lowpass(df_lp["Conductance"], cfg.scl_baseline_cutoff_hz, 1, order=4)
    df_lp["SCL_Global"] = lowpass(df_lp["Conductance"], cfg.scl_global_cutoff_hz, 1, order=4)
    df = df_lp.reset_index()
    return df


# -------------------- Export --------------------

def export_kml_kmz(merged: pd.DataFrame, out_dir: str, label: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    conductance_min = merged["Conductance"].min()
    conductance_max = merged["Conductance"].max()
    height_scale_m = 100
    alpha = 128

    def safe_coord(v):
        try:
            if pd.isna(v): return None
            return float(v)
        except Exception:
            return None

    def color_for_value(v):
        denom = (conductance_max - conductance_min) if conductance_max != conductance_min else 1.0
        norm = (v - conductance_min) / denom
        norm = min(max(norm, 0), 1)
        r = int(255 * norm)
        g = int(128 * (1 - abs(0.5 - norm) * 2))
        b = int(255 * (1 - norm))
        return f"{alpha:02x}{b:02x}{g:02x}{r:02x}"

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    kml_filename = os.path.join(out_dir, f"output_GSR_3DTrack_{label}_{time_str}.kml")
    kmz_filename = os.path.join(out_dir, f"output_GSR_3DTrack_{label}_{time_str}.kmz")

    kml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '  <Document>',
        f'    <name>GSR_3D_Track_{time_str}</name>',
        '    <open>1</open>'
    ]

    # Track
    kml_lines += [
        '    <Placemark>',
        '      <name>GPS Track</name>',
        '      <Style><LineStyle>',
        '        <color>800000ff</color>',
        '        <width>4</width>',
        '      </LineStyle></Style>',
        '      <LineString>',
        '        <tessellate>1</tessellate>',
        '        <altitudeMode>relativeToGround</altitudeMode>',
        '        <coordinates>'
    ]

    denom = (conductance_max - conductance_min) if conductance_max != conductance_min else 1.0
    for _, r in merged.sort_values("Timestamp").iterrows():
        lon, lat = safe_coord(r["longitude"]), safe_coord(r["latitude"])
        if lon is None or lat is None:
            continue
        height = (r["Conductance"] - conductance_min) / denom * height_scale_m
        kml_lines.append(f'          {lon},{lat},{height}')

    kml_lines += [
        '        </coordinates>',
        '      </LineString>',
        '    </Placemark>'
    ]

    # Bars
    kml_lines += ['    <Folder>', '      <name>Conductance Vertical Bars</name>']
    for _, r in merged.iterrows():
        lon, lat = safe_coord(r["longitude"]), safe_coord(r["latitude"])
        if lon is None or lat is None:
            continue
        height = (r["Conductance"] - conductance_min) / denom * height_scale_m
        color = color_for_value(r["Conductance"])
        kml_lines += [
            '      <Placemark>',
            '        <Style><LineStyle>',
            f'          <color>{color}</color>',
            '          <width>3</width>',
            '        </LineStyle></Style>',
            '        <LineString>',
            '          <altitudeMode>relativeToGround</altitudeMode>',
            '          <coordinates>',
            f'            {lon},{lat},0',
            f'            {lon},{lat},{height}',
            '          </coordinates>',
            '        </LineString>',
            '      </Placemark>'
        ]
    kml_lines += ['    </Folder>', '  </Document>', '</kml>']

    with open(kml_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(kml_lines))
    with ZipFile(kmz_filename, "w") as kmz:
        kmz.write(kml_filename, arcname=os.path.basename(kml_filename))

    return {"kml": kml_filename, "kmz": kmz_filename}

def plot_summary(df_gsr: pd.DataFrame, merged: pd.DataFrame, feedback_geo: pd.DataFrame, out_dir: str, title: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    ax1, ax2 = gs.subplots()

    fig.suptitle(title, fontsize=14)

    ax1.plot(df_gsr["Timestamp"], df_gsr["Conductance"], label="Conductance", alpha=0.9)
    if "SCL_Global" in df_gsr.columns:
        ax1.plot(df_gsr["Timestamp"], df_gsr["SCL_Global"], label="SCL Global", linewidth=1.3)
    if "SCL_Baseline" in df_gsr.columns:
        ax1.plot(df_gsr["Timestamp"], df_gsr["SCL_Baseline"], label="SCL Baseline", linewidth=1.3)

    trigs = df_gsr[df_gsr["Trigger"] == 1]
    peaks = df_gsr[df_gsr["SCR_Peak"] == 1]
    if not trigs.empty:
        ax1.scatter(trigs["Timestamp"], trigs["Conductance"], marker="*", s=90, label="Trigger")
    if not peaks.empty:
        ax1.scatter(peaks["Timestamp"], peaks["Conductance"], marker="^", s=60, label="SCR Peak")

    ax1.set_ylabel("Conductance (µS)")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Map scatter
    ax2.scatter(merged["longitude"], merged["latitude"], s=6, alpha=0.6, label="Track")
    if not feedback_geo.empty:
        ax2.scatter(feedback_geo["longitude"], feedback_geo["latitude"], s=40, marker="o", label="Feedback")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.legend()
    ax2.grid(alpha=0.2)

    out_png = os.path.join(out_dir, f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    return out_png

def zip_outputs(paths: list[str], zip_path: str) -> str:
    with ZipFile(zip_path, "w") as z:
        for p in paths:
            if p and os.path.exists(p):
                z.write(p, arcname=os.path.basename(p))
    return zip_path


# -------------------- High-level run --------------------

def run_pipeline(
    gsr_csv_path: str,
    db_path: str,
    gpx_path: str | None,
    out_dir: str,
    cfg: PipelineConfig,
    run_id: str | None,
    shimmer_ts_col: str,
    shimmer_gsr_col: str,
) -> dict:
    """
    Executes the full pipeline. Returns dict with output file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    df_gsr = read_gsr_csv(gsr_csv_path, shimmer_ts_col, shimmer_gsr_col, cfg)
    df_gsr = detect_scr(df_gsr, cfg)

    # Run window (for GPX offset; and optional cutoff)
    run_start = None
    run_end = None
    if run_id:
        runs = list_runs(db_path)
        col_id = pick_col(runs, ["id", "runId", "run_id"])
        col_start = pick_col(runs, ["start", "startTime", "start_ts"])
        col_end = pick_col(runs, ["end", "endTime", "end_ts"])
        if col_id and col_start and col_end:
            rr = runs[runs[col_id].astype(str) == str(run_id)]
            if not rr.empty:
                r = rr.iloc[0]
                run_start = safe_to_dt(r[col_start])
                run_end = safe_to_dt(r[col_end])

    # Apply cutoff if we have run window
    if run_start is not None and run_end is not None and not pd.isna(run_start) and not pd.isna(run_end):
        gsr_min = df_gsr["Timestamp"].min()
        gsr_max = df_gsr["Timestamp"].max()
        eff_start = max(run_start, gsr_min)
        eff_end = min(run_end, gsr_max)
        df_gsr = df_gsr[(df_gsr["Timestamp"] >= eff_start) & (df_gsr["Timestamp"] <= eff_end)].copy()

    # GPS: GPX preferred
    gps = pd.DataFrame(columns=["Timestamp","latitude","longitude"])
    gps_used = "none"
    if gpx_path:
        gps_gpx = read_gpx_file(gpx_path)
        if run_start is not None and run_end is not None and not gps_gpx.empty:
            h = best_gpx_offset_hours(gps_gpx, run_start, run_end, max_abs_h=3)
            gps_gpx = gps_gpx.copy()
            gps_gpx["Timestamp"] = gps_gpx["Timestamp"] + pd.Timedelta(hours=h)
        gps = normalize_gps_1hz(gps_gpx)
        gps_used = "gpx"

    if gps.empty:
        gps = read_gps_from_db(db_path, run_id=run_id)
        gps_used = "db"

    fb = read_feedback(db_path, run_id=run_id)

    feedback_geo = pd.merge_asof(
        fb.sort_values("Timestamp"),
        gps.sort_values("Timestamp"),
        on="Timestamp", direction="nearest",
        tolerance=pd.Timedelta(milliseconds=cfg.merge_tolerance_ms)
    ).dropna(subset=["latitude","longitude"]).reset_index(drop=True)

    merged = pd.merge_asof(
        df_gsr.sort_values("Timestamp"),
        gps.sort_values("Timestamp"),
        on="Timestamp", direction="nearest",
        tolerance=pd.Timedelta(milliseconds=cfg.merge_tolerance_ms)
    ).dropna(subset=["latitude","longitude"]).reset_index(drop=True)

    # Exports
    label = f"{cfg.output_prefix}_run{_safe_slug(run_id)}"
    out_csv_dir = os.path.join(out_dir, "csv")
    out_plot_dir = os.path.join(out_dir, "plots")
    out_kml_dir = os.path.join(out_dir, "kml")
    os.makedirs(out_csv_dir, exist_ok=True)

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_all = os.path.join(out_csv_dir, f"output_GSR_GPS_{label}_{time_str}.csv")
    out_peaks = os.path.join(out_csv_dir, f"output_GSR_GPS_SCRonly_{label}_{time_str}.csv")
    out_events = os.path.join(out_csv_dir, f"output_Feedback_to_SCR_{label}_{time_str}.csv")

    merged.to_csv(out_all, index=False)
    merged[merged["SCR_Peak"]==1].to_csv(out_peaks, index=False)
    feedback_geo.to_csv(out_events, index=False)

    kml_paths = export_kml_kmz(merged, out_kml_dir, label)
    plot_path = plot_summary(df_gsr, merged, feedback_geo, out_plot_dir, title=f"GSR/GPS – {label} (gps={gps_used})")

    zip_path = os.path.join(out_dir, f"outputs_{label}_{time_str}.zip")
    zip_outputs([out_all, out_peaks, out_events, plot_path, kml_paths["kml"], kml_paths["kmz"]], zip_path)

    return {
        "gps_used": gps_used,
        "outputs": {
            "csv_all": out_all,
            "csv_peaks": out_peaks,
            "csv_events": out_events,
            "plot_png": plot_path,
            "kml": kml_paths["kml"],
            "kmz": kml_paths["kmz"],
            "zip": zip_path,
        }
    }


def detect_shimmer_columns(csv_path: str) -> tuple[list[str], list[str]]:
    """
    Returns (timestamp_candidates, conductance_candidates)
    """
    try:
        df = pd.read_csv(csv_path, sep="\\t", header=1, skiprows=[2], nrows=5)
    except Exception:
        df = pd.read_csv(csv_path, nrows=5)

    cols = list(df.columns)
    ts_cands = try_pick_by_substring(cols, ["timestamp"])
    # prefer unix/cal
    ts_cands = sorted(ts_cands, key=lambda c: (0 if "unix" in c.lower() else 1, 0 if "cal" in c.lower() else 1, len(c)))
    gsr_cands = try_pick_by_substring(cols, ["gsr"]) or try_pick_by_substring(cols, ["conductance"]) or try_pick_by_substring(cols, ["skin", "conduct"])
    gsr_cands = sorted(gsr_cands, key=lambda c: (0 if "conduct" in c.lower() else 1, 0 if "cal" in c.lower() else 1, len(c)))
    return ts_cands, gsr_cands
