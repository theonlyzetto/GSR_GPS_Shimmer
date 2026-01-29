# -*- coding: utf-8 -*-
"""
GSR + GPS + Feedback (v4.9.4-stabil) + GUI-Auswahl fuer RUN & Shimmer
---------------------------------------------------------------------

Basierend auf GSR_GPS_MERGE_v4_9_4.py (stabil).
Ergaenzungen:
  - GUI-Auswahl (tkinter) fuer:
      * GSR-Datei
      * DB-Datei
      * Run aus GPX (Zeitfenster)
      * Shimmer/Spaltenpaar (Timestamp + Conductance)

Der Rest (SCR-Detektion, Lowpass, Merge, CSV-Exports, Plot + Karte) entspricht
v4.9.4, damit deine bisherigen Workflows vergleichbar bleiben.

Hinweis:
  - Falls die DB keine RUN-Tabelle hat oder keine passenden Spalten gefunden
    werden, wird die Run-Auswahl uebersprungen und es wird ohne Run-Cutoff
    gearbeitet.

"""

import os
import sqlite3
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# =========================
# GPS helper: DB + GPX merge
# =========================
def fill_db_gps_with_gpx(gps_db, gpx_df, max_gap_s=10):
    import numpy as np
    import pandas as pd

    gps_db = gps_db.copy().dropna(subset=["Timestamp"]).sort_values("Timestamp")
    gpx_df = gpx_df.copy().dropna(subset=["Timestamp"]).sort_values("Timestamp")

    gps_db = gps_db.drop_duplicates(subset=["Timestamp"], keep="last")
    gpx_df = gpx_df.drop_duplicates(subset=["Timestamp"], keep="last")

    db = gps_db.set_index("Timestamp")[["latitude", "longitude"]].resample("1s").asfreq()
    gpx = gpx_df.set_index("Timestamp")[["latitude", "longitude"]].resample("1s").asfreq()

    idx = pd.date_range(
        min(db.index.min(), gpx.index.min()),
        max(db.index.max(), gpx.index.max()),
        freq="1s"
    )
    db = db.reindex(idx)
    gpx = gpx.reindex(idx)

    # ---- Freeze-Erkennung über Bewegung (Meter) ----
    # alles < ~0.5 m pro Sekunde gilt als Stillstand
    EARTH_M = 111_320  # Meter pro Grad

    dlat = db["latitude"].diff().abs()
    dlon = db["longitude"].diff().abs()

    # grobe Distanz in Metern
    dist_m = np.sqrt((dlat * EARTH_M)**2 + (dlon * EARTH_M)**2)

    freeze = dist_m < 0.5  # < 0.5 m Bewegung = Freeze
    freeze = freeze.rolling(5, min_periods=5).sum() >= 5

    # ---- HIER KEINE EXTRA EINRÜCKUNG ----
    db.loc[freeze, ["latitude", "longitude"]] = np.nan
    print("GPS-Freeze-Sekunden:", int(freeze.sum()))


    db.loc[freeze, ["latitude", "longitude"]] = np.nan
    print("GPS-Freeze-Sekunden:", int(freeze.sum()))

    out = db.combine_first(gpx)
    out["latitude"] = out["latitude"].interpolate(limit=max_gap_s)
    out["longitude"] = out["longitude"].interpolate(limit=max_gap_s)

    return out.reset_index().rename(columns={"index": "Timestamp"})


import contextily as ctx
from scipy.signal import find_peaks, butter, filtfilt

import tkinter as tk
from tkinter import filedialog, messagebox


# ======================= Settings (v4.9.4) =========================
GPS_TABLE_NAME = "locationEventData"
FEEDBACK_TABLE = "feedbackEventData"
RUN_TABLE = "run"

scr_offset = 0.075
min_peak_distance_s = 2
stimulus_window_s = 10
sampling_rate_hz = 4
merge_tolerance_ms = 4000

# Candidate patterns for Shimmer columns
TS_CANDIDATE_SUBSTRINGS = ["Timestamp_Unix", "Timestamp", "Unix"]
GSR_CANDIDATE_SUBSTRINGS = ["GSR_Skin_Conductance", "Skin_Conductance", "Conductance", "EDA"]


# ===================== Helpers =============================

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, data)


def lowpass(series, cutoff_hz, fs, order=4):
    x = series.interpolate().bfill().ffill().to_numpy()
    y = butter_lowpass_filter(x, cutoff_hz, fs, order)
    return pd.Series(y, index=series.index)


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def try_pick_by_substring(columns, substrings):
    cols = list(columns)
    for sub in substrings:
        for c in cols:
            if sub in c:
                return c
    return None


def choose_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    try:
        root.destroy()
    except Exception:
        pass
    return path


def choose_from_list(title, items, label):
    """Return the selected item (the underlying dict), or None."""
    if not items:
        return None

    root = tk.Tk()
    root.title(title)
    root.geometry("900x500")

    selected = {"value": None}

    lbl = tk.Label(root, text=label, anchor="w", justify="left")
    lbl.pack(fill="x", padx=10, pady=(10, 6))

    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=10, pady=6)

    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")

    listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set, selectmode=tk.SINGLE, width=160)
    for i, it in enumerate(items):
        listbox.insert(tk.END, it["display"])
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=listbox.yview)

    def on_ok():
        sel = listbox.curselection()
        if not sel:
            messagebox.showwarning("Keine Auswahl", "Bitte einen Eintrag auswaehlen.")
            return
        selected["value"] = items[int(sel[0])]
        root.destroy()

    def on_cancel():
        selected["value"] = None
        root.destroy()

    btn_frame = tk.Frame(root)
    btn_frame.pack(fill="x", padx=10, pady=(6, 10))

    tk.Button(btn_frame, text="OK", command=on_ok, width=12).pack(side="right", padx=(6, 0))
    tk.Button(btn_frame, text="Abbrechen", command=on_cancel, width=12).pack(side="right")

    root.mainloop()
    return selected["value"]


def build_shimmer_pairs(df_columns):
    """Detect plausible (timestamp_col, conductance_col) pairs from column names."""
    cols = list(df_columns)

    # Heuristic 1: Shimmer_<ID>_Timestamp... + Shimmer_<ID>_GSR...
    pairs = []
    for c in cols:
        if "Shimmer_" in c and "Timestamp" in c:
            prefix = c.split("Timestamp")[0]  # e.g. Shimmer_C078_
            # Try find conductance with same prefix
            for g in cols:
                if g.startswith(prefix) and any(s in g for s in GSR_CANDIDATE_SUBSTRINGS) and ("Range" not in g) and ("Resistance" not in g):
                    pairs.append((c, g))

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for ts, gsr in pairs:
        key = (ts, gsr)
        if key not in seen:
            seen.add(key)
            uniq.append((ts, gsr))

    # Fallback: pick any timestamp-like and any gsr-like
    if not uniq:
        ts = try_pick_by_substring(cols, TS_CANDIDATE_SUBSTRINGS)
        gsr = try_pick_by_substring(cols, GSR_CANDIDATE_SUBSTRINGS)
        if ts and gsr:
            uniq.append((ts, gsr))

    return uniq


def safe_to_dt(series):
    """Robust datetime parsing (handles ISO strings, unix seconds, unix ms)."""
    s = series

    # If string-like, try direct parse first
    if getattr(s, "dtype", None) == object:
        dt = pd.to_datetime(s, errors="coerce")
        if dt.notna().mean() > 0.9:
            return dt

    x = pd.to_numeric(s, errors="coerce")
    x_non = x.dropna()
    if x_non.empty:
        return pd.to_datetime(s, errors="coerce")

    med = float(np.median(x_non))
    if med > 1e11:  # unix ms
        return pd.to_datetime(s, unit="ms", errors="coerce")
    if med > 1e8:   # unix s
        return pd.to_datetime(s, unit="s", errors="coerce")

    return pd.to_datetime(s, errors="coerce")

def read_gpx_file(path):
    import xml.etree.ElementTree as ET
    import pandas as pd

    tree = ET.parse(path)
    root = tree.getroot()

    # Namespace automatisch erkennen (GPX 1.0 / 1.1)
    if root.tag.startswith("{"):
        ns_uri = root.tag.split("}")[0].strip("{")
        ns = {"gpx": ns_uri}
        trkpt_path = ".//gpx:trkpt"
        time_path = "gpx:time"
    else:
        ns = {}
        trkpt_path = ".//trkpt"
        time_path = "time"

    rows = []
    for trkpt in root.findall(trkpt_path, ns):
        lat = trkpt.attrib.get("lat")
        lon = trkpt.attrib.get("lon")
        if lat is None or lon is None:
            continue

        time_el = trkpt.find(time_path, ns)
        if time_el is None or not time_el.text:
            continue

        t = pd.to_datetime(time_el.text, errors="coerce")
        if pd.isna(t):
            continue

        rows.append((t, float(lat), float(lon)))

    df = pd.DataFrame(rows, columns=["Timestamp", "latitude", "longitude"])

    if df.empty:
        return df

    # TZ entfernen (GPX ist UTC "Z")
    if getattr(df["Timestamp"].dt, "tz", None) is not None:
        df["Timestamp"] = df["Timestamp"].dt.tz_convert(None)

    return df.sort_values("Timestamp").reset_index(drop=True)


def fill_db_gps_with_gpx(gps_db, gpx_df, max_gap_s=10):
    """Prefer DB GPS, fill missing with GPX, then interpolate only short gaps (<= max_gap_s)."""
    gps_db = gps_db.copy().dropna(subset=["Timestamp"]).sort_values("Timestamp")
    gpx_df = gpx_df.copy().dropna(subset=["Timestamp"]).sort_values("Timestamp")

    # Dedupe timestamps
    gps_db = gps_db.drop_duplicates(subset=["Timestamp"], keep="last")
    gpx_df = gpx_df.drop_duplicates(subset=["Timestamp"], keep="last")

    # 1 Hz grids (no averaging)
    db_1hz = gps_db.set_index("Timestamp")[["latitude", "longitude"]].resample("1S").asfreq()
    gpx_1hz = gpx_df.set_index("Timestamp")[["latitude", "longitude"]].resample("1S").asfreq()

    t0 = min(db_1hz.index.min(), gpx_1hz.index.min())
    t1 = max(db_1hz.index.max(), gpx_1hz.index.max())
    idx = pd.date_range(t0, t1, freq="1S")

    db_1hz = db_1hz.reindex(idx)
    gpx_1hz = gpx_1hz.reindex(idx)

    # Prefer DB, fill with GPX
    filled = db_1hz.combine_first(gpx_1hz)

    # Only short gaps
    filled["latitude"] = filled["latitude"].interpolate(limit=max_gap_s, limit_direction="both")
    filled["longitude"] = filled["longitude"].interpolate(limit=max_gap_s, limit_direction="both")

    out = filled.reset_index().rename(columns={"index": "Timestamp"})
    return out


# ==================== 0) GUI Auswahl ===============================
print("Datei-Auswahl...")

input_gsr_filename = choose_file(
    title="Waehle GSR-Datei (Shimmer CSV)",
    filetypes=[("CSV", "*.csv"), ("Alle Dateien", "*")],
)
if not input_gsr_filename:
    raise SystemExit("Keine GSR-Datei gewaehlt.")

gps_db_path = choose_file(
    title="Waehle DB-Datei (SQLite / eDiary)",
    filetypes=[("SQLite DB", "*.db"), ("Alle Dateien", "*")],
)
if not gps_db_path:
    raise SystemExit("Keine DB-Datei gewaehlt.")

print(f"GSR: {os.path.basename(input_gsr_filename)}")
print(f"DB : {os.path.basename(gps_db_path)}")


# ==================== 1) GSR Laden + Shimmer Auswahl ================
print("Lade GSR...")

# v4.9.4 Import-Settings
raw_gsr = pd.read_csv(input_gsr_filename, sep="	", header=0, skiprows=[0,2], engine="python")
raw_gsr = raw_gsr.drop(columns=[c for c in raw_gsr.columns if str(c).startswith("Unnamed")], errors="ignore")
# Einige Shimmer-Exports enthalten eine Units-Zeile; entferne sie robust
if raw_gsr.shape[0] > 0 and str(raw_gsr.iloc[0,0]).strip().lower() in {"ms","s","sec","seconds"}:
    raw_gsr = raw_gsr.iloc[1:].reset_index(drop=True)

pairs = build_shimmer_pairs(raw_gsr.columns)
if not pairs:
    raise SystemExit("Keine passenden Timestamp/GSR-Spalten gefunden.")

pair_items = []
for ts_col, gsr_col in pairs:
    pair_items.append({
        "ts": ts_col,
        "gsr": gsr_col,
        "display": f"Timestamp: {ts_col}    |    GSR: {gsr_col}",
    })

pair_sel = choose_from_list(
    title="Shimmer-Auswahl",
    items=pair_items,
    label="Welche Timestamp/GSR-Spalten sollen genutzt werden?",
)
if pair_sel is None:
    raise SystemExit("Keine Shimmer-Spalten gewaehlt.")

timestamp_col = pair_sel["ts"]
conductance_col = pair_sel["gsr"]
print(f"Verwende Timestamp-Spalte: {timestamp_col}")
print(f"Verwende Conductance-Spalte: {conductance_col}")

# Schutz: Range/Resistance sind keine geeigneten Input-Signale fuer SCR-Detektion
if ("Range" in conductance_col) or ("Resistance" in conductance_col):
    raise SystemExit(f"Ausgewaehlte Spalte wirkt nicht wie Conductance (Range/Resistance): {conductance_col}")

# Parse
raw_gsr["Timestamp"] = safe_to_dt(raw_gsr[timestamp_col])
df_gsr = raw_gsr[["Timestamp", conductance_col]].rename(columns={conductance_col: "Conductance"})
df_gsr["Conductance"] = (
    df_gsr["Conductance"].astype(str).str.replace(",", ".", regex=False)
)
df_gsr["Conductance"] = pd.to_numeric(df_gsr["Conductance"], errors="coerce")
df_gsr = df_gsr.dropna(subset=["Timestamp", "Conductance"]).set_index("Timestamp").sort_index()
df_gsr = df_gsr.resample("1s").mean().interpolate().reset_index()


# ==================== 2) DB: Run Auswahl ============================
print("Lade DB-Tabellen...")
conn = sqlite3.connect(gps_db_path)

# Runs laden (optional)
runs = None
run_sel = None
try:
    runs = pd.read_sql_query(f"SELECT * FROM {RUN_TABLE}", conn)
except Exception:
    runs = None

if runs is not None and not runs.empty:
    col_id = pick_col(runs, ["id", "runId", "run_id"])
    col_start = pick_col(runs, ["start", "startTime", "start_time", "startTimestamp", "start_timestamp"])
    col_end = pick_col(runs, ["end", "endTime", "end_time", "endTimestamp", "end_timestamp"])

    if col_id and col_start and col_end:
        runs["start_dt"] = safe_to_dt(runs[col_start])
        runs["end_dt"] = safe_to_dt(runs[col_end])
        runs = runs.dropna(subset=["start_dt", "end_dt"])

        # Optional extra columns for display
        extra_cols = []
        for c in ["study", "studyName", "name", "participant", "gender", "sex"]:
            if c in runs.columns:
                extra_cols.append(c)

        run_items = []
        for _, r in runs.iterrows():
            rid = r[col_id]
            sdt = r["start_dt"]
            edt = r["end_dt"]
            extra_txt = " | ".join([f"{c}={r.get(c)}" for c in extra_cols])
            disp = f"run_id={rid} | {sdt} -> {edt}"
            if extra_txt:
                disp += f" | {extra_txt}"
            run_items.append({"run_id": rid, "start": sdt, "end": edt, "display": disp})

        # Vorauswahl: best overlap mit GSR
        gsr_min, gsr_max = df_gsr["Timestamp"].min(), df_gsr["Timestamp"].max()
        def overlap_s(a_start, a_end, b_start, b_end):
            start = max(a_start, b_start)
            end = min(a_end, b_end)
            return max(0.0, (end - start).total_seconds())
        for it in run_items:
            it["overlap_s"] = overlap_s(gsr_min, gsr_max, it["start"], it["end"])
        run_items = sorted(run_items, key=lambda x: x["overlap_s"], reverse=True)

        run_sel = choose_from_list(
            title="Run-Auswahl",
            items=run_items,
            label=(
                "Welchen Run moechtest du auswerten?\n"
                "Hinweis: Liste ist nach groesster Ueberlappung mit der GSR-Datei sortiert."
            ),
        )

        if run_sel is None:
            print("Run-Auswahl abgebrochen; arbeite ohne Run-Cutoff.")
        else:
            print(f"Run gewaehlt: run_id={run_sel['run_id']} | {run_sel['start']} -> {run_sel['end']}")
    else:
        print("Hinweis: RUN-Tabelle gefunden, aber Start/End/ID-Spalten nicht eindeutig; arbeite ohne Run-Auswahl.")
else:
    print("Hinweis: RUN-Tabelle nicht gefunden oder leer; arbeite ohne Run-Auswahl.")

# Run-Cutoff anwenden (GSR, spaeter GPS/Feedback)
if run_sel:
    run_start = run_sel["start"]
    run_end = run_sel["end"]

    # Falls der Run laenger ist als die Shimmer-Aufnahme, schneide auf Ueberlappung.
    gsr_min = df_gsr["Timestamp"].min()
    gsr_max = df_gsr["Timestamp"].max()
    eff_start = max(run_start, gsr_min)
    eff_end = min(run_end, gsr_max)
    if eff_start != run_start or eff_end != run_end:
        print(f"Hinweis: Run-Zeitfenster wird auf GSR-Ueberlappung gekuerzt: {eff_start} -> {eff_end}")

    df_gsr = df_gsr[(df_gsr["Timestamp"] >= eff_start) & (df_gsr["Timestamp"] <= eff_end)].copy()
    if df_gsr.empty:
        conn.close()
        raise SystemExit("GSR nach Run-Cutoff leer. Run passt vermutlich nicht zur Datei.")

    run_start, run_end = eff_start, eff_end

position_df = position_df[(position_df["Timestamp"] >= run_start) & (position_df["Timestamp"] <= run_end)].copy()
print("Position NaN ratio (IN RUN):", position_df[["latitude","longitude"]].isna().mean().to_dict())
print("Position rows (IN RUN):", len(position_df))


# ==================== 3) SCR Detection (v4.9.4) =====================
print("SCR-Detektion (classic)...")
window_seconds = 5

window_samples = max(1, int(round(window_seconds * sampling_rate_hz)))  # 5s @4Hz => 20 Samples

df_gsr["SCL_local"] = df_gsr["Conductance"].rolling(window=window_samples, min_periods=1).mean()
df_gsr["Threshold"] = df_gsr["SCL_local"] + scr_offset


sig = df_gsr["Conductance"].to_numpy()
thr = df_gsr["Threshold"].to_numpy()

scr_indices, _ = find_peaks(sig, height=thr, distance=min_peak_distance_s * sampling_rate_hz)

df_gsr["SCR_Peak"] = 0
df_gsr.iloc[scr_indices, df_gsr.columns.get_loc("SCR_Peak")] = 1
print(f"SCR-Peaks erkannt: {int(df_gsr['SCR_Peak'].sum())}")

# Trigger + Latenz
df_gsr["Trigger"] = 0
df_gsr["SCR_Latency_s"] = np.nan
last_stim = None

col_ts  = df_gsr.columns.get_loc("Timestamp")
col_c   = df_gsr.columns.get_loc("Conductance")
col_scl = df_gsr.columns.get_loc("SCL_local")
col_tr  = df_gsr.columns.get_loc("Trigger")
col_lat = df_gsr.columns.get_loc("SCR_Latency_s")

for idx in scr_indices:
    peak_t = df_gsr.iloc[idx, col_ts]

    pre = int(idx)
    while pre > 0 and (df_gsr.iloc[pre, col_c] > df_gsr.iloc[pre, col_scl]):
        pre -= 1

    stim_t = df_gsr.iloc[pre, col_ts]

    if (last_stim is None) or ((stim_t - last_stim).total_seconds() > stimulus_window_s):
        df_gsr.iloc[pre, col_tr] = 1
        df_gsr.iloc[idx, col_lat] = (peak_t - stim_t).total_seconds()
        last_stim = stim_t

# ==================== 3b) 1 Hz View für GPS-Merge/Map/Exports =====================
df_gsr_4hz = df_gsr.copy()

df_gsr_1hz = (
    df_gsr_4hz.set_index("Timestamp")
              .resample("1s")
              .agg({
                  "Conductance": "mean",
                  "SCL_local": "mean",
                  "Threshold": "mean",
                  "SCR_Peak": "max",        # wenn in dieser Sekunde ein Peak war -> 1
                  "Trigger": "max",         # wenn Trigger in dieser Sekunde -> 1
                  "SCR_Latency_s": "min"    # optional: min/mean – min ist oft ok
              })
              .reset_index()
)

print("df_gsr_4hz rows:", len(df_gsr_4hz), "| df_gsr_1hz rows:", len(df_gsr_1hz))


# ==================== 4) SCL-Lowpass (v4.9.4) ======================
print("SCL Lowpass...")
df_gsr["SCL_Baseline"] = lowpass(df_gsr["Conductance"], 0.01, sampling_rate_hz)
df_gsr["SCL_Global"] = lowpass(df_gsr["Conductance"], 0.02, sampling_rate_hz)


# ==================== 5) GPS (Position) ===================
print("Lade GPS (DB)...")
gps_db = pd.read_sql_query(f"SELECT * FROM {GPS_TABLE_NAME}", conn)

gps_db["Timestamp"] = safe_to_dt(gps_db["timestamp"] if "timestamp" in gps_db.columns else gps_db.iloc[:, 0])

# Standardspalten
need_cols = ["Timestamp", "latitude", "longitude"]
if not all(c in gps_db.columns for c in need_cols):
    lat_col = pick_col(gps_db, ["latitude", "lat"])
    lon_col = pick_col(gps_db, ["longitude", "lon", "lng"])
    if lat_col and lon_col:
        gps_db = gps_db[["Timestamp", lat_col, lon_col]].rename(columns={lat_col: "latitude", lon_col: "longitude"})
    else:
        conn.close()
        raise SystemExit("GPS-Spalten nicht gefunden (latitude/longitude).")
else:
    gps_db = gps_db[["Timestamp", "latitude", "longitude"]]

gps_db = gps_db.dropna(subset=["Timestamp", "latitude", "longitude"]).sort_values("Timestamp")
gps_db = gps_db.drop_duplicates(subset=["Timestamp"], keep="last")
gps_db = gps_db.set_index("Timestamp").resample("1s").nearest().reset_index()


# =========================
# Backup-GPX Auswahl (Datei)
# =========================

use_gpx = messagebox.askyesno(
    "Backup-GPX",
    "Backup-GPX-Datei nutzen, um GPS-Lücken zu füllen?"
)

gpx_file = None
if use_gpx:
    gpx_file = filedialog.askopenfilename(
        title="Backup-GPX-Datei auswählen",
        filetypes=[("GPX files", "*.gpx")]
    )

    if not gpx_file:
        raise SystemExit(
            "Backup-GPX wurde aktiviert, aber keine GPX-Datei ausgewählt."
        )

# ---- Debug (JETZT ist use_gpx definiert) ----
print("use_gpx =", use_gpx)
print("gpx_file =", gpx_file)

# =========================
# Position bauen (GPX-first + DB fallback)
# =========================
gpx_df = None
if gpx_file:
    print("Lese GPX:", gpx_file)
    gpx_df = read_gpx_file(gpx_file)

    if gpx_df is None or gpx_df.empty:
        print("WARN: GPX ist leer -> Positionsquelle bleibt DB.")
        gpx_df = None
    else:
        # --- GPX Zeitoffset automatisch an Run anpassen (UTC vs Lokal) ---
        best_off = 0
        best_overlap = -1
        for off_h in range(-3, 4):  # -3h .. +3h
            test = gpx_df.copy()
            test["Timestamp"] = test["Timestamp"] + pd.Timedelta(hours=off_h)
            overlap = ((test["Timestamp"] >= run_start) & (test["Timestamp"] <= run_end)).sum()
            if overlap > best_overlap:
                best_overlap = overlap
                best_off = off_h

        if best_off != 0:
            print(f"GPX Offset angewendet: {best_off:+d}h (overlap pts: {best_overlap})")
            gpx_df["Timestamp"] = gpx_df["Timestamp"] + pd.Timedelta(hours=best_off)
        else:
            print(f"GPX Offset: 0h (overlap pts: {best_overlap})")

        # jetzt normalisieren auf 1 Hz
        gpx_df = gpx_df.dropna(subset=["Timestamp", "latitude", "longitude"]).sort_values("Timestamp")
        gpx_df = gpx_df.drop_duplicates(subset=["Timestamp"], keep="last")
        gpx_df = gpx_df.set_index("Timestamp").resample("1s").nearest().reset_index()



# =========================
# Positionstabelle bauen (GPX-first + DB fallback)
# =========================
if gpx_df is not None:
    position_df = (
        gpx_df.set_index("Timestamp")[["latitude", "longitude"]]
              .combine_first(gps_db.set_index("Timestamp")[["latitude", "longitude"]])
              .reset_index()
    )
    print("Position source: GPX-first (+ DB fallback)")
else:
    position_df = gps_db.copy()
    print("Position source: DB-only")

print("Position NaN ratio:", position_df[["latitude", "longitude"]].isna().mean().to_dict())

# kurze Lücken schließen (z.B. wenn einzelne Sekunden fehlen)
position_df["latitude"]  = position_df["latitude"].interpolate(limit=10)
position_df["longitude"] = position_df["longitude"].interpolate(limit=10)


# Run-Cutoff für Position (immer, weil wir im Run arbeiten)
position_df = position_df[(position_df["Timestamp"] >= run_start) & (position_df["Timestamp"] <= run_end)].copy()

# =========================
# Positionstabelle bauen (GPX-first + DB fallback)
# =========================
if gpx_df is not None:
    position_df = (
        gpx_df.set_index("Timestamp")[["latitude", "longitude"]]
              .combine_first(gps_db.set_index("Timestamp")[["latitude", "longitude"]])
              .reset_index()
    )
    print("Position source: GPX-first (+ DB fallback)")
else:
    position_df = gps_db.copy()
    print("Position source: DB-only")

print("Position NaN ratio (BEFORE RUN CUT):", position_df[["latitude", "longitude"]].isna().mean().to_dict())

# Run-Cutoff (erst JETZT, weil position_df jetzt existiert)
position_df = position_df[(position_df["Timestamp"] >= run_start) & (position_df["Timestamp"] <= run_end)].copy()

print("Position rows (IN RUN):", len(position_df))
print("Position NaN ratio (IN RUN):", position_df[["latitude", "longitude"]].isna().mean().to_dict())



# =========================
# Feedback laden + vorbereiten
# =========================
print("Lade Feedback...")
feedback = pd.read_sql_query(f"SELECT * FROM {FEEDBACK_TABLE}", conn)

possible_feeling_cols = ["feeling", "feelingName", "feelingDescription", "emotion", "emotionLabel"]
possible_note_cols = ["note", "noteText", "comment", "description"]

# Timestamp-Spalte robust bestimmen
ts_col = None
cands_time = [c for c in feedback.columns if "time" in c.lower()]
if cands_time:
    ts_col = cands_time[0]
elif "timestamp" in feedback.columns:
    ts_col = "timestamp"
elif "Instant" in feedback.columns:
    ts_col = "Instant"

if ts_col is None:
    raise SystemExit(f"Keine Timestamp-Spalte in Feedback gefunden. Spalten: {list(feedback.columns)}")

if ts_col == "Instant":
    feedback["Timestamp"] = pd.to_datetime(feedback["Instant"], unit="ms", errors="coerce")
else:
    feedback["Timestamp"] = safe_to_dt(feedback[ts_col])

# feeling/note bestimmen
col_feeling = pick_col(feedback, possible_feeling_cols)
col_note = pick_col(feedback, possible_note_cols)

rename_map = {}
if col_feeling: rename_map[col_feeling] = "feeling"
if col_note: rename_map[col_note] = "note"
feedback = feedback.rename(columns=rename_map)

feedback = feedback.dropna(subset=["Timestamp"]).copy()

# Run-Cutoff (effektives Run-Fenster)
feedback = feedback[(feedback["Timestamp"] >= run_start) & (feedback["Timestamp"] <= run_end)].copy()


# Zusätzlich auf GSR-Zeitraum beschränken
feedback = feedback[
    (feedback["Timestamp"] >= df_gsr["Timestamp"].min()) &
    (feedback["Timestamp"] <= df_gsr["Timestamp"].max())
].copy()

# Spalten sicherstellen
if "feeling" not in feedback.columns:
    feedback["feeling"] = ""
if "note" not in feedback.columns:
    feedback["note"] = ""

feedback = feedback[["Timestamp", "feeling", "note"]].reset_index(drop=True)

# Feedback geo (Positionsquelle = position_df)
feedback_geo = pd.merge_asof(
    feedback.sort_values("Timestamp"),
    position_df.sort_values("Timestamp"),
    on="Timestamp",
    direction="nearest",
    tolerance=pd.Timedelta(milliseconds=merge_tolerance_ms),
).dropna(subset=["latitude", "longitude"]).reset_index(drop=True)



# ==================== 6) Merge (v4.9.4) ============================
merged = pd.merge_asof(
    df_gsr.sort_values("Timestamp"),
    position_df.sort_values("Timestamp"),
    on="Timestamp",
    direction="nearest",
    tolerance=pd.Timedelta(milliseconds=merge_tolerance_ms),
).dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

peaks_geo = merged[merged["SCR_Peak"] == 1].copy()
trigs_geo = merged[merged["Trigger"] == 1].copy()


# ==================== 7) Export (v4.9.4) ===========================
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

run_tag = ""
if run_sel:
    run_tag = f"_run{run_sel['run_id']}"

merged_out = f"output_GSR_GPS_QGIS_{time_str}{run_tag}.csv"
peaks_out = f"output_GSR_GPS_SCRonly_{time_str}{run_tag}.csv"
fb_out = f"output_Feedback_to_SCR_{time_str}{run_tag}.csv"

merged.to_csv(merged_out, index=False)
peaks_geo.to_csv(peaks_out, index=False)
feedback_geo.to_csv(fb_out, index=False)
print("CSV exportiert:")
print(f"  {merged_out}")
print(f"  {peaks_out}")
print(f"  {fb_out}")


# ==================== 8) Plot (v4.9.4) =============================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]})
fig.suptitle("EDA Signal + GPS Track (Classic Detection + Unified Symbols)", fontsize=14)

# ---- EDA Signal ----
ax1.plot(df_gsr["Timestamp"], df_gsr["Conductance"], color="steelblue", label="Conductance")
ax1.plot(df_gsr["Timestamp"], df_gsr["SCL_Global"], color="#ffb6c1", label="SCL 0.02 Hz")
ax1.plot(df_gsr["Timestamp"], df_gsr["SCL_Baseline"], color="green", label="Baseline 0.01 Hz")

# Trigger / Peak / Events
ax1.scatter(
    df_gsr.loc[df_gsr["Trigger"] == 1, "Timestamp"],
    df_gsr.loc[df_gsr["Trigger"] == 1, "Conductance"],
    color="orange",
    s=100,
    marker="v",
    label="Trigger",
)
ax1.scatter(
    df_gsr.loc[df_gsr["SCR_Peak"] == 1, "Timestamp"],
    df_gsr.loc[df_gsr["SCR_Peak"] == 1, "Conductance"],
    color="red",
    s=130,
    marker="*",
    label="SCR Peak",
)

for i, fb in feedback.iterrows():
    ax1.axvline(fb["Timestamp"], color="gold", linestyle="--", alpha=0.25)
    ax1.scatter(
        fb["Timestamp"],
        df_gsr["Conductance"].max() * 0.9,
        color="gold",
        s=80,
        label="Event" if i == 0 else "",
    )
    ax1.text(
        fb["Timestamp"],
        df_gsr["Conductance"].max() * 0.93,
        f"{i + 1}: {fb.get('feeling') or ''}\n{fb.get('note') or ''}",
        rotation=90,
        fontsize=8,
        ha="right",
        va="center",
    )

# Linien Trigger -> Peak
for idx in df_gsr.index[df_gsr["SCR_Peak"] == 1]:
    lat = df_gsr.loc[idx, "SCR_Latency_s"]
    if pd.notna(lat):
        t_peak = df_gsr.loc[idx, "Timestamp"]
        t_trig = t_peak - timedelta(seconds=float(lat))
        y_trig = df_gsr.loc[df_gsr["Timestamp"] == t_trig, "Conductance"]
        if len(y_trig) > 0:
            ax1.plot(
                [t_trig, t_peak],
                [y_trig.values[0], df_gsr.loc[idx, "Conductance"]],
                color="gray",
                linestyle="--",
                alpha=0.5,
            )

ax1.legend(fontsize=9)
ax1.set_ylabel("Conductance (uS)")
ax1.grid(alpha=0.3)

# ---- Karte ----
ax2.set_title("GPS Track (Events, Triggers, Peaks)", fontsize=11)
ax2.scatter(merged["longitude"], merged["latitude"], s=8, c="lightgray", label="GPS Pfad")
if not peaks_geo.empty:
    ax2.scatter(peaks_geo["longitude"], peaks_geo["latitude"], color="red", s=70, marker="*", label="SCR Peaks")
if not trigs_geo.empty:
    ax2.scatter(trigs_geo["longitude"], trigs_geo["latitude"], color="orange", s=80, marker="v", label="Trigger")
for i, fb in feedback_geo.iterrows():
    ax2.scatter(fb["longitude"], fb["latitude"], color="gold", s=70, label="Event" if i == 0 else "")
    ax2.text(
        fb["longitude"],
        fb["latitude"],
        str(i + 1),
        fontsize=9,
        color="black",
        ha="center",
        va="center",
        weight="bold",
    )

try:
    ctx.add_basemap(ax2, crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.85)
except Exception as e:
    print(f"OSM Fehler (Basemap): {e}")

ax2.legend(fontsize=9)
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
ax2.grid(alpha=0.3)

plt.tight_layout()
outplot = f"output_GSR_GPS_v4_9_4_GUI_{time_str}{run_tag}.png"
plt.savefig(outplot, dpi=200)
plt.close()

print(f"Plot gespeichert: {outplot}")
print("Fertig: v4.9.4 (stabil) + Run-Auswahl + Shimmer-Auswahl")
