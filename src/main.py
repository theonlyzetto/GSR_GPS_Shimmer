from pathlib import Path
from src.data_processing.gsr_gps_pipeline import run_gsr_gps_pipeline

def main():
    # Beispiel: Rohdaten aus data/raw/
    gsr_csv = Path("data/raw/KIT_OST_Session1_Shimmer_C078_Calibrated_SD.csv")
    db_path = Path("data/raw/eDiary2_2025-11-07T11-46-02.910172Z.db")

    if not gsr_csv.exists():
        raise FileNotFoundError(f"GSR CSV nicht gefunden: {gsr_csv}")
    if not db_path.exists():
        raise FileNotFoundError(f"DB nicht gefunden: {db_path}")

    results = run_gsr_gps_pipeline(
        gsr_csv_path=str(gsr_csv),
        db_path=str(db_path),
        out_dir="results"
    )
    print("Ergebnisse:")
    for k, v in results.items():
        print(k, "→", v)

if __name__ == "__main__":
    main()
