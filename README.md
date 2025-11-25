# GSR_GPS_Shimmer

Ein Projekt zur Analyse von GSR- und GPS-Daten (z. B. mit Shimmer-Sensoren).

## Struktur

- **src/** – Hauptcode (Datenverarbeitung, Modelle, Utils)
- **notebooks/** – Jupyter Notebooks für Exploration und Analyse
- **data/** – Datenordner (roh, verarbeitet, extern)
- **results/** – Ergebnisse, Modelle, Plots
- **papers/** – Paper und Literatur
- **docs/** – Projektdokumentation

## Erste Schritte

1. Virtuelle Umgebung anlegen:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
   ```

2. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```

3. Jupyter starten:
   ```bash
   jupyter notebook
   ```
