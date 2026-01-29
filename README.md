📍 GSR + GPS Processing Pipeline (Shimmer + eDiary + GPX) | Stand 20260128
Dieses Projekt verarbeitet EDA/GSR-Daten vom Shimmer, Events/Runs aus einer eDiary-SQLite-DB und GPS-Tracks aus GPX-Dateien zu einer konsistenten Zeit- und Ortsdarstellung inkl. Karten- und QGIS-Export.
Grundprinzip (wichtig)
Die Datenquellen haben klar getrennte Rollen:
•	eDiary-DB
o	Runs (Start/Ende)
o	Teilnehmer-Metadaten
o	Feedback / Events (Feeling, Notes)
o	Zeitliche Referenz (WANN ist etwas passiert?)
•	GPX (primäre Positionsquelle)
o	Räumliche Bewegung (WO war die Person?)
o	Wird priorisiert, da Android-DB-GPS im Hintergrund einfrieren kann
•	DB-GPS
o	Nur Fallback, falls GPX Lücken hat
•	Shimmer EDA
o	Physiologische Messung (GSR / Conductance)
________________________________________
🧠 Sampling-Strategie
•	EDA-Verarbeitung:
o	Analyse auf Original-Samplingrate (z. B. 4 Hz, 32 Hz, 128 Hz)
o	Glättung, SCL, SCR-Peak-Detection erfolgen ohne Downsampling
•	GPS / Mapping:
o	Separate 1 Hz-View für:
	GPS-Merge
	Kartenplot
	QGIS-Export
o	Keine Informationsverluste in der EDA-Analyse
________________________________________
🗺️ GPS-Handling (GPX-first)
1.	DB-GPS laden und auf 1 Hz resamplen (nearest)
2.	GPX laden
o	Robuster Parser für GPX 1.0 und 1.1 (Namespace-agnostisch)
o	Trackpoints (<trkpt>) werden verwendet
3.	Automatische Zeitoffset-Erkennung
o	GPX-Zeit (UTC) wird automatisch um −3 … +3 h verschoben
o	Offset mit maximaler Überlappung zum Run-Zeitfenster wird gewählt
4.	Positionsquelle bauen
o	position_df = GPX ⟶ DB-GPS (Fallback)
5.	Run-Cutoff
o	Positionsdaten werden auf das effektive Run-Fenster begrenzt
________________________________________
📊 EDA-Verarbeitung
•	Conductance aus Shimmer-CSV
•	Klassische SCR-Detektion:
o	lokale SCL
o	dynamischer Threshold
o	Peak-Detection
o	Trigger + Latenz
•	Unterstützung für mehrere Samplingraten
•	Separate 4 Hz- und 1 Hz-Views (df_gsr_4hz, df_gsr_1hz)
________________________________________
📍 Event-Geokodierung
•	Events stammen ausschließlich aus der DB
•	Geokodierung erfolgt zeitbasiert:
•	Event.Timestamp  →  position_df.Timestamp  →  (lat, lon)
•	Dadurch:
o	korrekte Event-Position auch bei GPS-Ausfällen
o	saubere Trennung von Semantik und Position
________________________________________
📤 Outputs
•	Zeitreihen-Plots (EDA + Events)
•	Kartenplot (GPX-Track + Peaks + Trigger + Events)
•	CSV-Exports:
o	QGIS-kompatibel
o	SCR-only
o	Feedback-to-SCR Mapping

