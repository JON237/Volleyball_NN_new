# Volleyball-Neuronales Netz

Dieses Repository enthält eine TensorFlow/Keras-Implementierung zur Vorhersage des Gewinners eines Volleyballspiels anhand statistischer Differenzen zwischen zwei Teams. Der Datensatz muss als CSV-Datei `vnl_dataset.csv` im Hauptverzeichnis gespeichert werden oder der Pfad entsprechend angepasst werden.

## Verwendung

1. Benötigte Abhängigkeiten installieren:
   ```bash
   pip install pandas scikit-learn tensorflow matplotlib
   ```
2. Trainingsskript ausführen:
   ```bash
   python train_volleyball_nn.py
   ```
   Das Skript lädt den Datensatz, teilt ihn in Trainings- und Testdaten auf, trainiert ein neuronales Netz, gibt Evaluierungsmetriken aus und zeigt optional Trainingsverläufe sowie eine ROC-Kurve an.

## Datensatzformat

Die CSV-Datei muss die folgenden Spalten enthalten:
- `attack_diff`
- `block_diff`
- `serve_diff`
- `opp_error_diff`
- `total_points_diff`
- `dig_diff`
- `reception_diff`
- `set_diff`
- `top_scorer_1_diff`
- `top_scorer_2_diff`
- `label` (1 wenn Team A gewinnt, 0 wenn Team B gewinnt)
- `team` (Name oder Identifier des betrachteten Teams)
- `opponent` (Name oder Identifier des Gegners)
- Optional: `match_date` oder `match_id`, um die zeitliche Reihenfolge der Partien festzulegen

Stelle sicher, dass für diese Spalten keine fehlenden Werte vorhanden sind. Das Skript erweitert den Datensatz automatisch um Sequenzen der letzten `SEQUENCE_LENGTH` (Standard: 5) Spiele pro Team und speichert sie als 3D-Arrays (`X_sequences.npy` und `y_sequences.npy`). Diese Sequenzen dienen anschließend als Eingabe für ein LSTM-basiertes Modell zur Spielvorhersage.
