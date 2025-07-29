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

Stelle sicher, dass für diese Spalten keine fehlenden Werte vorhanden sind.
