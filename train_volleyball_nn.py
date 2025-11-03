import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Tuple

# Datensatz laden
DATA_PATH = 'vnl_dataset.csv'

df = pd.read_csv(DATA_PATH)

# Ausgewählte Feature-Spalten
FEATURES = [
    'attack_diff',
    'block_diff',
    'serve_diff',
    'opp_error_diff',
    'total_points_diff',
    'dig_diff',
    'reception_diff',
    'set_diff',
    'top_scorer_1_diff',
    'top_scorer_2_diff'
]

TEAM_COLUMN = 'team'
OPPONENT_COLUMN = 'opponent'
LABEL_COLUMN = 'label'
SEQUENCE_LENGTH = 5


def prepare_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist and return a clean copy of the dataframe."""
    df_copy = raw_df.copy()

    missing_features = [feature for feature in FEATURES if feature not in df_copy.columns]
    if missing_features:
        raise ValueError(f"Fehlende Feature-Spalten im Datensatz: {missing_features}")

    # Sicherstellen, dass Team-Information vorhanden ist. Falls nicht, Fallback verwenden.
    if TEAM_COLUMN not in df_copy.columns:
        df_copy[TEAM_COLUMN] = 'Team'
    if OPPONENT_COLUMN not in df_copy.columns:
        df_copy[OPPONENT_COLUMN] = 'Opponent'

    # Sortierschlüssel für die zeitliche Reihenfolge bestimmen.
    if 'match_date' in df_copy.columns:
        df_copy['match_sort_key'] = pd.to_datetime(df_copy['match_date'])
    elif 'match_id' in df_copy.columns:
        df_copy['match_sort_key'] = df_copy['match_id']
    else:
        df_copy['match_sort_key'] = np.arange(len(df_copy))

    df_copy[LABEL_COLUMN] = df_copy[LABEL_COLUMN].astype(int)

    return df_copy


def expand_team_perspectives(df_in: pd.DataFrame) -> pd.DataFrame:
    """Create a row for each team perspective (team and opponent) per match."""
    expanded_rows = []
    for _, row in df_in.iterrows():
        features = {feature: float(row[feature]) for feature in FEATURES}
        base_entry = {
            TEAM_COLUMN: row[TEAM_COLUMN],
            OPPONENT_COLUMN: row[OPPONENT_COLUMN],
            LABEL_COLUMN: int(row[LABEL_COLUMN]),
            'match_sort_key': row['match_sort_key'],
        }
        expanded_rows.append({**base_entry, **features})

        # Gegnerische Perspektive hinzufügen (Feature-Differenzen invertieren, Label invertieren)
        opponent_features = {feature: -features[feature] for feature in FEATURES}
        opponent_entry = {
            TEAM_COLUMN: row[OPPONENT_COLUMN],
            OPPONENT_COLUMN: row[TEAM_COLUMN],
            LABEL_COLUMN: 1 - int(row[LABEL_COLUMN]),
            'match_sort_key': row['match_sort_key'],
        }
        expanded_rows.append({**opponent_entry, **opponent_features})

    expanded_df = pd.DataFrame(expanded_rows)
    return expanded_df


def build_team_sequences(df_in: pd.DataFrame, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build sequences of past matches for each team."""
    sequences = []
    labels = []

    for team, group in df_in.groupby(TEAM_COLUMN):
        group_sorted = group.sort_values('match_sort_key')
        if len(group_sorted) <= sequence_length:
            continue

        feature_array = group_sorted[FEATURES].to_numpy(dtype=np.float32)
        label_array = group_sorted[LABEL_COLUMN].to_numpy(dtype=np.float32)

        for idx in range(sequence_length, len(group_sorted)):
            past_slice = feature_array[idx - sequence_length:idx]
            target_label = label_array[idx]
            sequences.append(past_slice)
            labels.append(target_label)

    if not sequences:
        raise ValueError(
            "Nicht genügend Daten, um Sequenzen zu erstellen. Prüfe die Teamdaten oder verringere die Sequenzlänge."
        )

    X_array = np.asarray(sequences, dtype=np.float32)
    y_array = np.asarray(labels, dtype=np.float32)
    return X_array, y_array


prepared_df = prepare_dataframe(df)
expanded_df = expand_team_perspectives(prepared_df)
X_sequences, y_sequences = build_team_sequences(expanded_df, SEQUENCE_LENGTH)

# Sequenzen für spätere Nutzung speichern
np.save('X_sequences.npy', X_sequences)
np.save('y_sequences.npy', y_sequences)

print(
    f"Erzeugte {X_sequences.shape[0]} Sequenzen mit Länge {SEQUENCE_LENGTH} und {len(FEATURES)} Features."
)

# Aufteilung in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(
    X_sequences,
    y_sequences,
    test_size=0.2,
    random_state=42,
    stratify=y_sequences if len(np.unique(y_sequences)) > 1 else None,
)

# Neuronales Netz mit LSTM-Schichten aufbauen
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, len(FEATURES))),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modell trainieren
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    verbose=0,
)

# Auswertung auf dem Testdatensatz
pred_probs = model.predict(X_test, verbose=0).ravel()
pred_labels = (pred_probs > 0.5).astype(int)

accuracy = accuracy_score(y_test, pred_labels)
precision = precision_score(y_test, pred_labels, zero_division=0)
recall = recall_score(y_test, pred_labels, zero_division=0)

try:
    roc_auc = roc_auc_score(y_test, pred_probs)
except ValueError:
    roc_auc = float('nan')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Optionale Visualisierungen
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Trainingsverlauf anzeigen
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # ROC-Kurve
    if not np.isnan(roc_auc):
        fpr, tpr, _ = roc_curve(y_test, pred_probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='best')
        plt.show()
except Exception as e:
    print(f"Visualisierung fehlgeschlagen: {e}")

