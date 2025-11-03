import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from joblib import dump
import tensorflow as tf
import matplotlib.pyplot as plt

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

# Datensatz laden
DATA_PATH = 'vnl_dataset.csv'

df = pd.read_csv(DATA_PATH)

# DataFrame in Feature- und Label-Anteile aufteilen
feature_df = df[FEATURES].copy()
labels = df['label'].copy()

# Aufteilung in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(feature_df, labels, test_size=0.2, random_state=42)

# Feature-Skalierung
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Skalierer speichern
SCALER_PATH = 'feature_scaler.joblib'
dump(scaler, SCALER_PATH)

# Neuronales Netz aufbauen
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(FEATURES),)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modell trainieren
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0)

# Auswertung auf dem Testdatensatz
pred_probs = model.predict(X_test).ravel()
pred_labels = (pred_probs > 0.5).astype(int)

accuracy = accuracy_score(y_test, pred_labels)
precision = precision_score(y_test, pred_labels)
recall = recall_score(y_test, pred_labels)
roc_auc = roc_auc_score(y_test, pred_probs)

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

