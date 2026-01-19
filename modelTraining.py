import os
import pickle
import joblib
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical

# ================================================================
# Caricamento dati features (baseline ML)
# ================================================================
print(">>> Inizio caricamento dataset features...")
start = time.time()
df_features = pd.read_parquet("ims_1test_features.parquet")
end = time.time()
print(f">>> Dataset features caricato. Shape: {df_features.shape} (tempo: {end-start:.2f}s)")

X = df_features.drop(["bearing_id","label", "filepath", "x", "y"], axis=1).values
y = df_features["label"].values

# ================================================================
# Label Encoding (unico, usato sia per ML che CNN)
# ================================================================
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # y è il tuo array/serie di label testuali

# ADDED Save label encoder for deployment
os.makedirs("models", exist_ok=True)
joblib.dump(le, "models/label_encoder.pkl")

# Ora y_encoded contiene valori da 0 a 6
print("Mapping labels:", list(zip(le.classes_, range(len(le.classes_)))))

# ================================================================
# Train/test split per ML
# ================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ================================================================
# Scaler
# ================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")

# ================================================================
# RandomForest con tuning
# ================================================================
print("\n=== Training RandomForest con GridSearch ===")
rf_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=3,
    n_jobs=-1,  # OK su RF
    verbose=1
)
rf_grid.fit(X_train, y_train)
y_pred_rf = rf_grid.predict(X_test)

print("Migliori parametri RF:", rf_grid.best_params_)
print("RandomForest Results")
print(classification_report(y_test, y_pred_rf))

with open("models/rf_model.pkl", "wb") as f:
    pickle.dump(rf_grid.best_estimator_, f)

# ================================================================
# XGBoost con tuning
# ================================================================
print("\n=== Training XGBoost con GridSearch ===")
xgb_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 6],
    "learning_rate": [0.05, 0.1]
}

xgb_grid = GridSearchCV(
    XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
        n_jobs=1   # più sicuro su M1/M2
    ),
    xgb_param_grid,
    cv=3,
    n_jobs=-1,
    verbose=1
)

xgb_grid.fit(X_train, y_train)
y_pred_xgb = xgb_grid.predict(X_test)

print("Migliori parametri XGB:", xgb_grid.best_params_)
print("XGBoost Results")
print(classification_report(y_test, y_pred_xgb))

with open("models/xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_grid.best_estimator_, f)

# ================================================================
# CNN 1D sui segnali raw accelerometrici bi-assiali
# ================================================================
print("\n=== Preparazione CNN 1D sui segnali raw ===")
df_raw = pd.read_parquet("ims_1test_raw.parquet")

X_raw = df_raw[["x", "y"]].values
y_raw = df_raw["label"].values

# --------- Label encoding riutilizzando logiche salvate ---------
le = joblib.load("models/label_encoder.pkl")
y_proc = le.transform(y_raw)  # encoding coerente con ML

# --------- Conversione segnali in tensori numpy ---------
print("Converting raw signals in tensori numpy...")
X_proc = []
for row in X_raw:
    x_sig = np.array(row[0])
    y_sig = np.array(row[1])
    stacked = np.stack([x_sig, y_sig], axis=-1)  # (L, 2)
    X_proc.append(stacked)

X_proc = np.array(X_proc)
y_proc = np.array(y_proc)  # <-- etichette numeriche

# --------- Normalizzazione per asse ---------
global_mean_x = np.mean(X_proc[:, :, 0])
global_std_x = np.std(X_proc[:, :, 0])
global_mean_y = np.mean(X_proc[:, :, 1])
global_std_y = np.std(X_proc[:, :, 1])

X_proc[:, :, 0] = (X_proc[:, :, 0] - global_mean_x) / global_std_x
X_proc[:, :, 1] = (X_proc[:, :, 1] - global_mean_y) / global_std_y

# Salvataggio valori globali (per deployment / inference)
scaling_params = {
    "global_mean_x": float(global_mean_x),
    "global_std_x": float(global_std_x),
    "global_mean_y": float(global_mean_y),
    "global_std_y": float(global_std_y),
}
with open("models/raw_scaling_params.pkl", "wb") as f:
    pickle.dump(scaling_params, f)

print("Scaling parameters salvati in models/raw_scaling_params.pkl")

# --------- Split train/test ---------
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_proc, y_proc, test_size=0.2, random_state=42, stratify=y_proc
)

# --------- One-hot encoding labels per categorical_crossentropy ---------
num_classes = len(np.unique(y_proc))
y_train_cat = to_categorical(y_train_raw, num_classes)
y_test_cat = to_categorical(y_test_raw, num_classes)

# --------- Definizione CNN 1D ---------
print("\n=== Definizione CNN 1D ===")
cnn_model = models.Sequential([
    layers.Conv1D(32, 7, activation="relu", input_shape=(X_train_raw.shape[1], X_train_raw.shape[2])),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 5, activation="relu"),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

cnn_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

cnn_model.summary()

# --------- Callbacks ---------
early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3) # min_lr=1e-5

# --------- Training ---------
print("\n=== Training CNN 1D ===")
history = cnn_model.fit(
    X_train_raw, y_train_cat,
    epochs=30, batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=2
)

# --------- Valutazione ---------
print("\n=== Valutazione CNN 1D ===")
test_loss, test_acc = cnn_model.evaluate(X_test_raw, y_test_cat, verbose=2)
print(f"CNN 1D Test accuracy: {test_acc:.4f}")

# --------- Salvataggio modelli ---------
cnn_model.export("models/cnn1d_raw_savedmodel")  # crea una cartella
cnn_model.save("models/cnn1d_raw.keras")

converter = tf.lite.TFLiteConverter.from_saved_model("models/cnn1d_raw_savedmodel")
tflite_model = converter.convert()

with open("models/cnn1d_raw.tflite", "wb") as f:
    f.write(tflite_model)

print("\n=== Training completato. Modelli salvati in models/ ===")
