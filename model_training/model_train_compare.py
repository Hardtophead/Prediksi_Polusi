
"""
Multi-Model Training & Comparison (Metrics: MAE, RMSE, MAPE(%), R^2, Accuracy(%))
---------------------------------------------------------------------------------
- Time-aware split (no leakage)
- Scalers fit on TRAIN only
- Multi-step multivariate forecasting: LOOK_BACK -> LOOK_AHEAD
- Models: Naive, Dense (MLP), GRU, CNN+LSTM
- Metrics in original units: MAE, RMSE, MAPE(%), R^2, Accuracy(% = 100 - MAPE)
- Saves:
    * results_overall_simple.csv, results_per_feature_simple.csv
    * <model_name>_model.keras
    * <model_name>_history.png
Usage:
    python multimodel_train_compare_simplemetrics.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (GRU, LSTM, Dense, Dropout, RepeatVector,
                                     TimeDistributed, Conv1D, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------- Reproducibility -----------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------- Config -----------------------
DATA_PATH = "pollution_data.csv"           # Change if needed
FEATURES = ["PM2.5", "PM10", "CO", "CO2"] # Target features
LOOK_BACK = 20
LOOK_AHEAD = 20
TEST_SIZE = 0.2

# Training hyperparameters (applied to DL models)
UNITS = 128
DROPOUT = 0.0
LEARNING_RATE = 5e-4
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 8

# ----------------------- Helpers -----------------------
def create_sequences(X_data, y_data, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(X_data) - n_steps_in - n_steps_out + 1):
        X.append(X_data[i:i+n_steps_in])
        y.append(y_data[i+n_steps_in:i+n_steps_in+n_steps_out])
    return np.array(X), np.array(y)

def rmse_value(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

def mape_percent(y_true, y_pred, eps=1e-8):
    # mean absolute percentage error in percent
    denom = np.abs(y_true) + eps
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def accuracy_from_mape(mape_percent_value):
    return 100.0 - mape_percent_value

def per_feature_metrics(y_true, y_pred, feature_names):
    n_samples, n_ahead, n_feat = y_true.shape
    rows = []
    for f in range(n_feat):
        yt = y_true[:, :, f].reshape(-1)
        yp = y_pred[:, :, f].reshape(-1)
        mae = mean_absolute_error(yt, yp)
        rmse = rmse_value(yt, yp)
        mape_p = mape_percent(yt, yp)
        acc = accuracy_from_mape(mape_p)
        r2 = r2_score(yt, yp)
        rows.append({
            "Feature": feature_names[f],
            "MAE": mae,
            "RMSE": rmse,
            "MAPE(%)": mape_p,
            "R2": r2,
            "Accuracy(%)": acc
        })
    return pd.DataFrame(rows)

def overall_metrics(y_true, y_pred):
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    mae = mean_absolute_error(yt, yp)
    rmse = rmse_value(yt, yp)
    mape_p = mape_percent(yt, yp)
    acc = accuracy_from_mape(mape_p)
    r2 = r2_score(yt, yp)
    return {"MAE": mae, "RMSE": rmse, "MAPE(%)": mape_p, "R2": r2, "Accuracy(%)": acc}

def inverse_scale_y(y_scaled_3d, scaler):
    N, H, F = y_scaled_3d.shape
    y2d = y_scaled_3d.reshape((-1, F))
    y2d_inv = scaler.inverse_transform(y2d)
    return y2d_inv.reshape((N, H, F))

def plot_history(history, title, outpath):
    plt.figure()
    plt.plot(history.history.get("loss", []), label="loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

# ----------------------- Load & Preprocess (Time-aware) -----------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"CSV not found at {DATA_PATH}. Please update DATA_PATH.")

df = pd.read_csv(DATA_PATH)

if "created_at" not in df.columns:
    raise ValueError("Column 'created_at' not found.")

df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df = df.dropna(subset=["created_at"])
df["created_at"] = df["created_at"].dt.tz_localize(None)

for col in ["Temperature", "Humidity"]:
    if col in df.columns:
        df = df.drop(columns=[col])

df["created_at"] = df["created_at"].dt.floor("min")
df = df[df["created_at"].dt.minute % 3 == 0]
df = df.drop_duplicates(subset="created_at", keep="first")

df = df.rename(columns={"created_at": "datetime"})
df = df.set_index("datetime").sort_index()

# Optional calendar features (not used as inputs by default)
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["minute"] = df.index.minute

X_cols = FEATURES
Y_cols = FEATURES

data_X = df[X_cols].dropna()
data_y = df[Y_cols].dropna()

n_total = len(data_X)
split_idx = int(n_total * (1.0 - TEST_SIZE))

X_train_raw = data_X.iloc[:split_idx].values
X_test_raw  = data_X.iloc[split_idx:].values

y_train_raw = data_y.iloc[:split_idx].values
y_test_raw  = data_y.iloc[split_idx:].values

# Scalers fit on TRAIN only
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train_raw)
y_train_scaled = scaler_y.fit_transform(y_train_raw)

X_test_scaled = scaler_X.transform(X_test_raw)
y_test_scaled = scaler_y.transform(y_test_raw)

# Build sequences separately
X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, LOOK_BACK, LOOK_AHEAD)
X_test,  y_test  = create_sequences(X_test_scaled,  y_test_scaled,  LOOK_BACK, LOOK_AHEAD)

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("Sequence building yielded empty arrays. Check LOOK_BACK/LOOK_AHEAD vs data length.")

n_feat = X_train.shape[2]

# For inverse-transforming metrics to original units
y_test_inv = inverse_scale_y(y_test, scaler_y)
y_train_inv = inverse_scale_y(y_train, scaler_y)

# ----------------------- Build Models -----------------------
def build_gru():
    model = Sequential([
        GRU(UNITS, activation="tanh", input_shape=(LOOK_BACK, n_feat)),
        Dropout(DROPOUT),
        RepeatVector(LOOK_AHEAD),
        GRU(UNITS, activation="tanh", return_sequences=True),
        Dropout(DROPOUT),
        TimeDistributed(Dense(n_feat, activation="relu")),
    ])
    model.compile(optimizer=Adam(LEARNING_RATE), loss="mse", metrics=["mae"])
    return model

def build_cnn_lstm():
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation="relu", padding="same", input_shape=(LOOK_BACK, n_feat)),
        BatchNormalization(),
        Dropout(DROPOUT),
        LSTM(UNITS, activation="tanh"),
        Dropout(DROPOUT),
        RepeatVector(LOOK_AHEAD),
        LSTM(UNITS, activation="tanh", return_sequences=True),
        Dropout(DROPOUT),
        TimeDistributed(Dense(n_feat, activation="relu")),
    ])
    model.compile(optimizer=Adam(LEARNING_RATE), loss="mse", metrics=["mae"])
    return model

def build_dense_mlp():
    # Flatten input sequence and predict the whole output sequence
    model = Sequential([
        tf.keras.layers.Input(shape=(LOOK_BACK, n_feat)),
        tf.keras.layers.Flatten(),
        Dense(256, activation="relu"),
        Dropout(DROPOUT),
        Dense(256, activation="relu"),
        Dropout(DROPOUT),
        Dense(LOOK_AHEAD * n_feat, activation="linear"),
        tf.keras.layers.Reshape((LOOK_AHEAD, n_feat))
    ])
    model.compile(optimizer=Adam(LEARNING_RATE), loss="mse", metrics=["mae"])
    return model

MODELS = {
    "GRU": build_gru,
    "CNN_LSTM": build_cnn_lstm,
    "DENSE": build_dense_mlp,
}

# ----------------------- Train, Evaluate, Save -----------------------
results_overall = []
results_per_feature = []

# Naive baseline (no training)
last_step_test = X_test[:, -1, :]  # (N, n_feat)
naive_pred_test_scaled = np.repeat(last_step_test[:, None, :], LOOK_AHEAD, axis=1)
naive_pred_inv = inverse_scale_y(naive_pred_test_scaled, scaler_y)

ov = overall_metrics(y_test_inv, naive_pred_inv)
pf = per_feature_metrics(y_test_inv, naive_pred_inv, FEATURES)

results_overall.append({"Model": "NaivePersistence", **ov})
pf.insert(0, "Model", "NaivePersistence")
results_per_feature.append(pf)

for name, builder in MODELS.items():
    print(f"Training {name} ...")
    model = builder()
    es = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE, restore_best_weights=True)
    hist = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=1,
        shuffle=False,  # keep order for time series
    )

    # Predict & inverse scale
    y_pred_test_scaled = model.predict(X_test, verbose=0)
    y_pred_inv = inverse_scale_y(y_pred_test_scaled, scaler_y)

    # Metrics
    ov = overall_metrics(y_test_inv, y_pred_inv)
    pf = per_feature_metrics(y_test_inv, y_pred_inv, FEATURES)

    results_overall.append({"Model": name, **ov})
    pf.insert(0, "Model", name)
    results_per_feature.append(pf)

    # Save artifacts
    model.save(f"{name.lower()}_model.keras")
    plot_history(hist, f"{name} Training History", f"{name.lower()}_history.png")

# ----------------------- Save results -----------------------
df_overall = pd.DataFrame(results_overall).sort_values(by="RMSE")
df_per_feature = pd.concat(results_per_feature, ignore_index=True)

df_overall.to_csv("results_overall_simple.csv", index=False)
df_per_feature.to_csv("results_per_feature_simple.csv", index=False)

print("\n=== Overall Results (sorted by RMSE) ===")
print(df_overall)
print("\nSaved: results_overall_simple.csv, results_per_feature_simple.csv")
