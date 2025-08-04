
import pandas as pd

df = pd.read_csv('pollution_data.csv')
df['created_at'] = pd.to_datetime(df['created_at'])
df['date'] = df['created_at']
df.set_index('date', inplace=True)

display(df.head(10))

df = df.drop(columns=['Temperature', 'Humidity', 'created_at'])
df_hourly = df.resample('3T').mean()
display(df_hourly.head(10))

from sklearn.preprocessing import MinMaxScaler

features = ['PM2.5', 'PM10', 'CO', 'CO2']
data = df_hourly[features].dropna()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

import numpy as np

def create_sequences(data, seq_len=20):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:(i + seq_len)])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

seq_len = 20
X, y = create_sequences(scaled_data, seq_len)

X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(4))  # 4 output: PM2.5, PM10, CO, CO2
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)

# Prediksi semua data (untuk evaluasi)
y_pred_scaled = model.predict(X)

# Kembalikan ke skala asli
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Simpan model
model.save("lstm_model.h5")

# Simpan scaler
import joblib
joblib.dump(scaler, "scaler.save")

from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Load model dan scaler
model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.save")

recent_data = np.array([
    [29.1, 30.2, 1.1, 462],
    [30.5, 32.3, 0.7, 438],
    [31.8, 33.7, 1.8, 492],
    [29.6, 30.9, 0.5, 422],
    [32.4, 34.1, 2.0, 508],
    [30.2, 31.5, 0.9, 448],
    [33.3, 35.0, 1.5, 478],
    [30.8, 32.7, 1.0, 454],
    [31.6, 33.3, 1.3, 470],
    [34.0, 36.2, 2.2, 510],
    [29.7, 30.8, 0.6, 428],
    [32.1, 34.0, 1.9, 500],
    [30.0, 31.1, 0.8, 442],
    [33.5, 35.3, 1.6, 484],
    [31.9, 32.8, 1.2, 466],
    [28.6, 30.0, 0.4, 418],
    [34.3, 36.0, 2.1, 504],
    [30.9, 31.7, 1.1, 460],
    [31.2, 32.5, 1.2, 468],
    [29.9, 31.0, 0.7, 440]
])  # Bentuk: (20 (3 menit x 1 jam), 4 fitur)

recent_scaled = scaler.transform(recent_data)
recent_scaled = recent_scaled.reshape((1, 20, 4))  # (1 sample, 20 time steps, 4 features)

pred_scaled = model.predict(recent_scaled)
predicted = scaler.inverse_transform(pred_scaled)  # Kembali ke skala asli

print("Prediksi kualitas udara 1 jam ke depan:")
print(f"PM2.5: {predicted[0][0]:.2f}")
print(f"PM10 : {predicted[0][1]:.2f}")
print(f"CO   : {predicted[0][2]:.2f}")
print(f"CO2  : {predicted[0][3]:.2f}")

features = ['PM2.5', 'PM10', 'CO', 'CO2']
look_back = 20
n_features = len(features)

# Ambil 20 langkah terakhir
recent_data = df_hourly[features].dropna().tail(look_back).values

def predict_20x3min(model, scaler, initial_seq, look_back=20):
    preds = []
    seq = initial_seq.copy()
    for _ in range(20):  # 20 langkah ke depan
        scaled = scaler.transform(seq)
        input_seq = scaled.reshape((1, 20, n_features))
        pred_scaled = model.predict(input_seq)
        pred = scaler.inverse_transform(pred_scaled)[0]
        preds.append(pred)
        seq = np.vstack([seq[1:], pred])
    return np.array(preds)

# Jalankan prediksi
predictions = predict_20x3min(model, scaler, recent_data)

# Tampilkan hasil prediksi
for i, step in enumerate(predictions):
    print(f"Prediksi T+{(i)*3} menit:")
    print(f"  PM2.5: {step[0]:.2f}")
    print(f"  PM10 : {step[1]:.2f}")
    print(f"  CO   : {step[2]:.2f}")
    print(f"  CO2  : {step[3]:.2f}")
    print("-" * 30)