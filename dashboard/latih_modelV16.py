import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

if not tf.config.list_physical_devices('GPU'):
    sys.exit("CUDA GPU not available. Please run on a machine with a CUDA-enabled GPU.")

df = pd.read_csv("cleaned_data.csv")
data = df.values

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

X = scaled_data[:-1]
y = scaled_data[1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
joblib.dump(rf_model, "best_rf_model.joblib")

gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_mse = mean_squared_error(y_test, gb_pred)
joblib.dump(gb_model, "best_gb_model.joblib")

mlp_model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500)
mlp_model.fit(X_train, y_train)
mlp_pred = mlp_model.predict(X_test)
mlp_mse = mean_squared_error(y_test, mlp_pred)
joblib.dump(mlp_model, "best_mlp_model.joblib")

def create_lstm_sequences(data, seq_len=10):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    return np.array(xs), np.array(ys)

SEQ_LEN = 10
X_seq, y_seq = create_lstm_sequences(scaled_data, SEQ_LEN)
X_seq_train, X_seq_test = X_seq[:-1000], X_seq[-1000:]
y_seq_train, y_seq_test = y_seq[:-1000], y_seq[-1000:]

lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X_seq.shape[2])),
    LSTM(64),
    Dense(y_seq.shape[1])
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_seq_train, y_seq_train, epochs=500, batch_size=64, verbose=0)

lstm_pred = lstm_model.predict(X_seq_test)
lstm_mse = mean_squared_error(y_seq_test, lstm_pred)
lstm_model.save("lstm_pollutant_model.h5")

mse_scores = {
    'Random Forest': rf_mse,
    'Gradient Boosting': gb_mse,
    'MLP Regressor': mlp_mse,
    'LSTM': lstm_mse
}

best_model_name = min(mse_scores, key=mse_scores.get)
best_mse = mse_scores[best_model_name]

print(best_model_name)
print(best_mse)
