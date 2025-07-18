# Import library
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from keras.models import load_model

# URL ThingSpeak API
url = "https://api.thingspeak.com/channels/2990169/feeds.json"
params = {
    "api_key": "LDXFP3LRNTBZCFMU",
    "results": 100  # Ambil 100 data terakhir
}

# Load model LSTM jika tersedia
try:
    lstm_model = load_model("lstm_pollutant_model.h5")
except:
    lstm_model = None

# Konfigurasi metrik polutan
metrics = {
    "pm25": {"display": "PM2.5 (μg/m³)", "field": 3, "color": "blue"},
    "pm10": {"display": "PM10 (μg/m³)", "field": 4, "color": "green"},
    "co":   {"display": "CO (ppm)", "field": 5, "color": "red"},
    "co2":  {"display": "CO₂ (ppm)", "field": 6, "color": "orange"},
}

# Inisialisasi aplikasi Dash
app = Dash(__name__)
app.title = "Dashboard Polutan + Prediksi"

# Layout halaman web Dash
app.layout = html.Div([
    html.H1("Dashboard Polutan Aktual + Prediksi", style={"textAlign": "center"}),

    # Kotak metrik aktual
    html.Div(id="latest-metrics", children=[
        html.Div(id=f"metric-{key}", style={
            "border": "1px solid #ccc", "padding": "20px", "textAlign": "center",
            "borderRadius": "10px", "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
            "backgroundColor": "#f9f9f9"
        }) for key in metrics.keys()
    ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "20px", "margin": "20px"}),

    # Grafik aktual
    html.Div([
        *[dcc.Graph(id=f"graph-{key}", style={"height": "400px"}) for key in metrics.keys()]
    ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}),

    # Judul prediksi
    html.H2("Prediksi 1 Jam Ke Depan", style={"textAlign": "center", "marginTop": "30px"}),

    # Kotak metrik prediksi
    html.Div(id="forecast-metrics", children=[
        html.Div(id=f"forecast-metric-{key}", style={
            "border": "1px solid #ccc", "padding": "20px", "textAlign": "center",
            "borderRadius": "10px", "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
            "backgroundColor": "#fffaf0"
        }) for key in metrics.keys()
    ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "20px", "margin": "20px"}),

    # Grafik prediksi
    html.Div([
        *[dcc.Graph(id=f"graph-{key}-forecast", style={"height": "400px"}) for key in metrics.keys()]
    ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}),

    # Interval update setiap 60 detik
    dcc.Interval(id="interval-component", interval=60 * 1000, n_intervals=0)
])

# Fungsi ambil data dari ThingSpeak
def fetch_data():
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        now_wib = datetime.utcnow() + timedelta(hours=7)
        one_hour_ago = now_wib - timedelta(hours=1)

        timestamps = []
        values = {key: [] for key in metrics.keys()}

        for feed in data["feeds"]:
            utc_dt = datetime.strptime(feed["created_at"], "%Y-%m-%dT%H:%M:%SZ")
            wib_dt = utc_dt + timedelta(hours=7)
            if wib_dt >= one_hour_ago:
                timestamps.append(wib_dt)
                for key, meta in metrics.items():
                    field_val = feed.get(f"field{meta['field']}")
                    values[key].append(float(field_val) if field_val else np.nan)

        dfs = {}
        for key in metrics.keys():
            dfs[key] = pd.Series(values[key], index=pd.to_datetime(timestamps)).dropna()
        return dfs
    else:
        return {}

# Fungsi buat grafik aktual
def generate_figure(series, name, color):
    if series.empty:
        return go.Figure(layout={"title": f"Gagal ambil data: {name}"})

    ewma = series.ewm(span=10, adjust=False).mean()

    trace_actual = go.Scatter(x=series.index, y=series.values, mode="lines+markers",
                              name=f"{name} Aktual", line=dict(color=color))

    trace_pred = go.Scatter(x=series.index, y=ewma, mode="lines+markers",
                            name=f"{name} Prediksi", line=dict(color="magenta"))

    layout = go.Layout(title=name, xaxis={"title": "Waktu", "tickformat": "%H:%M"},
                       yaxis={"title": name}, hovermode="closest",
                       legend=dict(orientation="h"))

    return go.Figure(data=[trace_actual, trace_pred], layout=layout)

# Fungsi prediksi
def generate_forecast(series, name, key):
    if series.empty or len(series) < 10:
        return go.Figure(layout={"title": f"Tidak cukup data untuk prediksi: {name}"})

    try:
        series_resampled = series.resample('3T').mean().dropna()
        recent_data = series_resampled[-10:]

        if lstm_model:
            
            pred_vals = np.linspace(recent_data.values[-1], recent_data.values[-1] + 10, 20)
        else:
            diffs = series_resampled.diff().dropna()
            if len(diffs) < 5:
                diffs = pd.Series(np.random.normal(0, 0.01, 10))

            pattern = diffs[-10:].values.flatten()
            if key == "pm25":
                pattern *= 1.0
            elif key == "pm10":
                pattern *= 1.2
            elif key == "co":
                pattern *= 0.5
            elif key == "co2":
                pattern *= 0.1

            last_val = series_resampled.iloc[-1]
            pred_vals = [last_val]
            for i in range(19):
                delta = pattern[i % len(pattern)]
                noise = np.random.normal(0, abs(delta) * 0.2)
                next_val = max(pred_vals[-1] + delta + noise, 0)
                pred_vals.append(next_val)

        future_times = [series.index[-1] + timedelta(minutes=3 * i) for i in range(20)]

        trace_future = go.Scatter(
            x=future_times,
            y=pred_vals,
            mode="lines+markers",
            name=f"{name} Prediksi 1 Jam",
            line=dict(color="orange")
        )

        layout = go.Layout(
            title=f"Prediksi {name} 1 Jam Ke Depan",
            xaxis={"title": "Waktu", "tickformat": "%H:%M"},
            yaxis={"title": name},
            hovermode="closest"
        )

        return go.Figure(data=[trace_future], layout=layout)

    except Exception as e:
        return go.Figure(layout={"title": f"Error prediksi: {str(e)}"})

# CALLBACK untuk grafik aktual
def create_actual_callback(metric_key):
    @app.callback(
        Output(f"graph-{metric_key}", "figure"),
        Input("interval-component", "n_intervals"),
        prevent_initial_call="initial_duplicate"
    )
    def update_graph(n):
        dfs = fetch_data()
        if metric_key in dfs:
            return generate_figure(dfs[metric_key], metrics[metric_key]["display"], metrics[metric_key]["color"])
        else:
            return go.Figure(layout={"title": f"Gagal ambil data: {metrics[metric_key]['display']}"})

# CALLBACK untuk grafik prediksi
def create_forecast_callback(metric_key):
    @app.callback(
        Output(f"graph-{metric_key}-forecast", "figure"),
        Input("interval-component", "n_intervals"),
        prevent_initial_call="initial_duplicate"
    )
    def update_forecast(n):
        dfs = fetch_data()
        if metric_key in dfs:
            return generate_forecast(dfs[metric_key], metrics[metric_key]["display"], metric_key)
        else:
            return go.Figure(layout={"title": f"Gagal ambil data: {metrics[metric_key]['display']}"})

# CALLBACK untuk metrik aktual
for key in metrics.keys():
    @app.callback(
        Output(f"metric-{key}", "children"),
        Input("interval-component", "n_intervals"),
        prevent_initial_call="initial_duplicate"
    )
    def update_metric(n, key=key):
        dfs = fetch_data()
        if key in dfs and not dfs[key].empty:
            latest_val = round(dfs[key].iloc[-1], 2)
            return html.Div([
                html.H4(metrics[key]["display"]),
                html.H2(f"{latest_val}", style={"color": metrics[key]["color"], "fontSize": "36px"})
            ])
        else:
            return html.Div([
                html.H4(metrics[key]["display"]),
                html.H2("N/A", style={"color": "gray"})
            ])

# CALLBACK untuk metrik prediksi
for key in metrics.keys():
    @app.callback(
        Output(f"forecast-metric-{key}", "children"),
        Input("interval-component", "n_intervals"),
        prevent_initial_call="initial_duplicate"
    )
    def update_forecast_metric(n, key=key):
        dfs = fetch_data()
        if key in dfs and not dfs[key].empty:
            series = dfs[key]
            forecast_fig = generate_forecast(series, metrics[key]["display"], key)
            try:
                forecast_y = forecast_fig["data"][0]["y"]
                if len(forecast_y) > 0:
                    forecast_val = round(forecast_y[-1], 2)
                    return html.Div([
                        html.H4(f"Prediksi {metrics[key]['display']}"),
                        html.H2(f"{forecast_val}", style={"color": "orange", "fontSize": "36px"})
                    ])
            except:
                pass
        return html.Div([
            html.H4(f"Prediksi {metrics[key]['display']}"),
            html.H2("N/A", style={"color": "gray"})
        ])

# Registrasi semua callback grafik
for key in metrics.keys():
    create_actual_callback(key)
    create_forecast_callback(key)

# Jalankan aplikasi
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8001)
