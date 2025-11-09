import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pickle
import arch
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import gdown

# --- Background Image ---
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://www.oilandgasmiddleeast.com/cloud/2025/06/24/p9AfpNk9-shutterstock_1722600523-1200x729.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)

# --- Title and Intro ---
st.title("🛢️ BrentPredict: Brent Crude Price Forecasting")
st.write("**Created by:** Ramdarsh M S  |  **Powered by:** GARCH + ARIMA")
st.warning("⚠️ This app is for educational purposes only. It is **not financial advice.**")

# --- Initialize Scaler ---
scalerMM = MinMaxScaler()

# --- Load Models from Google Drive ---
arima_url = "https://drive.google.com/uc?id=11R20EzImKolCVqT2OAVjshurCN66ppEk"
garch_url = "https://drive.google.com/uc?id=1GFIjKkfN_c_ddqEvAbhbIOJ_NLFaduGl"

# Download the models
gdown.download(arima_url, 'modelARIMA.pkl', quiet=True)
gdown.download(garch_url, 'resultGarch.pkl', quiet=True)

# Load models
with open('modelARIMA.pkl', 'rb') as f:
    loadedARIMA = pickle.load(f)

with open('resultGarch.pkl', 'rb') as f:
    loadedGARCH = pickle.load(f)

# --- Forecast Function ---
def forecast(HORIZON):
    # ARIMA forecast
    forecastARIMA = loadedARIMA.forecast(HORIZON)
    
    # GARCH forecast
    forecastGARCH = loadedGARCH.forecast(horizon=HORIZON)
    varGARCH = forecastGARCH.variance.values[-1, :]
    varGARCH_scaled = scalerMM.fit_transform(varGARCH.reshape(-1, 1)).flatten()
    
    # Combine ARIMA mean + scaled GARCH variance
    brentForecast = forecastARIMA + varGARCH_scaled
    return brentForecast

# --- User Input ---
HORIZON = st.number_input("🔢 Enter Forecast Horizon (Number of Days):", min_value=1, max_value=365, value=7)

# --- Generate Forecast ---
if st.button("🔮 Generate Forecast"):
    with st.spinner("Calculating forecast..."):
        brentForecast = forecast(HORIZON)

        data = pd.DataFrame({
            'Day': np.arange(1, len(brentForecast) + 1),
            'Forecasted Price': brentForecast
        })

        # Plotly Chart
        fig = px.line(
            data,
            x='Day',
            y='Forecasted Price',
            markers=True,
            line_shape='linear',
            width=900,
            height=450,
            labels={'Forecasted Price': 'Brent Crude Price ($)', 'Day': 'Forecast Horizon'},
            title=f'📈 Brent Crude Price Forecast for Next {HORIZON} Days'
        )

        st.markdown(
            '<p style="font-family:Helvetica; color:blue; font-size:20px;">Hover over the chart to view forecasted prices</p>',
            unsafe_allow_html=True
        )

        st.plotly_chart(fig, use_container_width=True)

