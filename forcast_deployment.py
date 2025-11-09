import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pickle
import arch
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import gdown
import os

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

# --- Safe Google Drive Download Function ---
def safe_gdown_download(url, output):
    """Safely downloads models from Google Drive using gdown (handles large files too)."""
    try:
        if not os.path.exists(output):
            st.info(f"⬇️ Downloading {output} from Google Drive...")
            gdown.download(url, output, quiet=False, fuzzy=True)
            st.success(f"✅ {output} downloaded successfully.")
        else:
            st.success(f"✅ {output} already exists locally.")
    except Exception as e:
        st.error(f"❌ Failed to download {output}. Error: {e}")
        st.stop()

# --- Google Drive URLs (make sure they are public!) ---
arima_url = "https://drive.google.com/uc?export=download&id=11R20EzImKolCVqT2OAVjshurCN66ppEk"
garch_url = "https://drive.google.com/uc?export=download&id=1GFIjKkfN_c_ddqEvAbhbIOJ_NLFaduGl"

# --- Download Models Safely ---
safe_gdown_download(arima_url, "modelARIMA.pkl")
safe_gdown_download(garch_url, "resultGarch.pkl")

# --- Load Models ---
try:
    with open("modelARIMA.pkl", "rb") as f:
        loadedARIMA = pickle.load(f)
    with open("resultGarch.pkl", "rb") as f:
        loadedGARCH = pickle.load(f)
    st.success("📦 Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.stop()

# --- Forecast Function ---
def forecast(HORIZON):
    """Combine ARIMA and GARCH forecasts into a single Brent crude forecast."""
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
HORIZON = st.number_input(
    "🔢 Enter Forecast Horizon (Number of Days):",
    min_value=1,
    max_value=365,
    value=7,
    step=1
)

# --- Generate Forecast ---
if st.button("🔮 Generate Forecast"):
    with st.spinner("Calculating forecast..."):
        try:
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

        except Exception as e:
            st.error(f"❌ Forecasting failed: {e}")
