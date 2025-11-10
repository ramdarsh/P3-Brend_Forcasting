import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pickle
import arch
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import requests
import os

# --- Background ---
background_image = """
<style>
.stApp {
    background: url("https://raw.githubusercontent.com/ramdarsh/P3-Brend_Forcasting/main/p9AfpNk9-shutterstock_1722600523-1200x729.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

/* Remove default dark overlay */
[data-testid="stAppViewContainer"] {
    background-color: transparent !important;
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

[data-testid="stToolbar"] {
    right: 2rem;
}
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)

# --- Load Models ---
scalerMM = MinMaxScaler()

# --- Cached function to download and load models ---
@st.cache_resource(show_spinner=False)
def load_models():
    """Downloads models (if not present) and loads them into memory."""
    def download_from_url(url, dest_path):
        if not os.path.exists(dest_path):
            with st.spinner(f"📦 Downloading {dest_path} ... please wait"):
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(dest_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    st.success(f"✅ Downloaded {dest_path}")
                else:
                    st.error(f"❌ Failed to download {dest_path}. HTTP {response.status_code}")
                    st.stop()
        else:
            st.info(f"📂 Using cached {dest_path}")

    # Hugging Face URLs
    arima_url = "https://huggingface.co/Ramdarsh/brentpredict/resolve/main/modelARIMA.pkl"
    garch_url = "https://huggingface.co/Ramdarsh/brentpredict/resolve/main/resultGarch.pkl"

    download_from_url(arima_url, "modelARIMA.pkl")
    download_from_url(garch_url, "resultGarch.pkl")

    with st.spinner("🔍 Loading ARIMA and GARCH models..."):
        with open("modelARIMA.pkl", "rb") as f:
            loadedARIMA = pickle.load(f)
        with open("resultGarch.pkl", "rb") as f:
            loadedGARCH = pickle.load(f)
        st.success("✅ Models loaded successfully!")

    return loadedARIMA, loadedGARCH


# --- Load models (with caching) ---
loadedARIMA, loadedGARCH = load_models()

# --- Forecast Function ---
def forecast(HORIZON):
    forecastARIMA = loadedARIMA.forecast(HORIZON)
    forecastGARCH = loadedGARCH.forecast(horizon=HORIZON)

    varGARCH = forecastGARCH.variance.values[-1, :]
    varGARCH = scalerMM.fit_transform(varGARCH.reshape(-1, 1)).reshape(-1)

    brentForecast = np.array(forecastARIMA) + varGARCH
    return brentForecast

# --- Streamlit UI ---
st.markdown("<h3 style='color:#FFD700;'>Created by: RMS</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='color:#00FFFF;'>Powered by: GARCH + ARIMA</h4>", unsafe_allow_html=True)

st.title("BrentPredict")
st.markdown("<p style='color:#FFB6C1;'>⚠️ Caution: This is not financial advice!</p>", unsafe_allow_html=True)

user_input = st.number_input("Enter HORIZON (Forecasting periods)", 0)
HORIZON = int(user_input)

if HORIZON > 0:
    with st.spinner("📈 Generating forecast..."):
        brentForecast = forecast(HORIZON)
    st.markdown(
        '<p style="font-family:Helvetica; color:#FFD700; font-size:20px; font-weight:bold;">HOVER OVER THE CHART TO SEE PRICES...</p>',
        unsafe_allow_html=True
    )

    data = pd.DataFrame({
        'Horizon': np.arange(1, len(brentForecast) + 1),
        'Brent Crude Price ($)': brentForecast
    })

    fig = px.line(
        data, x='Horizon', y='Brent Crude Price ($)',
        markers=True, line_shape='linear', width=800, height=400,
        title=f'Brent Crude Price Forecast for {HORIZON} Days'
    )
    fig.update_layout(
        title_font_color='gold',
        font_color='white',
        plot_bgcolor='rgba(0,0,0,0.6)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_size=22
    )
    st.plotly_chart(fig)
