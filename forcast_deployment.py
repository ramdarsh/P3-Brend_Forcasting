import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pickle
import arch
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import gdown

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
    background-color: transparent! important;
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

arima_url = "https://drive.google.com/uc?id=11R20EzImKolCVqT2OAVjshurCN66ppEk"
garch_url = "https://drive.google.com/uc?id=1GFIjKkfN_c_ddqEvAbhbIOJ_NLFaduGl"

gdown.download(arima_url, 'modelARIMA.pkl', quiet=False)
gdown.download(garch_url, 'resultGarch.pkl', quiet=False)

with open('modelARIMA.pkl', 'rb') as f:
    loadedARIMA = pickle.load(f)

with open('resultGarch.pkl', 'rb') as f:
    loadedGARCH = pickle.load(f)

# --- Forecast Function ---
def forecast(HORIZON):
    forecastARIMA = loadedARIMA.forecast(HORIZON)
    forecastGARCH = loadedGARCH.forecast(horizon=HORIZON)

    varGARCH = forecastGARCH.variance.values[-1, :]
    varGARCH = scalerMM.fit_transform(varGARCH.reshape(-1, 1)).reshape(-1)

    brentForecast = np.array(forecastARIMA) + varGARCH
    return brentForecast

# --- Streamlit UI ---
st.text("Created by: RMS \nPowered by: GARCH + ARIMA")
st.title("BrentPredict")
st.text("Caution: This is not financial advice!")

user_input = st.number_input("Enter HORIZON (Forecasting periods)", 0)
HORIZON = int(user_input)

if HORIZON > 0:
    brentForecast = forecast(HORIZON)
    st.markdown(
        '<p style="font-family:Helvetica; color:#FFD700; font-size:20px;">HOVER OVER THE CHART TO SEE PRICES.</p>',
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
    st.plotly_chart(fig)





