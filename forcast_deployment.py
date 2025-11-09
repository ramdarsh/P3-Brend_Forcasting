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
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://www.oilandgasmiddleeast.com/cloud/2025/06/24/p9AfpNk9-shutterstock_1722600523-1200x729.jpg");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)

scalerMM =  MinMaxScaler()

# --- Load Models from Google Drive ---

# 🔹 Replace these with your actual Google Drive share links
arima_url = "https://drive.google.com/uc?id=11R20EzImKolCVqT2OAVjshurCN66ppEk"
garch_url = "https://drive.google.com/uc?id=1GFIjKkfN_c_ddqEvAbhbIOJ_NLFaduGl"

# 🔹 Download files locally
gdown.download(arima_url, 'modelARIMA.pkl', quiet=False)
gdown.download(garch_url, 'resultGarch.pkl', quiet=False)

# 🔹 Then load as usual
with open('modelARIMA.pkl', 'rb') as f:
    loadedARIMA = pickle.load(f)

with open('resultGarch.pkl', 'rb') as f:
    loadedGARCH = pickle.load(f)


def forecast(HORIZON):
    
        forecastARIMA = loadedARIMA.forecast(HORIZON)
        forecastGARCH = loadedGARCH.forecast(horizon=HORIZON)
        varGARCH = forecastGARCH.variance.values[-1,:]
        varGARCH = scalerMM.fit_transform(forecastGARCH.variance.values[-1, :].reshape(-1, 1))
        
        brentForecast = forecastARIMA.mean() + varGARCH
        
        return  brentForecast
        
st.text(" Created by:Ramdarsh M S \n Powered by: GARCH + ARIMA")

st.title("BrentPredict")

st.text("Caution: This is not a financial advice !")

user_input = st.number_input("Enter HORIZON (Forecasting periods)",0)

HORIZON = int(user_input)

if HORIZON:
     
	brentForecast = forecast(HORIZON)
	brentForecast = brentForecast.reshape(-1)
	st.markdown(
    '<p style="font-family:Helvetica; color:blue; font-size:20px;">HOVER OVER THE CHART TO SEE PRICES...</p>',
    unsafe_allow_html=True
	)

	data = pd.DataFrame({'x': np.arange(1, len(brentForecast) + 1), 'y': brentForecast})



	fig = px.line(data, x='x', y='y', markers=True, line_shape='linear', width=800, height=400,
              labels={'y': 'Brent Crude Price ($)', 'x': 'Horizon'},
              title=f'Brent Crude Price Forecast for {HORIZON} days')

# Display the Plotly Express figure using Streamlit

	st.plotly_chart(fig)

