import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pickle
import arch
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px




background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://blog.close.com/content/images/2021/01/sales-forecast-templates.jpg");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)

scalerMM =  MinMaxScaler()

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
        
st.text(" Created by:P317G1:07/12/2023 \n Powered by: GARCH + ARIMA")

st.title("BrentPredict")

st.text("Caution: This is not a financial advice !")

user_input = st.number_input("Enter HORIZON (Forecasting periods)",0)

HORIZON = int(user_input)

if HORIZON:
     
	brentForecast = forecast(HORIZON)
	brentForecast = brentForecast.reshape(-1)
	st.write('HOVER OVER THE CHART TO SEE PRICES...',font="Helvetica 20", color="blue" )
	data = pd.DataFrame({'x': np.arange(1, len(brentForecast) + 1), 'y': brentForecast})



	fig = px.line(data, x='x', y='y', markers=True, line_shape='linear', width=800, height=400,
              labels={'y': 'Brent Crude Price ($)', 'x': 'Horizon'},
              title=f'Brent Crude Price Forecast for {HORIZON} days')

# Display the Plotly Express figure using Streamlit
	st.plotly_chart(fig)