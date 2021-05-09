import streamlit as st
from datetime import date
from yahoo_fin.stock_info import *
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
from PIL import Image

START = "2011-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.write("""
# Stock Market Analysis Portal!!""")

image = Image.open("C:/Users/asus/Desktop/3.jpg")
st.image(image, use_column_width = True)

stocks = tickers_sp500()
st.sidebar.header("User Input")
selected_stock = st.sidebar.selectbox('Select dataset for prediction', stocks)
# def get_input():
#     start_date = st.sidebar.text_imput("Start Date", "START")
#     end_date = st.sidebar.text_imput("End Date", "TODAY")
n_years = st.sidebar.slider('Years of prediction:', 1, 10)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    #fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)


st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)