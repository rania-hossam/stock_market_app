# Import the necessary modules
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Set the start date and today's date
START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Set the title of the app
st.title('Stock Forecast App')

# Define a list of stock symbols
stocks = ['GOOG', 'AAPL', 'MSFT', 'GME', 'AKBNK.IS']
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# Slider to choose the number of months for prediction
n_months = st.slider('Months of prediction:', 1, 5)
period = n_months * 30  # Convert months to days

@st.cache_data(ttl=3600)  # Adjust the ttl as needed (in seconds)
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Display a loading message while data is being loaded
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Display the raw data
st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
    fig.update_layout(title_text='Time Series Data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Prepare data for Prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Initialize and fit the Prophet model
m = Prophet()
m.fit(df_train)

# Create a future dataframe for forecasting
future = m.make_future_dataframe(periods=period)

# Make predictions
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_months} months')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
