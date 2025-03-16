import streamlit as st
import pandas as pd
import numpy as np
import requests
import asyncio
import aiohttp
import plotly.express as px


@st.cache_data
def load_data(file):
    df = pd.read_csv(file, dtype={'temperature': float, 'city': str, 'season': str})
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d', errors='coerce')
    return df


# Получение температуры из OpenWeather (sync)
def get_current_temperature(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    if data["cod"] == 401:
        st.error("Invalid API key. Please see https://openweathermap.org/faq#error401 for more info.")
        return None
    return data["main"]["temp"]


# Получение температуры из OpenWeather (async)
async def get_current_temperature_async(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()
            if data["cod"] == 401:
                st.error("Invalid API key. Please see https://openweathermap.org/faq#error401 for more info.")
                return None
            return data["main"]["temp"]


# Анализ температурных данных
def analyze_temperature(df):
    df = df.copy()
    df['moving_avg'] = df['temperature'].rolling(window=30, min_periods=1).mean()
    df['std_dev'] = df['temperature'].rolling(window=30, min_periods=1).std()
    df['anomaly'] = np.where(
        (df['temperature'] < df['moving_avg'] - 2 * df['std_dev']) |
        (df['temperature'] > df['moving_avg'] + 2 * df['std_dev']), 1, 0)
    return df


# Построение графика температуры
def plot_temperature(df, city):
    fig = px.line(df, x='timestamp', y='temperature', title=f"Temperature in {city}",
                  labels={'timestamp': 'Date', 'temperature': 'Temperature (°C)'})
    fig.add_scatter(x=df['timestamp'], y=df['moving_avg'], mode='lines', name='30-day Moving Average',
                    line=dict(dash='dash', color='green'))
    fig.add_scatter(x=df['timestamp'][df['anomaly'] == 1], y=df['temperature'][df['anomaly'] == 1], mode='markers',
                    name='Anomalies', marker=dict(color='red'))
    st.plotly_chart(fig, use_container_width=True)


# Построение сезонных профилей
def plot_seasonal_profiles(df):
    seasonal_profiles = df.groupby('season')['temperature'].agg(['mean', 'std']).reset_index()
    fig = px.bar(seasonal_profiles, x='season', y='mean', error_y='std',
                 labels={'season': 'Season', 'mean': 'Average Temperature (°C)'},
                 title="Seasonal Temperature Profiles")
    st.plotly_chart(fig, use_container_width=True)


# Интерфейс Streamlit
st.title('Climate Change and Temperature Monitoring')
st.sidebar.header('Settings')

# Загрузка CSV
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = load_data(uploaded_file)

    city = st.sidebar.selectbox("Choose City", df['city'].unique())

    st.subheader(f"Temperature Analysis for {city}")

    # Анализ данных по выбранному городу
    city_df = df[df['city'] == city].copy()
    analyzed_df = analyze_temperature(city_df)

    # Графики
    plot_temperature(analyzed_df, city)
    plot_seasonal_profiles(analyzed_df)

    # Статистика
    st.subheader(f"Descriptive Statistics for {city}")
    stat_df = analyzed_df.describe()
    stat_df.index = stat_df.index.astype(str)
    st.table(stat_df)

    # Выбор режима и API ключа
    mode = st.sidebar.selectbox("Choose mode", ["Sync", "Async"])
    api_key = st.sidebar.text_input("Enter API Key")

    if api_key:
        current_temp = None
        if mode == "Sync":
            current_temp = get_current_temperature(city, api_key)
        elif mode == "Async":
            current_temp = asyncio.run(get_current_temperature_async(city, api_key))

        if current_temp is not None:
            # Определение текущего сезона
            latest_date = analyzed_df['timestamp'].max()
            latest_season = analyzed_df[analyzed_df['timestamp'] == latest_date]['season'].values[0]
            seasonal_avg = analyzed_df[analyzed_df['season'] == latest_season]['temperature'].mean()

            # Сравнение с нормой
            temp_status = "normal"
            if current_temp > seasonal_avg + 2:
                temp_status = "above normal"
            elif current_temp < seasonal_avg - 2:
                temp_status = "below normal"

            st.write(f"Current temperature in {city}: {current_temp} °C ({temp_status} for {latest_season})")

else:
    st.write("Upload a file to start")
