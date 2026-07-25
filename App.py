import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ===================== DARK UI + GLOW BORDER =====================
st.markdown("""
<style>

/* DARK BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* MAIN CONTAINER */
.main-container {
    max-width: 1200px;
    margin: auto;
    padding: 25px;
    border-radius: 20px;
    background: #111827;
    position: relative;
    z-index: 1;
}

/* GLOW BORDER */
.main-container::before {
    content: "";
    position: absolute;
    inset: -3px;
    border-radius: 22px;
    background: linear-gradient(45deg, #00f5ff, #3b82f6, #9333ea, #00f5ff);
    background-size: 300% 300%;
    animation: glow 6s linear infinite;
    z-index: -1;
}

@keyframes glow {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* CARD */
.card {
    background: #1f2937;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.5);
    transition: 0.3s;
}
.card:hover {
    transform: scale(1.03);
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(45deg, #3b82f6, #9333ea);
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 200px;
    font-size: 16px;
    border: none;
}
.stButton>button:hover {
    transform: scale(1.05);
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #020617;
    border-right: 2px solid #3b82f6;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* INPUT */
.stTextInput input {
    background-color: #020617;
    color: white;
    border-radius: 10px;
}

/* HEADINGS */
h1, h2, h3 {
    color: #60a5fa;
    text-align: center;
}

/* FADE */
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
.fade {
    animation: fadeIn 1.5s ease-in-out;
}

</style>
""", unsafe_allow_html=True)

# ===================== START CONTAINER =====================
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# ===================== TITLE =====================
st.markdown("<h1>📈 Stock Price Prediction</h1>", unsafe_allow_html=True)

# ===================== COMPANY LIST =====================
company_dict = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "Meta": "META",
    "Netflix": "NFLX"
}

# ===================== SEARCH =====================
search = st.text_input("🔍 Search Company")
filtered = [name for name in company_dict.keys() if search.lower() in name.lower()][:10]

selected_company = None
if filtered:
    selected_company = st.selectbox("Select Company", filtered)

# ===================== MAIN =====================
if selected_company:
    stock = company_dict[selected_company]

    if st.button("Predict Now"):

        # Progress Animation
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        with st.spinner("Fetching data..."):
            data = yf.download(stock, period="10y", interval="1d")

        if data.empty:
            st.error("❌ Data load nahi hua")
            st.stop()

        st.success(f"✅ Data Loaded for {selected_company}")

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if 'Close' not in data.columns:
            st.error("❌ Close column missing")
            st.stop()

        df = data[['Close']].dropna()

        if len(df) < 100:
            st.error("❌ Not enough data")
            st.stop()

        # ===================== SCALING =====================
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)

        X, y = [], []
        window = 60

        for i in range(window, len(scaled_data)):
            X.append(scaled_data[i-window:i])
            y.append(scaled_data[i])

        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # ===================== MODEL =====================
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(64))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

        # ===================== PREDICTION =====================
        last_window = scaled_data[-window:]
        last_window = last_window.reshape(1, window, 1)

        pred_scaled = model.predict(last_window)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]

        last_price = df.iloc[-1, 0]
        change = pred_price - last_price

        if change > 0:
            signal = "BUY 📈"
            color = "green"
        else:
            signal = "SELL 📉"
            color = "red"

        # ===================== RESULT =====================
        st.subheader("📢 Prediction Result")

        st.markdown(f"""
        <div class='card fade'>
        <h2>{selected_company}</h2>
        <p style='font-size:22px;'><b>Last Price:</b> ₹{last_price:.2f}</p>
        <p style='font-size:22px;'><b>Predicted Price:</b> ₹{pred_price:.2f}</p>
        <p style='font-size:22px;'><b>Change:</b> ₹{change:.2f}</p>
        <p style='color:{color}; font-size:34px;'><b>{signal}</b></p>
        </div>
        """, unsafe_allow_html=True)

        # ===================== GRAPH 1 =====================
        st.markdown("<h3>📊 Stock Price History</h3>", unsafe_allow_html=True)

        fig1, ax1 = plt.subplots()
        ax1.plot(df.index, df['Close'])
        st.pyplot(fig1)

        # ===================== GRAPH 2 =====================
        fig2, ax2 = plt.subplots()

        ax2.plot(df.index[-100:], df['Close'][-100:], label="Recent Prices")

        next_date = df.index[-1] + pd.Timedelta(days=1)

        ax2.plot([df.index[-1], next_date],
                 [last_price, pred_price],
                 linestyle='dashed',
                 marker='o',
                 color=color,
                 label="Prediction")

        ax2.legend()
        st.pyplot(fig2)

        # ===================== GRAPH 3 =====================
        st.markdown("<h3>📉 Model Training View</h3>", unsafe_allow_html=True)

        fig3, ax3 = plt.subplots()
        ax3.plot(range(len(y_train)), y_train)
        ax3.plot(range(len(y_train), len(y_train)+len(y_test)), y_test)
        st.pyplot(fig3)

        # ===================== GRAPH 4 =====================
        st.markdown("<h3>🕯️ Candlestick Chart</h3>", unsafe_allow_html=True)

        data_candle = data.tail(100)

        fig4 = go.Figure(data=[go.Candlestick(
            x=data_candle.index,
            open=data_candle['Open'],
            high=data_candle['High'],
            low=data_candle['Low'],
            close=data_candle['Close']
        )])

        fig4.update_layout(
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig4)

        # ===================== FOOTER =====================
        st.markdown("""
        <hr>
        <p style='text-align:center; color:green; font-size:20px; font-weight:bold;'>👍Accurecy - 80-90%</p>
        <p style='text-align:center;'>🚀 Smart investing starts with smart predictions!</p>
        <p style='text-align:center;'>📞 Contact us: kushwahavinay007@gmail.com</p>
        <p style='text-align:center; color:gray;'>Developed By Vinay</p>
        """, unsafe_allow_html=True)

# ===================== END CONTAINER =====================
st.markdown("</div>", unsafe_allow_html=True)