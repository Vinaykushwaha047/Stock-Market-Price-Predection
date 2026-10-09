import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Market Predictor Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(251, 191, 36, 0.16), transparent 28%),
            radial-gradient(ellipse at 92% 8%, rgba(217, 70, 239, 0.12), transparent 26%),
            linear-gradient(135deg, #fff7ed 0%, #f5f3ff 42%, #ecfeff 100%);
        color: #172033;
    }

    [data-testid="stAppViewContainer"] { background: transparent; }
    [data-testid="stHeader"] { background: rgba(255, 255, 255, 0.72); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #fff7ed 48%, #f5f3ff 100%);
        border-right: 1px solid #e9d5ff;
    }
    [data-testid="stSidebar"] * { color: #243249; }
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        background: linear-gradient(100deg, #7c3aed, #db2777, #f97316);
        border: 0;
        color: white;
        font-weight: 750;
        box-shadow: 0 8px 18px rgba(190, 24, 93, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 24px rgba(190, 24, 93, 0.3);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        animation: fade-up 0.65s ease-out both;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(90deg, #7c3aed, #db2777, #f97316, #059669, #2563eb);
        background-size: 200% auto;
        color: #7c3aed;
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: 0.03em;
        animation: gradient-shift 6s ease-in-out infinite;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 17px;
        margin-bottom: 24px;
    }

    .market-strip {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        justify-content: center;
        margin-bottom: 26px;
    }

    .market-pill {
        background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(250,245,255,0.94));
        border: 1px solid #e9d5ff;
        border-radius: 999px;
        padding: 10px 18px;
        color: #334155;
        font-weight: 600;
        box-shadow: 0 8px 22px rgba(124, 58, 237, 0.1);
        animation: float-in 0.55s ease-out both;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .market-pill:nth-child(2) { animation-delay: 0.08s; }
    .market-pill:nth-child(3) { animation-delay: 0.16s; }
    .market-pill:nth-child(4) { animation-delay: 0.24s; }
    .market-pill:hover {
        transform: translateY(-5px) scale(1.04);
        box-shadow: 0 16px 30px rgba(219, 39, 119, 0.24), 0 0 0 3px rgba(217, 70, 239, 0.08);
        filter: saturate(1.2);
    }

    .asset-card, .glass-panel, .summary-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.98), #fff7ed 58%, #f5f3ff);
        padding: 22px 20px;
        border-radius: 18px;
        border: 1px solid #eadcf5;
        box-shadow: 0 12px 30px rgba(124, 58, 237, 0.1);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        animation: fade-up 0.7s ease-out both;
    }

    .asset-card:hover, .summary-card:hover {
        transform: translateY(-8px) scale(1.025);
        border-color: #d8b4fe;
        box-shadow:
            0 22px 44px rgba(124, 58, 237, 0.18),
            0 8px 20px rgba(236, 72, 153, 0.16),
            0 0 0 3px rgba(217, 70, 239, 0.08);
        filter: saturate(1.12);
        animation: hover-glow 1.4s ease-in-out infinite alternate;
    }

    .price-title { text-align: center; font-size: 18px; color: #6b5b7b; }
    .price-value { text-align: center; font-size: 38px; font-weight: 800; color: #31204f; }

    .prediction-box {
        background: linear-gradient(135deg, #ede9fe, #fce7f3 55%, #ffedd5);
        padding: 26px 20px;
        border-radius: 20px;
        text-align: center;
        border: 1px solid #e9a8d4;
        box-shadow: 0 12px 30px rgba(219, 39, 119, 0.14);
        animation: fade-up 0.75s ease-out both;
    }

    .predicted-price { font-size: 42px; font-weight: 800; color: #7c3aed; }
    .buy { color: #16a34a; font-size: 34px; font-weight: 800; }
    .sell { color: #dc2626; font-size: 34px; font-weight: 800; }
    .hold { color: #d97706; font-size: 34px; font-weight: 800; }

    .section-title {
        color: #7c3aed;
        font-size: 26px;
        font-weight: 700;
        margin-top: 28px;
        margin-bottom: 14px;
        border-left: 4px solid #db2777;
        padding-left: 12px;
    }

    .summary-card { padding: 18px; min-height: 120px; }
    .summary-label {
        font-size: 13px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .summary-value { font-size: 28px; font-weight: 800; margin-top: 10px; color: #6d28d9; }
    .footer { text-align: center; color: #766681; padding: 30px 0 10px 0; }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff, #fdf4ff 55%, #fff7ed);
        padding: 18px 16px;
        border-radius: 12px;
        border: 1px solid #eadcf5;
        border-top: 3px solid #8b5cf6;
        box-shadow: 0 10px 22px rgba(124, 58, 237, 0.09);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease, filter 0.25s ease;
    }
    div[data-testid="stMetric"]:nth-of-type(4n+2) { border-top-color: #ec4899; }
    div[data-testid="stMetric"]:nth-of-type(4n+3) { border-top-color: #f97316; }
    div[data-testid="stMetric"]:nth-of-type(4n+4) { border-top-color: #14b8a6; }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-7px) scale(1.035);
        border-color: #d8b4fe;
        box-shadow:
            0 18px 34px rgba(124, 58, 237, 0.18),
            0 7px 18px rgba(236, 72, 153, 0.15),
            0 0 0 3px rgba(217, 70, 239, 0.08);
        filter: saturate(1.12);
        animation: hover-glow 1.4s ease-in-out infinite alternate;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #243249; }

    .confidence-high { color: #16a34a; font-weight: 700; }
    .confidence-medium { color: #d97706; font-weight: 700; }
    .confidence-low { color: #dc2626; font-weight: 700; }

    .forecast-chip {
        display: inline-block;
        padding: 8px 12px;
        border-radius: 999px;
        background: linear-gradient(100deg, #ede9fe, #fce7f3);
        border: 1px solid #e9a8d4;
        color: #6d28d9;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
        transition: transform 0.2s ease, background 0.2s ease;
    }
    .forecast-chip:hover { transform: translateY(-2px); background: linear-gradient(100deg, #ddd6fe, #fbcfe8); }

    .prediction-box:hover {
        transform: translateY(-7px) scale(1.02);
        border-color: #c084fc;
        box-shadow:
            0 22px 42px rgba(124, 58, 237, 0.2),
            0 8px 22px rgba(236, 72, 153, 0.16),
            0 0 28px rgba(217, 70, 239, 0.14);
        filter: saturate(1.14);
        animation: hover-glow 1.4s ease-in-out infinite alternate;
    }

    [data-testid="stPlotlyChart"] {
        border: 1px solid rgba(192, 132, 252, 0.35);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.62);
        box-shadow: 0 12px 28px rgba(124, 58, 237, 0.1);
        transition: transform 0.28s ease, box-shadow 0.28s ease, border-color 0.28s ease;
    }
    [data-testid="stPlotlyChart"]:hover {
        transform: translateY(-5px) scale(1.008);
        border-color: #c084fc;
        box-shadow:
            0 22px 42px rgba(124, 58, 237, 0.17),
            0 8px 20px rgba(236, 72, 153, 0.12);
    }

    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
        transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
    }
    [data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {
        transform: translateY(-3px) scale(1.025);
        filter: saturate(1.25);
        box-shadow: 0 12px 25px rgba(124, 58, 237, 0.22);
    }
    [data-testid="stButton"] button:active, [data-testid="stDownloadButton"] button:active {
        transform: translateY(0) scale(0.99);
    }

    [data-testid="stTabs"] [role="tab"] {
        transition: color 0.2s ease, background-color 0.2s ease, transform 0.2s ease;
        border-radius: 10px 10px 0 0;
    }
    [data-testid="stTabs"] [role="tab"]:hover {
        color: #a21caf;
        background: rgba(217, 70, 239, 0.08);
        transform: translateY(-2px);
    }

    [data-testid="stDataFrame"] {
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }
    [data-testid="stDataFrame"]:hover {
        transform: translateY(-4px);
        border-color: #d8b4fe;
        box-shadow: 0 18px 34px rgba(124, 58, 237, 0.16);
    }

    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #7c3aed;
        border-bottom-color: #db2777;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #eadcf5;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 10px 24px rgba(124, 58, 237, 0.08);
    }

    @keyframes fade-up {
        from { opacity: 0; transform: translateY(14px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes float-in {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% center; }
        50% { background-position: 100% center; }
    }
    @keyframes hover-glow {
        from {
            box-shadow: 0 18px 38px rgba(124, 58, 237, 0.16), 0 0 0 2px rgba(217, 70, 239, 0.06);
        }
        to {
            box-shadow: 0 24px 48px rgba(219, 39, 119, 0.2), 0 0 0 4px rgba(124, 58, 237, 0.1);
        }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================

st.markdown(
    "<div class='main-title'>🚀 AI Stock & Market Predictor Pro</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>"
    "15 Years Historical Data • Advanced LSTM • Walk-Forward Validation • "
    "Ensemble Model • Next Trading Day Prediction"
    "</div>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='market-strip'>
        <span class='market-pill'>📈 15Y Data</span>
        <span class='market-pill'>🤖 LSTM + Ensemble</span>
        <span class='market-pill'>📊 Technical Signals</span>
        <span class='market-pill'>🧠 Confidence Score</span>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# ASSETS
# ============================================================

assets = {
    # Indian Stocks
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "ITC": "ITC.NS",
    "Bharti Airtel": "BHARTIARTL.NS",

    # Global Stocks
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "Meta": "META",
    "Netflix": "NFLX",
    "NVIDIA": "NVDA",

    # Commodity
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Crude Oil": "CL=F",

    # Crypto
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",

    # Indian Indices
    "NIFTY 50": "^NSEI",
    "Bank NIFTY": "^NSEBANK",
    "Sensex": "^BSESN",

    # Global Indices
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI"
}

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Prediction Settings")

    search = st.text_input(
        "🔍 Search Asset",
        placeholder="Example: NIFTY, Gold, Bitcoin"
    )

    filtered_assets = [
        name for name in assets
        if search.lower() in name.lower()
    ]

    if not filtered_assets:
        filtered_assets = list(assets.keys())

    selected_asset = st.selectbox(
        "📊 Select Asset",
        filtered_assets
    )

    st.markdown("---")

    st.subheader("🎯 Model Configuration")

    window_size = st.slider(
        "Lookback Window (Days)",
        min_value=30,
        max_value=120,
        value=60,
        step=10,
        help="Number of past days to consider for prediction"
    )

    epochs = st.slider(
        "Training Epochs",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
        help="More epochs = better training but risk of overfitting"
    )

    use_ensemble = st.checkbox(
        "Use Ensemble Model",
        value=True,
        help="Average predictions from 3 models for better accuracy"
    )

    st.markdown("---")
    st.subheader("📊 Chart Controls")

    chart_period = st.selectbox(
        "Chart history",
        ["1 month", "3 months", "6 months", "1 year", "3 years", "All available"],
        index=3,
        help="Choose the historical window shown in the interactive charts."
    )
    show_moving_averages = st.checkbox("Show moving averages", value=True)
    show_bollinger_bands = st.checkbox("Show Bollinger Bands", value=False)

    st.markdown("---")

    st.info(
        "📌 Historical data: Maximum 15 years available from Yahoo Finance."
    )

    st.info(
        "🎯 Target: Next Trading Day"
    )

    st.warning(
        "⚠️ **Note:** 99% accuracy scientifically impossible. "
        "Market influenced by unpredictable factors (news, politics, etc.). "
        "Best achievable: 60-75% directional accuracy."
    )

    st.markdown("---")

    predict_button = st.button(
        "🚀 RUN ADVANCED PREDICTION",
        use_container_width=True,
        type="primary"
    )

# ============================================================
# TECHNICAL INDICATORS (ENHANCED)
# ============================================================

def add_indicators(df):
    df = df.copy()

    # Moving Averages
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()

    # Exponential Moving Averages
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands
    df["BB_middle"] = df["Close"].rolling(20).mean()
    df["BB_std"] = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_middle"] + (df["BB_std"] * 2)
    df["BB_lower"] = df["BB_middle"] - (df["BB_std"] * 2)
    df["BB_position"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

    # Stochastic Oscillator
    low_14 = df["Low"].rolling(14).min()
    high_14 = df["High"].rolling(14).max()
    df["Stoch_K"] = 100 * (df["Close"] - low_14) / (high_14 - low_14)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # ATR (Volatility)
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    # Daily return & Volatility
    df["Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility"] = df["Return"].rolling(20).std()
    df["Volatility_50"] = df["Return"].rolling(50).std()

    # Volume indicators
    df["Volume_SMA"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"]

    # Price position
    df["Price_Position"] = (df["Close"] - df["Low"].rolling(50).min()) / \
                           (df["High"].rolling(50).max() - df["Low"].rolling(50).min())

    return df.dropna()

# ============================================================
# LOAD DATA
# ============================================================

def download_data(ticker):
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=15)

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    return data

# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(data):
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.loc[:, ~data.columns.duplicated()]

    required = ["Open", "High", "Low", "Close", "Volume"]

    for col in required:
        if col not in data.columns:
            data[col] = 0

    data = data[required].copy()
    data = data.dropna()

    return data

# ============================================================
# CREATE DATASET (IMPROVED)
# ============================================================

def create_dataset(values, window=60):
    X = []
    y = []

    for i in range(window, len(values)):
        X.append(values[i-window:i])
        y.append(values[i, 0])

    return np.array(X), np.array(y)

# ============================================================
# BUILD MODEL (ENHANCED ARCHITECTURE)
# ============================================================

def build_model(input_shape, dropout_rate=0.3, lstm_units=128):
    model = Sequential()

    # Bidirectional LSTM for better pattern capture
    model.add(
        Bidirectional(
            LSTM(
                lstm_units,
                return_sequences=True,
                input_shape=input_shape
            )
        )
    )
    model.add(Dropout(dropout_rate))

    model.add(
        LSTM(
            lstm_units // 2,
            return_sequences=False
        )
    )
    model.add(Dropout(dropout_rate))

    # Additional dense layers
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(dropout_rate / 2))

    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))

    # Adam optimizer with learning rate
    optimizer = Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )

    return model

# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def walk_forward_validation(scaled_data, window, n_folds=5):
    """
    Walk-forward validation for realistic performance estimation
    """
    fold_scores = []

    fold_size = len(scaled_data) // n_folds

    for fold in range(n_folds - 1):
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = min(train_end + fold_size // 4, len(scaled_data))

        train_data = scaled_data[:train_end]
        test_data = scaled_data[test_start:test_end]

        if len(train_data) < window + 10 or len(test_data) < 10:
            continue

        X_train, y_train = create_dataset(train_data, window)
        X_test, y_test = create_dataset(test_data, window)

        if len(X_train) < 50 or len(X_test) < 10:
            continue

        model = build_model((X_train.shape[1], X_train.shape[2]))

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=0
        )

        model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=32,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=0
        )

        pred = model.predict(X_test, verbose=0).flatten()

        # Calculate directional accuracy
        actual_change = np.diff(y_test)
        pred_change = np.diff(pred)

        direction_acc = np.mean(np.sign(actual_change) == np.sign(pred_change)) * 100
        fold_scores.append(direction_acc)

    return np.mean(fold_scores) if fold_scores else 0

# ============================================================
# ENSEMBLE PREDICTION
# ============================================================

def ensemble_predict(models, next_input):
    """
    Average predictions from multiple models
    """
    predictions = []

    for model in models:
        pred = model.predict(next_input, verbose=0)[0][0]
        predictions.append(pred)

    return np.mean(predictions), np.std(predictions)

# ============================================================
# CONFIDENCE SCORE
# ============================================================

def calculate_confidence(data, next_price, last_price):
    """
    Calculate prediction confidence based on volatility and trend
    """
    latest = data.iloc[-1]

    # Volatility factor (lower volatility = higher confidence)
    volatility = latest["Volatility"]
    vol_score = max(0, 100 - (volatility * 100 * 10))

    # RSI factor (extreme RSI = lower confidence)
    rsi = latest["RSI"]
    if 30 <= rsi <= 70:
        rsi_score = 100
    elif 20 <= rsi < 30 or 70 < rsi <= 80:
        rsi_score = 70
    else:
        rsi_score = 40

    # Volume factor
    volume_ratio = latest["Volume_Ratio"]
    if 0.8 <= volume_ratio <= 1.5:
        volume_score = 100
    else:
        volume_score = 60

    # Overall confidence
    confidence = (vol_score * 0.4 + rsi_score * 0.3 + volume_score * 0.3)

    return min(95, max(30, confidence))


def get_signal_summary(last_price, next_price, confidence):
    """Return structured summary for market and signal narratives."""
    delta = next_price - last_price
    pct = (delta / last_price) * 100

    if pct > 1.0:
        trend = "Bullish Momentum"
        tone = "Strong upward bias"
    elif pct < -1.0:
        trend = "Bearish Pressure"
        tone = "Downside risk remains elevated"
    else:
        trend = "Neutral Range"
        tone = "Sideways movement is likely"

    if confidence >= 70:
        risk = "Low Risk"
    elif confidence >= 50:
        risk = "Moderate Risk"
    else:
        risk = "High Risk"

    return {
        "delta": delta,
        "pct": pct,
        "trend": trend,
        "tone": tone,
        "risk": risk,
    }


def build_watchlist_summary(asset_name, ticker):
    """Build a compact market snapshot to enhance the dashboard."""
    try:
        df = download_data(ticker)
        cleaned = clean_data(df)
        cleaned = add_indicators(cleaned)
        if cleaned.empty:
            return None

        last = cleaned.iloc[-1]
        prev = cleaned.iloc[-2] if len(cleaned) > 1 else last
        current = float(last["Close"])
        previous = float(prev["Close"])
        change = ((current - previous) / previous) * 100 if previous else 0.0
        rsi = float(last["RSI"])

        if change > 0:
            direction = "Uptrend"
        elif change < 0:
            direction = "Downtrend"
        else:
            direction = "Flat"

        return {
            "name": asset_name,
            "price": current,
            "change": change,
            "direction": direction,
            "rsi": rsi,
        }
    except Exception:
        return None

# ============================================================
# MAIN PREDICTION
# ============================================================

if predict_button:
    ticker = assets[selected_asset]

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------
    with st.spinner(f"📥 Downloading 15 years data for {selected_asset}..."):
        try:
            raw_data = download_data(ticker)
        except Exception as e:
            st.error(f"Data download failed: {e}")
            st.stop()

    if raw_data.empty:
        st.error("❌ Historical data available nahi hai.")
        st.stop()

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------
    data = clean_data(raw_data)

    if len(data) < 500:
        st.warning(
            f"⚠️ Is asset ke liye sirf {len(data)} trading days available hain. "
            f"Kam se kam 500 days recommended hain."
        )

    # --------------------------------------------------------
    # INDICATORS
    # --------------------------------------------------------
    with st.spinner("📊 Calculating technical indicators..."):
        data = add_indicators(data)

    # --------------------------------------------------------
    # FEATURE SELECTION (ENHANCED)
    # --------------------------------------------------------
    features = [
        "Close",
        "SMA_10", "SMA_20", "SMA_50",
        "EMA_10", "EMA_20",
        "RSI",
        "MACD", "MACD_Signal", "MACD_Hist",
        "BB_position",
        "Stoch_K", "Stoch_D",
        "ATR",
        "Return", "Log_Return",
        "Volatility", "Volatility_50",
        "Volume_Ratio",
        "Price_Position"
    ]

    model_data = data[features].copy()

    # --------------------------------------------------------
    # SCALE
    # --------------------------------------------------------
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(model_data)

    # --------------------------------------------------------
    # WALK-FORWARD VALIDATION (for realistic accuracy estimate)
    # --------------------------------------------------------
    with st.spinner("🔄 Running walk-forward validation..."):
        wf_accuracy = walk_forward_validation(scaled, window_size, n_folds=5)

    # --------------------------------------------------------
    # CREATE SEQUENCES
    # --------------------------------------------------------
    X, y = create_dataset(scaled, window_size)

    # --------------------------------------------------------
    # TRAIN TEST SPLIT (Chronological - no lookahead bias)
    # --------------------------------------------------------
    split = int(len(X) * 0.80)
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    # --------------------------------------------------------
    # MODEL TRAINING
    # --------------------------------------------------------
    models = []

    if use_ensemble:
        st.info("🤖 Training 3-model ensemble for better accuracy...")

        configs = [
            {"dropout": 0.3, "units": 128},
            {"dropout": 0.4, "units": 96},
            {"dropout": 0.25, "units": 160}
        ]

        for i, config in enumerate(configs):
            with st.spinner(f"Training model {i+1}/3..."):
                model = build_model(
                    (X_train.shape[1], X_train.shape[2]),
                    dropout_rate=config["dropout"],
                    lstm_units=config["units"]
                )

                early_stop = EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True
                )

                reduce_lr = ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6
                )

                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=32,
                    validation_split=0.15,
                    callbacks=[early_stop, reduce_lr],
                    verbose=0
                )

                models.append(model)
    else:
        with st.spinner("🤖 Training LSTM model..."):
            model = build_model(
                (X_train.shape[1], X_train.shape[2]),
                dropout_rate=0.3,
                lstm_units=128
            )

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )

            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=32,
                validation_split=0.15,
                callbacks=[early_stop],
                verbose=0
            )

            models = [model]

    # --------------------------------------------------------
    # TEST PREDICTION
    # --------------------------------------------------------
    predicted_scaled = models[0].predict(X_test, verbose=0).flatten()

    # --------------------------------------------------------
    # INVERSE TRANSFORM
    # --------------------------------------------------------
    dummy_actual = np.zeros((len(y_test), len(features)))
    dummy_predicted = np.zeros((len(predicted_scaled), len(features)))

    dummy_actual[:, 0] = y_test
    dummy_predicted[:, 0] = predicted_scaled

    actual_prices = scaler.inverse_transform(dummy_actual)[:, 0]
    predicted_prices = scaler.inverse_transform(dummy_predicted)[:, 0]

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------
    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    r2 = r2_score(actual_prices, predicted_prices)

    # Directional accuracy
    actual_change = np.diff(actual_prices)
    predicted_change = np.diff(predicted_prices)

    direction_accuracy = np.mean(np.sign(actual_change) == np.sign(predicted_change)) * 100

    # --------------------------------------------------------
    # NEXT DAY PREDICTION
    # --------------------------------------------------------
    last_60 = scaled[-window_size:]
    next_input = last_60.reshape(1, window_size, len(features))

    if use_ensemble:
        next_scaled, pred_std = ensemble_predict(models, next_input)
    else:
        next_scaled = models[0].predict(next_input, verbose=0)[0][0]
        pred_std = 0

    dummy_next = np.zeros((1, len(features)))
    dummy_next[0, 0] = next_scaled

    next_price = scaler.inverse_transform(dummy_next)[0, 0]

    # --------------------------------------------------------
    # LAST PRICE
    # --------------------------------------------------------
    last_price = float(data["Close"].iloc[-1])

    # --------------------------------------------------------
    # CHANGE
    # --------------------------------------------------------
    price_change = next_price - last_price
    percentage_change = (price_change / last_price) * 100

    # --------------------------------------------------------
    # CONFIDENCE SCORE
    # --------------------------------------------------------
    confidence = calculate_confidence(data, next_price, last_price)

    # Adjust confidence based on walk-forward accuracy
    confidence = confidence * (wf_accuracy / 70)  # Normalize to 70% baseline
    confidence = min(95, max(30, confidence))

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------
    if percentage_change > 0.5:
        signal = "BUY 📈"
        signal_class = "buy"
    elif percentage_change < -0.5:
        signal = "SELL 📉"
        signal_class = "sell"
    else:
        signal = "HOLD ⏸️"
        signal_class = "hold"

    # ========================================================
    # RESULT HEADER
    # ========================================================
    summary = get_signal_summary(last_price, next_price, confidence)

    st.markdown(
        "<div class='section-title'>🎯 Next Trading Day Prediction</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class='market-strip'>
            <span class='market-pill'>📌 {selected_asset}</span>
            <span class='market-pill'>📉 {summary['trend']}</span>
            <span class='market-pill'>🛡️ {summary['risk']}</span>
            <span class='market-pill'>🧭 {summary['tone']}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # RESULT CARDS
    # ========================================================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class='asset-card'>
            <div class='price-title'>Current Price</div>
            <div class='price-value'>{last_price:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class='prediction-box'>
            <div class='price-title'>Predicted Next-Day Price</div>
            <div class='predicted-price'>{next_price:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class='asset-card'>
            <div class='price-title'>Expected Change</div>
            <div class='price-value'>{percentage_change:+.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        confidence_class = "confidence-high" if confidence >= 70 else \
                          "confidence-medium" if confidence >= 50 else "confidence-low"

        st.markdown(
            f"""
            <div class='asset-card'>
            <div class='price-title'>Confidence</div>
            <div class='price-value {confidence_class}'>{confidence:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # SIGNAL
    # ========================================================
    st.markdown(
        f"""
        <div class='prediction-box' style='margin-top:25px;'>
        <div class='{signal_class}'>{signal}</div>
        <p style='color:#475569;'>
        Expected price change: {price_change:+.2f} |
        Confidence: <span class='{confidence_class}'>{confidence:.1f}%</span>
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # AI MARKET OUTLOOK
    # ========================================================
    st.markdown(
        "<div class='section-title'>🧠 AI Market Outlook</div>",
        unsafe_allow_html=True
    )

    outlook_col1, outlook_col2, outlook_col3 = st.columns(3)

    with outlook_col1:
        st.markdown(
            f"""
            <div class='summary-card'>
                <div class='summary-label'>Price Delta</div>
                <div class='summary-value'>{price_change:+,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with outlook_col2:
        st.markdown(
            f"""
            <div class='summary-card'>
                <div class='summary-label'>Expected Return</div>
                <div class='summary-value'>{percentage_change:+.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with outlook_col3:
        st.markdown(
            f"""
            <div class='summary-card'>
                <div class='summary-label'>Risk Level</div>
                <div class='summary-value'>{summary['risk']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    tab1, tab2, tab3 = st.tabs(["Market Outlook", "Scenario Analysis", "Watchlist"])

    with tab1:
        st.info(
            f"{selected_asset} is currently showing a **{summary['trend']}** outlook with a **{summary['risk'].lower()}** profile. "
            f"The model estimates a next-day move of **{percentage_change:+.2f}%** with confidence at **{confidence:.1f}%**."
        )

    with tab2:
        bullish_price = last_price * (1 + (percentage_change / 100) * 1.4)
        base_price = next_price
        bearish_price = last_price * (1 + (percentage_change / 100) * 0.6)

        scenario_data = {
            "Bullish Case": bullish_price,
            "Base Case": base_price,
            "Bearish Case": bearish_price,
        }

        for label, value in scenario_data.items():
            st.markdown(
                f"<div class='forecast-chip'>{label}: {value:,.2f}</div>",
                unsafe_allow_html=True
            )

        st.caption("Model scenario view: bullish, base, and bearish price outcomes based on current technical momentum and confidence range.")

    with tab3:
        watchlist_assets = [
            ("NIFTY 50", "^NSEI"),
            ("Gold", "GC=F"),
            ("Bitcoin", "BTC-USD"),
            ("S&P 500", "^GSPC")
        ]
        watchlist_rows = []

        for name, ticker in watchlist_assets:
            row = build_watchlist_summary(name, ticker)
            if row is not None:
                watchlist_rows.append(row)

        if watchlist_rows:
            watch_df = pd.DataFrame(watchlist_rows)
            watch_df["change"] = watch_df["change"].map(lambda x: f"{x:+.2f}%")
            st.dataframe(
                watch_df[["name", "price", "change", "direction", "rsi"]].rename(columns={
                    "name": "Asset",
                    "price": "Price",
                    "change": "Daily Change",
                    "direction": "Trend",
                    "rsi": "RSI"
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Watchlist data is temporarily unavailable.")

    # ========================================================
    # MODEL ACCURACY
    # ========================================================
    st.markdown(
        "<div class='section-title'>📊 Model Performance</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Directional Accuracy",
            f"{direction_accuracy:.2f}%",
            delta=f"Walk-Forward: {wf_accuracy:.1f}%"
        )

    with c2:
        st.metric("MAE", f"{mae:.2f}")

    with c3:
        st.metric("RMSE", f"{rmse:.2f}")

    with c4:
        st.metric("R² Score", f"{r2:.4f}")

    with c5:
        st.metric("Training Days", f"{len(X_train):,}")

    # ========================================================
    # DATA INFORMATION
    # ========================================================
    st.markdown(
        "<div class='section-title'>📅 Historical Data Analysis</div>",
        unsafe_allow_html=True
    )

    d1, d2, d3 = st.columns(3)

    with d1:
        st.metric(
            "Data From",
            data.index[0].strftime("%d-%m-%Y")
        )

    with d2:
        st.metric(
            "Data Till",
            data.index[-1].strftime("%d-%m-%Y")
        )

    with d3:
        st.metric(
            "Trading Days",
            f"{len(data):,}"
        )

    # ========================================================
    # TECHNICAL ANALYSIS
    # ========================================================
    st.markdown(
        "<div class='section-title'>📈 Technical Analysis</div>",
        unsafe_allow_html=True
    )

    latest = data.iloc[-1]

    t1, t2, t3, t4, t5 = st.columns(5)

    with t1:
        st.metric("RSI", f"{latest['RSI']:.2f}")

    with t2:
        st.metric("SMA 20", f"{latest['SMA_20']:.2f}")

    with t3:
        st.metric("SMA 50", f"{latest['SMA_50']:.2f}")

    with t4:
        st.metric("MACD", f"{latest['MACD']:.2f}")

    with t5:
        st.metric("Stochastic", f"{latest['Stoch_K']:.2f}")

    st.markdown("#### Market snapshot")
    snapshot_cols = st.columns(4)
    one_day_return = float(latest["Return"] * 100)
    year_data = data.tail(252)
    year_return = (
        (float(data["Close"].iloc[-1]) / float(year_data["Close"].iloc[0]) - 1) * 100
        if len(year_data) > 1 else 0.0
    )
    annualized_volatility = float(latest["Volatility"] * np.sqrt(252) * 100)
    year_high = float(year_data["High"].max())
    year_low = float(year_data["Low"].min())

    with snapshot_cols[0]:
        st.metric("Daily Return", f"{one_day_return:+.2f}%")
    with snapshot_cols[1]:
        st.metric("1Y Return", f"{year_return:+.2f}%")
    with snapshot_cols[2]:
        st.metric("Annualized Volatility", f"{annualized_volatility:.1f}%")
    with snapshot_cols[3]:
        st.metric("52-Week Range", f"{year_low:,.2f} – {year_high:,.2f}")

    # ========================================================
    # INTERACTIVE MARKET TERMINAL CHART
    # ========================================================
    st.markdown(
        "<div class='section-title'>📊 Interactive Market Terminal</div>",
        unsafe_allow_html=True
    )

    chart_period_days = {
        "1 month": 21,
        "3 months": 63,
        "6 months": 126,
        "1 year": 252,
        "3 years": 756,
        "All available": len(data),
    }
    chart_data = data.tail(min(chart_period_days[chart_period], len(data)))
    volume_colors = np.where(
        chart_data["Close"] >= chart_data["Open"],
        "#16a34a",
        "#dc2626"
    )
    market_chart = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.62, 0.18, 0.20],
        subplot_titles=("Price action", "Volume", "RSI (14)")
    )

    market_chart.add_trace(
        go.Candlestick(
            x=chart_data.index,
            open=chart_data["Open"],
            high=chart_data["High"],
            low=chart_data["Low"],
            close=chart_data["Close"],
            name="OHLC",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626"
        ),
        row=1,
        col=1
    )

    if show_moving_averages:
        for column, label, color in [
            ("SMA_20", "SMA 20", "#2563eb"),
            ("SMA_50", "SMA 50", "#f97316"),
            ("SMA_200", "SMA 200", "#7c3aed"),
        ]:
            market_chart.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data[column],
                    name=label,
                    mode="lines",
                    line={"width": 1.5, "color": color}
                ),
                row=1,
                col=1
            )

    if show_bollinger_bands:
        market_chart.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["BB_upper"],
                name="Bollinger Upper",
                mode="lines",
                line={"width": 1, "color": "#94a3b8"},
                showlegend=False
            ),
            row=1,
            col=1
        )
        market_chart.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["BB_lower"],
                name="Bollinger Bands",
                mode="lines",
                line={"width": 1, "color": "#94a3b8"},
                fill="tonexty",
                fillcolor="rgba(59, 130, 246, 0.10)"
            ),
            row=1,
            col=1
        )

    market_chart.add_trace(
        go.Bar(
            x=chart_data.index,
            y=chart_data["Volume"],
            name="Volume",
            marker_color=volume_colors,
            opacity=0.75
        ),
        row=2,
        col=1
    )
    market_chart.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["RSI"],
            name="RSI",
            mode="lines",
            line={"width": 1.8, "color": "#7c3aed"}
        ),
        row=3,
        col=1
    )
    market_chart.add_hline(
        y=70,
        line_dash="dash",
        line_color="#dc2626",
        opacity=0.65,
        row=3,
        col=1
    )
    market_chart.add_hline(
        y=30,
        line_dash="dash",
        line_color="#16a34a",
        opacity=0.65,
        row=3,
        col=1
    )
    market_chart.update_yaxes(title_text="Price", row=1, col=1)
    market_chart.update_yaxes(title_text="Volume", row=2, col=1)
    market_chart.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    market_chart.update_layout(
        template="plotly_white",
        height=850,
        title=f"{selected_asset} · {chart_period}",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "y": 1.02, "x": 0},
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(market_chart, use_container_width=True)

    export_data = data.copy()
    export_data.index.name = "Date"
    st.download_button(
        "⬇️ Download technical history (CSV)",
        data=export_data.to_csv().encode("utf-8"),
        file_name=f"{ticker.replace('^', '').replace('=', '-')}_technical_history.csv",
        mime="text/csv"
    )

    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================
    st.markdown(
        "<div class='section-title'>🤖 Actual vs Predicted</div>",
        unsafe_allow_html=True
    )

    comparison = pd.DataFrame({
        "Actual": actual_prices,
        "Predicted": predicted_prices
    })

    comparison = comparison.tail(150)

    comparison_chart = go.Figure()

    comparison_chart.add_trace(
        go.Scatter(y=comparison["Actual"], mode="lines", name="Actual Price")
    )

    comparison_chart.add_trace(
        go.Scatter(y=comparison["Predicted"], mode="lines", name="Predicted Price")
    )

    comparison_chart.update_layout(
        template="plotly_white",
        height=500,
        title="Test Data: Actual vs Predicted"
    )

    st.plotly_chart(comparison_chart, use_container_width=True)

    # ========================================================
    # TRAINING LOSS
    # ========================================================
    st.markdown(
        "<div class='section-title'>🧠 LSTM Training Loss</div>",
        unsafe_allow_html=True
    )

    loss_chart = go.Figure()

    loss_chart.add_trace(
        go.Scatter(
            y=history.history["loss"],
            mode="lines+markers",
            name="Training Loss"
        )
    )

    loss_chart.add_trace(
        go.Scatter(
            y=history.history["val_loss"],
            mode="lines+markers",
            name="Validation Loss"
        )
    )

    loss_chart.update_layout(
        template="plotly_white",
        height=450,
        title="Training vs Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss"
    )

    st.plotly_chart(loss_chart, use_container_width=True)

    # ========================================================
    # DISCLAIMER
    # ========================================================
    st.error(
        "⚠️ **CRITICAL DISCLAIMER:** Stock/crypto/commodity prices **CANNOT** be predicted "
        "with 99% accuracy. This is **scientifically impossible** due to:\n\n"
        "- Market influenced by unpredictable news, politics, global events\n"
        "- Black swan events (pandemics, wars, crashes)\n"
        "- Human psychology and market sentiment\n"
        "- Regulatory changes\n\n"
        "**Best achievable directional accuracy: 60-75%** on test data.\n"
        "**Real-world trading accuracy: 50-65%** typically.\n\n"
        "This application is for **EDUCATIONAL PURPOSES ONLY** and is **NOT financial advice**. "
        "Never trade based solely on AI predictions. Always do your own research and consult "
        "a qualified financial advisor."
    )

# ============================================================
# DEFAULT SCREEN
# ============================================================

else:
    st.markdown(
        """
        <div class='asset-card'>
        <h2 style='text-align:center;'>👋 Welcome to AI Market Predictor Pro</h2>
        <p style='text-align:center;color:#475569;font-size:18px;'>
        Select an asset from the left sidebar and click <b>RUN ADVANCED PREDICTION</b>.
        </p>
        <p style='text-align:center;color:#64748b;font-size:15px;'>
        ✨ New Features: Walk-Forward Validation, Ensemble Models, Confidence Scores,
        Technical Indicators, Scenario Analysis, and Watchlist Monitoring
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    feature_cols = st.columns(4)
    feature_cards = [
        ("📈", "Market Forecast", "Next-day directional prediction with AI-based signal generation"),
        ("🧠", "Advanced Models", "LSTM + ensemble learning for pattern recognition"),
        ("📊", "Technical View", "RSI, MACD, moving averages, volatility, and price structure"),
        ("🔍", "Portfolio Lens", "Quick multi-asset overview using watchlist performance snapshots")
    ]

    for i, (icon, title, desc) in enumerate(feature_cards):
        with feature_cols[i]:
            st.markdown(
                f"""
                <div class='summary-card'>
                    <div style='font-size:28px;'>{icon}</div>
                    <div style='font-weight:700; margin-top:8px; font-size:18px;'>{title}</div>
                    <div style='color:#64748b; margin-top:8px; font-size:14px;'>{desc}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class='footer'>
    🚀 AI Market Predictor Pro<br>
    Developed by <b>Vinay</b><br>
    Educational Project • Advanced LSTM • Walk-Forward Validation • Ensemble Model<br>
    <span style='color:#ef4444;'>⚠️ NOT Financial Advice - For Educational Purposes Only</span>
    </div>
    """,
    unsafe_allow_html=True
)
