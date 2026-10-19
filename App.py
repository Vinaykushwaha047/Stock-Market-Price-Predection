import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Sequential


st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="centered",
)

assets = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "ITC": "ITC.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "Meta": "META",
    "Netflix": "NFLX",
    "NVIDIA": "NVDA",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Crude Oil": "CL=F",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "NIFTY 50": "^NSEI",
    "Bank NIFTY": "^NSEBANK",
    "Sensex": "^BSESN",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
}


def normalize_market_data(data):
    if data is None or not isinstance(data, pd.DataFrame):
        return data

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    elif isinstance(data.columns, pd.Index):
        data.columns = [str(col[0]) if isinstance(col, tuple) else str(col) for col in data.columns]

    return data


@st.cache_data(ttl=300)
def download_data(ticker, interval="1d", period="5y"):
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    data = normalize_market_data(data)
    if data.empty and interval == "1d":
        data = yf.download(
            ticker,
            period="max",
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        data = normalize_market_data(data)
    return data


@st.cache_data(ttl=900)
def get_option_expirations(ticker):
    return yf.Ticker(ticker).options


@st.cache_data(ttl=900)
def get_option_chain(ticker, expiration):
    chain = yf.Ticker(ticker).option_chain(expiration)
    required_columns = {"strike", "openInterest"}
    if not required_columns.issubset(chain.calls.columns) or not required_columns.issubset(chain.puts.columns):
        raise ValueError("The options data source did not return open-interest values.")

    calls = chain.calls[["strike", "openInterest"]].rename(
        columns={"openInterest": "Call OI"}
    )
    puts = chain.puts[["strike", "openInterest"]].rename(
        columns={"openInterest": "Put OI"}
    )
    options = calls.merge(puts, on="strike", how="outer").fillna(0)
    options["strike"] = pd.to_numeric(options["strike"], errors="coerce")
    options["Call OI"] = pd.to_numeric(options["Call OI"], errors="coerce").fillna(0)
    options["Put OI"] = pd.to_numeric(options["Put OI"], errors="coerce").fillna(0)
    return options.dropna(subset=["strike"]).sort_values("strike")


def create_sequences(values, window_size):
    sequences = []
    targets = []
    for index in range(window_size, len(values)):
        sequences.append(values[index - window_size:index])
        targets.append(values[index, 0])
    return np.asarray(sequences), np.asarray(targets)


def make_prediction(ticker, timeframe):
    timeframe_settings = {
        "15 minutes": {"interval": "15m", "period": "60d", "minutes": 15},
        "30 minutes": {"interval": "30m", "period": "60d", "minutes": 30},
        "1 hour": {"interval": "60m", "period": "60d", "minutes": 60},
        "1 day": {"interval": "1d", "period": "5y", "minutes": None},
    }
    settings = timeframe_settings[timeframe]
    data = normalize_market_data(download_data(
        ticker,
        interval=settings["interval"],
        period=settings["period"],
    ))
    if data.empty:
        raise ValueError(f"No {timeframe} price history is available for this asset. Try another asset or timeframe.")

    close_series = None
    for key in ["Close", ("Close", ticker), ("Close", ticker.upper())]:
        if key in data.columns:
            close_series = data[key]
            break
    if close_series is None:
        close_candidates = [col for col in data.columns if str(col).lower().startswith("close")]
        if close_candidates:
            close_series = data[close_candidates[0]]
    if close_series is None:
        raise ValueError("No price history was found for this asset. Please try another one.")

    close = pd.Series(close_series).dropna()
    if len(close) <= 60:
        raise ValueError(f"Not enough {timeframe} price candles are available to make a prediction.")

    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(close.to_numpy().reshape(-1, 1))
    window_size = 60
    sequences, targets = create_sequences(scaled_prices, window_size)

    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(32),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(
        sequences,
        targets,
        epochs=20,
        batch_size=32,
        verbose=0,
    )

    recent_prices = scaled_prices[-window_size:].reshape(1, window_size, 1)
    predicted_scaled = model.predict(recent_prices, verbose=0)
    predicted_price = float(scaler.inverse_transform(predicted_scaled)[0, 0])
    current_price = float(close.iloc[-1])

    return {
        "asset": ticker,
        "timeframe": timeframe,
        "forecast_minutes": settings["minutes"],
        "history": data.loc[close.index, ["Open", "High", "Low", "Close"]].tail(180),
        "current_price": current_price,
        "predicted_price": predicted_price,
        "change_percent": (predicted_price / current_price - 1) * 100,
    }


st.title("Stock Price Predictor")
st.caption("Choose an asset and timeframe to estimate its next price move.")

with st.sidebar:
    st.header("Choose a stock")
    selected_asset = st.selectbox("Stock or market", list(assets))
    selected_timeframe = st.selectbox(
        "Prediction timeframe",
        ["15 minutes", "30 minutes", "1 hour", "1 day"],
    )
    predict_button = st.button("Predict price", type="primary", use_container_width=True)
    st.caption("Intraday predictions use recent market candles. Data availability varies by asset.")

if predict_button:
    st.session_state["prediction"] = None
    try:
        with st.spinner(f"Loading {selected_timeframe} data and predicting {selected_asset}..."):
            st.session_state["prediction"] = make_prediction(
                assets[selected_asset],
                selected_timeframe,
            )
    except Exception as error:
        st.error(f"Could not create a prediction: {error}")

prediction = st.session_state.get("prediction")
if (
    prediction
    and prediction["asset"] == assets[selected_asset]
    and prediction["timeframe"] == selected_timeframe
):
    st.subheader(f"{selected_asset} - {selected_timeframe} forecast")
    change_percent = prediction["change_percent"]
    if change_percent > 0:
        direction = "Market may move up"
        signal = "BUY"
        signal_type = "success"
        signal_color = "#16a34a"
    elif change_percent < 0:
        direction = "Market may move down"
        signal = "SELL"
        signal_type = "error"
        signal_color = "#dc2626"
    else:
        direction = "Market may stay flat"
        signal = "HOLD"
        signal_type = "info"
        signal_color = "#64748b"

    current_column, estimate_column, change_column, signal_column = st.columns(4)
    current_column.metric("Latest price", f"{prediction['current_price']:,.2f}")
    estimate_column.metric("Estimated next price", f"{prediction['predicted_price']:,.2f}")
    change_column.metric("Estimated change", f"{change_percent:+.2f}%")
    signal_column.metric("Model signal", signal)
    getattr(st, signal_type)(f"**{direction}** based on the model's price estimate. This is uncertain, not a guarantee.")

    history = prediction["history"]
    forecast_date = (
        history.index[-1] + pd.offsets.BDay(1)
        if prediction["forecast_minutes"] is None
        else history.index[-1] + pd.Timedelta(minutes=prediction["forecast_minutes"])
    )
    chart_type = st.radio(
        "Chart type",
        ["Candlestick", "Line"],
        horizontal=True,
        key="price_chart_type",
    )
    chart = go.Figure()
    if chart_type == "Candlestick":
        chart.add_trace(go.Candlestick(
            x=history.index,
            open=history["Open"],
            high=history["High"],
            low=history["Low"],
            close=history["Close"],
            name=f"{selected_timeframe} candles",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        ))
        chart.update_layout(xaxis_rangeslider_visible=False)
    else:
        chart.add_trace(go.Scatter(
            x=history.index,
            y=history["Close"],
            mode="lines",
            name="Closing price",
            line={"color": "#2563eb", "width": 2},
        ))
    chart.add_trace(go.Scatter(
        x=[forecast_date],
        y=[prediction["predicted_price"]],
        mode="markers",
        name=f"Estimated next price ({direction.lower()})",
        marker={"color": signal_color, "size": 11},
    ))
    chart.update_layout(
        title=f"Recent price history - {selected_timeframe} - {chart_type}",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=420,
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
        legend={"orientation": "h", "y": 1.12},
    )
    st.plotly_chart(chart, use_container_width=True)
elif not prediction or prediction["asset"] != assets[selected_asset] or prediction["timeframe"] != selected_timeframe:
    st.info("Choose an asset and timeframe from the sidebar, then select **Predict price**.")

with st.expander("Options open interest"):
    ticker = assets[selected_asset]
    st.caption(
        "View current Call and Put open interest by strike. Historical OI changes are not available here, and options coverage depends on Yahoo Finance."
    )

    if st.session_state.get("oi_ticker") != ticker:
        if st.button("Load options open interest", key="load_options_oi"):
            try:
                with st.spinner("Loading available option expirations..."):
                    expirations = get_option_expirations(ticker)
                if expirations:
                    st.session_state["oi_ticker"] = ticker
                    st.session_state["oi_expirations"] = expirations
                else:
                    st.info("Options data is not available for this asset.")
            except Exception as error:
                st.error(f"Could not load options data: {error}")

    if st.session_state.get("oi_ticker") == ticker:
        expirations = st.session_state["oi_expirations"]
        expiration = st.selectbox(
            "Expiration date",
            expirations,
            key=f"oi_expiration_{ticker}",
        )
        try:
            with st.spinner("Loading open interest by strike..."):
                options = get_option_chain(ticker, expiration)
            if options.empty or (options["Call OI"].sum() + options["Put OI"].sum() == 0):
                st.info("No open-interest values are available for this expiration.")
            else:
                call_total = int(options["Call OI"].sum())
                put_total = int(options["Put OI"].sum())
                call_column, put_column = st.columns(2)
                call_column.metric("Total Call OI", f"{call_total:,}")
                put_column.metric("Total Put OI", f"{put_total:,}")

                oi_chart = go.Figure()
                oi_chart.add_trace(go.Bar(
                    x=options["strike"],
                    y=options["Put OI"],
                    name="Put OI",
                    marker_color="#5acb63",
                ))
                oi_chart.add_trace(go.Bar(
                    x=options["strike"],
                    y=options["Call OI"],
                    name="Call OI",
                    marker_color="#ed6464",
                ))
                oi_chart.update_layout(
                    title=f"Open interest by strike · {expiration}",
                    xaxis_title="Strike price",
                    yaxis_title="Open interest (contracts)",
                    barmode="group",
                    template="plotly_white",
                    height=420,
                    margin={"l": 10, "r": 10, "t": 55, "b": 10},
                    legend={"orientation": "h", "y": 1.12},
                )
                st.plotly_chart(oi_chart, use_container_width=True)
        except Exception as error:
            st.error(f"Could not load open interest for this expiration: {error}")

st.caption(
    "For educational purposes only. This model-based estimate is uncertain and is not financial advice."
)
