import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TIMEFRAMES = {
    '1 minute': '1m',
    '5 minutes': '5m',
    '15 minutes': '15m',
    '1 hour': '1h',
    '4 hours': '4h',
    '1 day': '1d'
}

TIMEFRAME_LIMITS = {
    '1m': 500,    # ~8 hours
    '5m': 500,    # ~42 hours
    '15m': 500,   # ~5 days
    '1h': 200,    # ~8 days
    '4h': 200,    # ~33 days
    '1d': 100     # ~3 months
}

INTERVALS = {
    '1 minute': '1m',
    '5 minutes': '5m',
    '15 minutes': '15m'
}

LOOKBACK_PERIODS = {
    '1 hour': 1/24,
    '4 hours': 4/24,
    '12 hours': 12/24,
    '1 day': 1,
    '3 days': 3,
    '1 week': 7,
    '2 weeks': 14,
    '1 month': 30,
    '2 months': 60,
    '3 months': 90
}

# Add timeframe selection in the UI, before symbol selection
st.subheader("Select Data Range")
col1, col2 = st.columns(2)

with col1:
    selected_lookback = st.selectbox(
        'Look back period',
        list(LOOKBACK_PERIODS.keys()),
        index=6  # Default to 1 week
    )

with col2:
    selected_interval = st.selectbox(
        'Data interval',
        list(INTERVALS.keys()),
        index=1  # Default to 5 minutes
    )

# Calculate required number of candles
days = LOOKBACK_PERIODS[selected_lookback]
interval_minutes = int(INTERVALS[selected_interval][:-1])
required_candles = int((days * 24 * 60) / interval_minutes)

# Show data points info
st.info(f"""
Data Configuration:
- Looking back: {selected_lookback}
- Interval: {selected_interval}
- Total data points: {required_candles:,}
""")

if required_candles > 10000:
    st.warning("⚠️ Large amount of data requested. Dashboard may take longer to load.")

# Initialize exchange
@st.cache_resource
def get_exchange():
    return ccxt.kucoin({
        'enableRateLimit': True,
    })

exchange = get_exchange()

# Get available symbols for validation
@st.cache_data(ttl=3600)
def get_available_symbols():
    try:
        markets = exchange.load_markets()
        # Convert all symbols to dash format to match input
        symbols = [market.replace('/', '-') for market in markets.keys() 
                  if market.endswith('USDT')]
        return sorted(symbols)
    except Exception as e:
        st.error(f"Error fetching symbols: {str(e)}")
        return []

# Fetch and calculate data functions
@st.cache_data(ttl=60)
def fetch_crypto_data(symbol, interval, lookback_days, required_candles):
    try:
        # Convert symbol format for KuCoin
        kucoin_symbol = symbol.replace('/', '-')
        
        # Add delay to respect rate limits
        time.sleep(exchange.rateLimit / 1000)
        
        ohlcv = exchange.fetch_ohlcv(
            kucoin_symbol,
            timeframe=INTERVALS[interval],
            limit=required_candles
        )
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except ccxt.NetworkError as e:
        st.error(f"Network error: {str(e)}")
        return None
    except ccxt.ExchangeError as e:
        st.error(f"Exchange error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

# Rest of your functions remain the same
def calculate_metrics(df):
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Normalized Volatility
    df['Volatility'] = (df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()) * 100
    
    # Volume momentum
    df['Volume_MA'] = df['volume'].rolling(window=20).mean()
    df['Volume_Momentum'] = df['volume'] / df['Volume_MA']
    
    return df

def calculate_strength(metrics, weights):
    normalized = {}
    for key in weights.keys():
        if key in metrics:
            value = metrics[key].iloc[-1]
            max_val = metrics[key].max()
            min_val = metrics[key].min()
            if key == 'Volatility':
                normalized[key] = 100 - (value / max_val * 100)
            else:
                normalized[key] = (value - min_val) / (max_val - min_val) * 100 if max_val != min_val else 50
    
    strength_score = sum(normalized[key] * weights[key] for key in weights.keys())
    return strength_score

# Main dashboard
st.title("Crypto Strength Dashboard")

st.sidebar.header("Strength Score Weights")
st.sidebar.write("Adjust the weights for different metrics (total will be normalized to 1)")

# Get weights from sliders
weights = {
    'RSI': st.sidebar.slider('RSI Weight', 0.0, 1.0, 0.25, 0.05),
    'MACD': st.sidebar.slider('MACD Weight', 0.0, 1.0, 0.25, 0.05),
    'Volatility': st.sidebar.slider('Volatility Weight', 0.0, 1.0, 0.25, 0.05),
    'Volume_Momentum': st.sidebar.slider('Volume Momentum Weight', 0.0, 1.0, 0.25, 0.05)
}

# Normalize weights
weight_sum = sum(weights.values())
weights = {k: v/weight_sum for k, v in weights.items()}

# Display normalized weights
st.sidebar.write("\nNormalized Weights:")
for k, v in weights.items():
    st.sidebar.write(f"{k}: {v:.3f}")

# Get available symbols
available_symbols = get_available_symbols()

# Symbol selection
st.subheader("Select Trading Pairs (Max 4)")
cols = st.columns(4)
selected_symbols = []

# Create 4 symbol input fields
for i in range(4):
    with cols[i]:
        symbol = st.text_input(f"Symbol {i+1} (e.g. BTC-USDT)", 
                             value="BTC-USDT" if i == 0 else "SOL-USDT" if i == 1 else "",
                             key=f"symbol_{i}")
        if symbol:
            # No need to convert format as we're now using dash format consistently
            if symbol in available_symbols:
                selected_symbols.append(symbol)
            else:
                st.error(f"Invalid symbol: {symbol}. Please use format: XXX-USDT")

# Remove duplicates and limit to 4
selected_symbols = list(dict.fromkeys(selected_symbols))[:4]

if selected_symbols:
    # Create dynamic columns based on number of selected symbols
    num_cols = min(len(selected_symbols), 2)
    num_rows = (len(selected_symbols) + 1) // 2
    
    # Fetch and process data for all selected symbols
    data_dict = {}
    for symbol in selected_symbols:
        data = fetch_crypto_data(
            symbol,
            selected_interval,
            LOOKBACK_PERIODS[selected_lookback],
            required_candles
        )
        if data is not None:
            data_dict[symbol] = calculate_metrics(data)

    # Display metrics in grid
    for row in range(num_rows):
        cols = st.columns(num_cols)
        for col in range(num_cols):
            idx = row * num_cols + col
            if idx < len(selected_symbols):
                symbol = selected_symbols[idx]
                if symbol in data_dict:
                    with cols[col]:
                        st.subheader(symbol)
                        data = data_dict[symbol]
                        strength = calculate_strength(data, weights)
                        st.metric("Current Price", f"${data['close'].iloc[-1]:,.2f}")
                        st.metric("Strength Score", f"{strength:.2f}")
                        
                        # Display metrics
                        st.write("Technical Indicators")
                        metrics = {
                            "RSI": data['RSI'].iloc[-1],
                            "MACD": data['MACD'].iloc[-1],
                            "Volatility (%)": data['Volatility'].iloc[-1],
                            "Volume Momentum": data['Volume_Momentum'].iloc[-1]
                        }
                        st.write(metrics)

    # Create comparison chart
    fig = make_subplots(rows=3, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.05,
                       subplot_titles=('Price Comparison', 'RSI', 'Volume'))
    
    # Add traces for each selected symbol
    colors = ['blue', 'red', 'green', 'purple']
    for i, symbol in enumerate(selected_symbols):
        if symbol in data_dict:
            data = data_dict[symbol]
            
            # Price (normalized to percentage change)
            fig.add_trace(
                go.Scatter(x=data['timestamp'], 
                          y=data['close']/data['close'].iloc[0]*100,
                          name=f'{symbol} Price',
                          line=dict(color=colors[i])),
                row=1, col=1
            )
            
            # RSI
            fig.add_trace(
                go.Scatter(x=data['timestamp'], 
                          y=data['RSI'],
                          name=f'{symbol} RSI',
                          line=dict(color=colors[i])),
                row=2, col=1
            )
            
            # Volume
            fig.add_trace(
                go.Bar(x=data['timestamp'], 
                      y=data['volume'],
                      name=f'{symbol} Volume',
                      marker_color=colors[i]),
                row=3, col=1
            )
    
    fig.update_layout(height=800, title_text="Comparative Analysis")
    st.plotly_chart(fig, use_container_width=True)

    # Add footer
    st.markdown("---")
    st.markdown("Data updates every minute. Last updated: " + 
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
else:
    st.warning("Please enter at least one valid trading pair")
