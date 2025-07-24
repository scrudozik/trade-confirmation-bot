import pandas as pd
import numpy as np
import yfinance as yf
import ta
import time
import matplotlib.pyplot as plt

# --- Default Bot Configuration ---
DEFAULT_TICKER = "AAPL"
ATR_PERIOD = 14

# --- Risk Multiplier Configuration ---
RISK_CONFIG = {
    "low": {"profit_multiplier": 1.5, "loss_multiplier": 1.0},
    "medium": {"profit_multiplier": 2.5, "loss_multiplier": 1.5},
    "high": {"profit_multiplier": 4.0, "loss_multiplier": 2.0}
}

# --- Data Fetching Function ---
def get_yfinance_data(ticker, period, interval):
    """
    Fetches historical data from Yahoo Finance based on the analysis timeframe.
    """
    print(f"Fetching {period} of {interval} data for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval, auto_adjust=True)
        if data.empty:
            print(f"No data found for {ticker}. It may be delisted, an invalid ticker, or no recent data is available for the requested interval.")
            return None

        print("Successfully fetched yfinance data.")
        data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True, errors='ignore')
        return data
    except Exception as e:
        print(f"Error fetching data from yfinance for {ticker}: {e}")
        return None

# --- Plotting Function ---
def plot_price_graph(data, ticker, period_name):
    """
    Displays a multi-panel chart with the price, Bollinger Bands, RSI, and MACD.
    """
    # Calculate indicators for plotting
    indicator_bb = ta.volatility.BollingerBands(close=data['close'], window=20, window_dev=2)
    data['bb_high'] = indicator_bb.bollinger_hband()
    data['bb_low'] = indicator_bb.bollinger_lband()
    data['bb_ma'] = indicator_bb.bollinger_mavg()

    data['rsi'] = ta.momentum.RSIIndicator(close=data['close'], window=14).rsi()

    macd = ta.trend.MACD(close=data['close'])
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['macd_hist'] = macd.macd_diff()

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    plt.style.use('seaborn-v0_8-darkgrid')

    # --- Plot 1: Price and Bollinger Bands ---
    ax1.plot(data.index, data['close'], label='Close Price', color='cyan', lw=2)
    ax1.plot(data.index, data['bb_high'], label='Upper Band', color='orange', linestyle='--', lw=1)
    ax1.plot(data.index, data['bb_low'], label='Lower Band', color='purple', linestyle='--', lw=1)
    ax1.plot(data.index, data['bb_ma'], label='20-Period SMA', color='yellow', linestyle=':', lw=1.5)
    ax1.fill_between(data.index, data['bb_low'], data['bb_high'], color='orange', alpha=0.1)
    ax1.set_title(f"{ticker} Price Analysis ({period_name})", fontsize=16)
    ax1.set_ylabel("Price (USD)", fontsize=12)
    ax1.legend()

    # Set tighter Y-axis limits for the price chart
    y_min = data['bb_low'].min() * 0.98
    y_max = data['bb_high'].max() * 1.02
    ax1.set_ylim(y_min, y_max)

    # --- Plot 2: RSI ---
    ax2.plot(data.index, data['rsi'], label='RSI', color='lightgreen')
    ax2.axhline(70, linestyle='--', color='red', alpha=0.7, lw=1)
    ax2.axhline(30, linestyle='--', color='blue', alpha=0.7, lw=1)
    ax2.set_ylabel("RSI", fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.legend()

    # --- Plot 3: MACD ---
    colors = ['green' if val >= 0 else 'red' for val in data['macd_hist']]
    ax3.bar(data.index, data['macd_hist'], label='Histogram', color=colors, alpha=0.6)
    ax3.plot(data.index, data['macd'], label='MACD', color='blue', lw=1.5)
    ax3.plot(data.index, data['macd_signal'], label='Signal Line', color='red', linestyle='--', lw=1.5)
    ax3.set_ylabel("MACD", fontsize=12)
    ax3.legend()

    plt.xlabel("Date", fontsize=12)
    plt.tight_layout()
    plt.show()


# --- Recommendation Engine ---
def generate_recommendation(data, signal, confidence, details, holding_period, risk_level):
    """
    Generates a detailed trade recommendation dictionary.
    """
    latest_price = data['close'].iloc[-1]

    profit_multiplier = RISK_CONFIG[risk_level]["profit_multiplier"]
    loss_multiplier = RISK_CONFIG[risk_level]["loss_multiplier"]

    if len(data) > ATR_PERIOD:
        atr = ta.volatility.AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=ATR_PERIOD).average_true_range().iloc[-1]
    else:
        atr = latest_price * 0.02

    recommendation = {
        "Ticker": data.attrs.get('ticker', 'N/A'),
        "Current Price": f"${latest_price:,.2f}",
        "Final Recommendation": signal,
        "Confidence": confidence,
        "Selected Holding Period": holding_period,
        "Selected Risk Level": risk_level.capitalize(),
        "Entry Price": "N/A",
        "Target Price (Profit Exit)": "N/A",
        "Stop-Loss (Loss Exit)": "N/A",
        "Indicator Analysis": details
    }

    if "BUY" in signal:
        recommendation["Entry Price"] = f"~${latest_price:,.2f}"
        recommendation["Target Price (Profit Exit)"] = f"${latest_price + (atr * profit_multiplier):,.2f}"
        recommendation["Stop-Loss (Loss Exit)"] = f"${latest_price - (atr * loss_multiplier):,.2f}"
    elif "SELL" in signal:
        recommendation["Entry Price"] = f"~${latest_price:,.2f}"
        recommendation["Target Price (Profit Exit)"] = f"${latest_price - (atr * profit_multiplier):,.2f}"
        recommendation["Stop-Loss (Loss Exit)"] = f"${latest_price + (atr * loss_multiplier):,.2f}"

    return recommendation

# --- Technical Indicator Strategies ---

def moving_average_crossover_strategy(data):
    short_window, long_window = 20, 50
    if len(data) < long_window: return "HOLD"
    data['short_mavg'] = ta.trend.sma_indicator(data['close'], window=short_window)
    data['long_mavg'] = ta.trend.sma_indicator(data['close'], window=long_window)
    if data['short_mavg'].iloc[-1] > data['long_mavg'].iloc[-1] and data['short_mavg'].iloc[-2] <= data['long_mavg'].iloc[-2]:
        return "BUY"
    elif data['short_mavg'].iloc[-1] < data['long_mavg'].iloc[-1] and data['short_mavg'].iloc[-2] >= data['long_mavg'].iloc[-2]:
        return "SELL"
    return "HOLD"

def rsi_strategy(data):
    if len(data) < 15: return "HOLD"
    rsi = ta.momentum.RSIIndicator(data['close'], window=14).rsi()
    if rsi.iloc[-1] < 30: return "BUY"
    if rsi.iloc[-1] > 70: return "SELL"
    return "HOLD"

def bollinger_bands_strategy(data):
    if len(data) < 20: return "HOLD"
    indicator_bb = ta.volatility.BollingerBands(close=data['close'], window=20, window_dev=2)
    if data['close'].iloc[-1] < indicator_bb.bollinger_lband().iloc[-1]: return "BUY"
    if data['close'].iloc[-1] > indicator_bb.bollinger_hband().iloc[-1]: return "SELL"
    return "HOLD"

def macd_strategy(data):
    if len(data) < 26: return "HOLD"
    macd_diff = ta.trend.MACD(data['close']).macd_diff()
    if macd_diff.iloc[-1] > 0 and macd_diff.iloc[-2] <= 0: return "BUY"
    if macd_diff.iloc[-1] < 0 and macd_diff.iloc[-2] >= 0: return "SELL"
    return "HOLD"

def stochastic_oscillator_strategy(data):
    if len(data) < 14: return "HOLD"
    stoch = ta.momentum.StochasticOscillator(high=data['high'], low=data['low'], close=data['close']).stoch()
    if stoch.iloc[-1] < 20: return "BUY"
    if stoch.iloc[-1] > 80: return "SELL"
    return "HOLD"

def obv_strategy(data):
    if 'volume' not in data.columns or len(data) < 20: return "HOLD"
    obv = ta.volume.OnBalanceVolumeIndicator(close=data['close'], volume=data['volume']).on_balance_volume()
    obv_sma = ta.trend.sma_indicator(obv, window=20)
    if obv.iloc[-1] > obv_sma.iloc[-1]: return "BUY"
    if obv.iloc[-1] < obv_sma.iloc[-1]: return "SELL"
    return "HOLD"

def ichimoku_strategy(data):
    if len(data) < 52: return "HOLD"
    ichimoku = ta.trend.IchimokuIndicator(high=data['high'], low=data['low'])
    price = data['close'].iloc[-1]
    cloud_a = ichimoku.ichimoku_a().iloc[-1]
    cloud_b = ichimoku.ichimoku_b().iloc[-1]
    if price > cloud_a and price > cloud_b: return "BUY"
    if price < cloud_a and price < cloud_b: return "SELL"
    return "HOLD"

def vwap_strategy(data):
    if 'volume' not in data.columns or len(data) < 2: return "HOLD"
    vwap = ta.volume.VolumeWeightedAveragePrice(high=data['high'], low=data['low'], close=data['close'], volume=data['volume']).volume_weighted_average_price()
    if data['close'].iloc[-1] > vwap.iloc[-1]: return "BUY"
    if data['close'].iloc[-1] < vwap.iloc[-1]: return "SELL"
    return "HOLD"

def adx_strategy(data):
    if len(data) < 28: return "HOLD"
    adx_indicator = ta.trend.ADXIndicator(high=data['high'], low=data['low'], close=data['close'])
    if adx_indicator.adx().iloc[-1] > 25 and adx_indicator.adx_pos().iloc[-1] > adx_indicator.adx_neg().iloc[-1]: return "BUY"
    if adx_indicator.adx().iloc[-1] > 25 and adx_indicator.adx_neg().iloc[-1] > adx_indicator.adx_pos().iloc[-1]: return "SELL"
    return "HOLD"

def awesome_oscillator_strategy(data):
    if len(data) < 34: return "HOLD"
    ao = ta.momentum.AwesomeOscillatorIndicator(high=data['high'], low=data['low']).awesome_oscillator()
    if ao.iloc[-1] > 0 and ao.iloc[-2] <= 0: return "BUY"
    if ao.iloc[-1] < 0 and ao.iloc[-2] >= 0: return "SELL"
    return "HOLD"

def mfi_strategy(data):
    if 'volume' not in data.columns or len(data) < 14: return "HOLD"
    mfi = ta.volume.MFIIndicator(high=data['high'], low=data['low'], close=data['close'], volume=data['volume']).money_flow_index()
    if mfi.iloc[-1] < 20: return "BUY"
    if mfi.iloc[-1] > 80: return "SELL"
    return "HOLD"

def williams_r_strategy(data):
    if len(data) < 14: return "HOLD"
    wr = ta.momentum.WilliamsRIndicator(high=data['high'], low=data['low'], close=data['close']).williams_r()
    if wr.iloc[-1] < -80: return "BUY"
    if wr.iloc[-1] > -20: return "SELL"
    return "HOLD"

# --- Main Bot Logic ---
def run_analysis_for_scenario(data, all_strategies, holding_period_name, risk_level):
    """
    Runs a single analysis for one combination of data and risk.
    Returns the recommendation dictionary.
    """
    data.attrs['ticker'] = data.attrs.get('ticker', 'N/A')

    signals = [func(data.copy()) for name, func in all_strategies]
    indicator_details = [{"Indicator": name, "Signal": signal} for (name, func), signal in zip(all_strategies, signals)]

    buy_count = signals.count("BUY")
    sell_count = signals.count("SELL")
    total_signals = len(signals)

    final_signal = "HOLD"
    confidence = "Low"

    if buy_count > sell_count and buy_count >= total_signals / 2:
        final_signal = "BUY"
        confidence = f"High ({buy_count}/{total_signals})"
    elif sell_count > buy_count and sell_count >= total_signals / 2:
        final_signal = "SELL"
        confidence = f"High ({sell_count}/{total_signals})"
    elif buy_count > sell_count:
        final_signal = "WEAK BUY"
        confidence = f"Low ({buy_count}/{total_signals})"
    elif sell_count > buy_count:
        final_signal = "WEAK SELL"
        confidence = f"Low ({sell_count}/{total_signals})"

    return generate_recommendation(data, final_signal, confidence, indicator_details, holding_period_name, risk_level)

if __name__ == "__main__":
    all_strategies = [
        ("Moving Avg Crossover", moving_average_crossover_strategy), ("RSI", rsi_strategy),
        ("Bollinger Bands", bollinger_bands_strategy), ("MACD", macd_strategy),
        ("Stochastic Oscillator", stochastic_oscillator_strategy), ("On-Balance Volume", obv_strategy),
        ("Ichimoku Cloud", ichimoku_strategy), ("VWAP", vwap_strategy),
        ("ADX", adx_strategy), ("Awesome Oscillator", awesome_oscillator_strategy),
        ("Money Flow Index", mfi_strategy), ("Williams %R", williams_r_strategy),
    ]

    holding_map = {
        "Short-Term": {"period": "1mo", "interval": "1h"},
        "Medium-Term": {"period": "6mo", "interval": "1d"},
        "Long-Term": {"period": "2y", "interval": "1d"}
    }
    risk_levels = ["low", "medium", "high"]

    while True:
        ticker_input = input(f"Enter a ticker symbol (e.g., {DEFAULT_TICKER}, BTC-USD) or type 'exit' to quit: ")
        if ticker_input.lower() == 'exit': break
        ticker = ticker_input.strip().upper()

        results_matrix = {period_name: {} for period_name in holding_map.keys()}
        full_details = None
        data_for_plotting = None

        print(f"\n--- Running Full Spectrum Analysis for {ticker} ---")

        for period_name, params in holding_map.items():
            data = get_yfinance_data(ticker, params["period"], params["interval"])
            if data is None:
                print(f"Skipping analysis for {period_name} due to data fetching issues.")
                for risk in risk_levels:
                    results_matrix[period_name][risk.capitalize()] = "DATA FAILED"
                continue

            data.attrs['ticker'] = ticker

            for risk in risk_levels:
                recommendation = run_analysis_for_scenario(data, all_strategies, period_name, risk)
                results_matrix[period_name][risk.capitalize()] = recommendation["Final Recommendation"]

                if full_details is None:
                    full_details = recommendation
                    data_for_plotting = data.copy()

                if period_name == "Medium-Term" and risk == "medium":
                    full_details = recommendation
                    data_for_plotting = data.copy()

        # --- Print the final results matrix ---
        print("\n" + "="*70)
        print(f" STRATEGY MATRIX FOR {ticker} ".center(70, "="))
        print("="*70)

        results_df = pd.DataFrame(results_matrix).T
        results_df.index.name = "Holding Period"
        results_df.columns.name = "Risk Tolerance"
        print(results_df)
        print("="*70)

        # --- Print the detailed breakdown ---
        if full_details:
            print("\n" + "-"*70)
            breakdown_title = f" DETAILED BREAKDOWN FOR {full_details['Selected Holding Period'].upper()} / {full_details['Selected Risk Level'].upper()}-RISK "
            print(breakdown_title.center(70, "-"))
            print("-"*70)
            print(f"Current Price: {full_details['Current Price']}")
            print(f"Target Price (Profit Exit): {full_details['Target Price (Profit Exit)']}")
            print(f"Stop-Loss (Loss Exit): {full_details['Stop-Loss (Loss Exit)']}")
            print("\n--- Individual Indicator Results ---")
            details_df = pd.DataFrame(full_details["Indicator Analysis"])
            print(details_df.to_string(index=False))
            print("-"*70)

        # --- Show the plot ---
        if data_for_plotting is not None:
            print("\nDisplaying price chart... Close the chart window to continue.")
            plot_price_graph(data_for_plotting, ticker, full_details['Selected Holding Period'])

        print("\nAnalysis complete. You can now analyze another stock or exit.")
