
"""
Professional Trading Analysis System v3.0
==========================================
Integrated trading signal generator combining technical, fundamental, and sentiment analysis.
Supports stocks and cryptocurrencies. Recommends trade ideas with:
- Short-term holds (<24 hours) using intraday data, less aggressive.
- 2% stop loss.
- Entry price (current close).
- Exit price targeting at least 25% gain for buys (ambitious, especially for stocks).
- Multi-timeframe analysis: short (<24h), medium (days-weeks), long (months+).
- Uses yfinance for data, TextBlob for sentiment.
- Added backtesting functionality based on technical signals.
- Integrated machine learning using PyTorch LSTM for price prediction to enhance signals.

Note: 25% gain in <24h is high-risk/high-reward; more realistic for volatile assets like crypto.
Note: ML integration uses a simple LSTM model for forecasting next prices, influencing recommendations.
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import mplfinance as mpf
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
from pathlib import Path
import logging
import requests
from textblob import TextBlob
import time
import matplotlib.pyplot as plt
import re
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# ==================== ENUMS & DATA CLASSES ====================

class Signal(Enum):
    """Trading signal enumeration"""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2

class Sentiment(Enum):
    """Market sentiment enumeration"""
    VERY_BULLISH = 2
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1
    VERY_BEARISH = -2

@dataclass
class TechnicalMetrics:
    """Technical analysis metrics"""
    trend_score: float
    momentum_score: float
    volatility_score: float
    volume_score: float
    overall_score: float
    signal: Signal
    indicators: Dict

@dataclass
class FundamentalMetrics:
    """Fundamental analysis metrics"""
    valuation_score: float
    profitability_score: float
    growth_score: float
    health_score: float
    overall_score: float
    metrics: Dict

@dataclass
class SentimentMetrics:
    """Sentiment analysis metrics"""
    news_sentiment: float
    sentiment_label: Sentiment
    article_count: int
    articles: List[Dict]

@dataclass
class TradeRecommendation:
    """Trade recommendation with risk management"""
    signal: Signal
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float
    risk_reward_ratio: float
    timeframe: str
    reasoning: str
    predicted_price: float = 0.0  # Added for ML prediction

# ==================== CONFIGURATION ====================

class Config:
    """System configuration"""
    TIMEFRAMES = {
        "short": {"interval": "5m", "period": "7d"},  # <24h holds, intraday
        "medium": {"interval": "1h", "period": "730d"},  # Days-weeks
        "long": {"interval": "1d", "period": "10y"}  # Months+
    }
    PRIMARY_TIMEFRAME = "medium"

    # Technical settings
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    ADX_STRONG_TREND = 25

    # Risk management
    DEFAULT_STOP_LOSS_PCT = 0.02  # 2%
    TARGET_GAIN_PCT = 0.25  # 25% minimum per trade
    DEFAULT_RISK_REWARD = 12.5  # 25% / 2% = 12.5 RR
    MAX_POSITION_SIZE = 0.25  # 25% of portfolio
    HOLDING_PERIOD_SHORT_HOURS = 24  # Less than 24h for short-term

    # ML settings
    SEQ_LENGTH = 60
    HIDDEN_SIZE = 50
    NUM_LAYERS = 1
    LEARNING_RATE = 0.001
    EPOCHS = 10  # Small for quick training

    # Caching
    CACHE_DIR = Path("cache")
    CACHE_EXPIRY_HOURS = 1

    # Export
    EXPORT_DIR = Path("reports")

    @classmethod
    def initialize(cls):
        """Initialize directories"""
        cls.CACHE_DIR.mkdir(exist_ok=True)
        cls.EXPORT_DIR.mkdir(exist_ok=True)

# ==================== ML PREDICTOR ====================

class LSTM(nn.Module):
    """LSTM model for price prediction"""
    def __init__(self, input_size=1, hidden_size=Config.HIDDEN_SIZE, num_layers=Config.NUM_LAYERS, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class TimeSeriesDataset(Dataset):
    """Dataset for time series"""
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLPredictor:
    """Machine Learning predictor using LSTM"""
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = LSTM()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def prepare_data(self, series: pd.Series, seq_length: int = Config.SEQ_LENGTH):
        """Prepare sequences"""
        data = series.values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(len(scaled_data) - seq_length):
            X.append(scaled_data[i:i+seq_length])
            y.append(scaled_data[i+seq_length])

        X = np.array(X)
        y = np.array(y)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def train(self, series: pd.Series):
        """Train the model on historical data"""
        X, y = self.prepare_data(series)

        if len(X) == 0:
            logger.warning("Insufficient data for training")
            return

        dataset = TimeSeriesDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

        self.model.train()
        for epoch in range(Config.EPOCHS):
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logger.info(f"Epoch [{epoch+1}/{Config.EPOCHS}], Loss: {loss.item():.4f}")

    def predict(self, series: pd.Series, steps: int = 1) -> float:
        """Predict next price"""
        self.model.eval()
        with torch.no_grad():
            last_seq = self.scaler.transform(series[-Config.SEQ_LENGTH:].values.reshape(-1, 1))
            last_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(self.device)

            predicted = self.model(last_seq)
            predicted = self.scaler.inverse_transform(predicted.cpu().numpy())

            return predicted[0][0]

# ==================== DATA FETCHER ====================

class DataFetcher:
    """Handles all data fetching operations with caching"""

    @staticmethod
    def get_period_for_interval(interval: str) -> str:
        """Returns appropriate data period for interval"""
        period_map = {
            "1mo": "30y", "1wk": "10y", "1d": "10y",
            "4h": "730d", "1h": "730d", "30m": "60d",
            "15m": "60d", "5m": "7d", "1m": "7d"
        }
        return period_map.get(interval, "1y")

    @staticmethod
    def fetch_price_data(ticker: str, period: str, interval: str = "1d") -> Optional[pd.DataFrame]:
        """Fetch historical price data"""
        try:
            logger.info(f"Fetching {interval} data for {ticker}...")

            fetch_interval = "1h" if interval == "4h" else interval
            df = yf.Ticker(ticker).history(period=period, interval=fetch_interval)

            if df.empty:
                logger.warning(f"No data found for {ticker}")
                return None

            df.columns = df.columns.str.lower()

            if 'volume' not in df.columns:
                df['volume'] = 0

            # Resample 4h if needed
            if interval == "4h":
                df = df.resample('4H').agg({
                    'open': 'first', 'high': 'max', 'low': 'min',
                    'close': 'last', 'volume': 'sum'
                }).dropna()

            return df
        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            return None

    @staticmethod
    def fetch_fundamental_data(ticker: str) -> Optional[Dict]:
        """Fetch fundamental data"""
        try:
            logger.info(f"Fetching fundamentals for {ticker}...")
            stock = yf.Ticker(ticker)
            info = stock.info

            return {
                'company_name': info.get('longName', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'return_on_equity': info.get('returnOnEquity'),
                'return_on_assets': info.get('returnOnAssets'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'free_cashflow': info.get('freeCashflow'),
                'revenue': info.get('totalRevenue'),
                'earnings_per_share': info.get('trailingEps'),
                'beta': info.get('beta'),
                '52w_high': info.get('fiftyTwoWeekHigh'),
                '52w_low': info.get('fiftyTwoWeekLow'),
                'analyst_recommendation': info.get('recommendationKey', 'N/A'),
                'target_price': info.get('targetMeanPrice'),
                'description': info.get('longBusinessSummary', 'N/A')
            }
        except Exception as e:
            logger.error(f"Error fetching fundamental data: {e}")
            return None

    @staticmethod
    def fetch_news_sentiment(ticker: str) -> SentimentMetrics:
        """Fetch recent news and perform sentiment analysis"""
        logger.info(f"Fetching news sentiment for {ticker}...")
        sentiments = []
        articles = []

        try:
            stock = yf.Ticker(ticker)
            news = stock.news

            if news:
                for item in news[:10]:  # Up to 10 recent articles
                    title = item.get('title', '')
                    summary = item.get('summary', '')
                    text = title + " " + summary
                    sentiment = TextBlob(text).sentiment.polarity
                    sentiments.append(sentiment)

                    articles.append({
                        'title': title,
                        'publisher': item.get('publisher', 'Unknown'),
                        'link': item.get('link', ''),
                        'sentiment': sentiment
                    })

            avg_sentiment = np.mean(sentiments) if sentiments else 0
            article_count = len(articles)

            if avg_sentiment > 0.2:
                label = Sentiment.VERY_BULLISH
            elif avg_sentiment > 0.05:
                label = Sentiment.BULLISH
            elif avg_sentiment < -0.2:
                label = Sentiment.VERY_BEARISH
            elif avg_sentiment < -0.05:
                label = Sentiment.BEARISH
            else:
                label = Sentiment.NEUTRAL

            return SentimentMetrics(
                news_sentiment=avg_sentiment,
                sentiment_label=label,
                article_count=article_count,
                articles=articles
            )
        except Exception as e:
            logger.error(f"Error fetching news sentiment: {e}")
            return SentimentMetrics(0, Sentiment.NEUTRAL, 0, [])

# ==================== TECHNICAL ANALYZER ====================

class TechnicalAnalyzer:
    """Performs technical analysis"""

    def __init__(self):
        pass

    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        indicators = {}

        try:
            # Moving Averages
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['sma_50'] = ta.sma(df['close'], length=50)
            df['ema_12'] = ta.ema(df['close'], length=12)
            df['ema_26'] = ta.ema(df['close'], length=26)
            df['hma_50'] = ta.hma(df['close'], length=50)

            # MACD
            macd = ta.macd(df['close'])
            df = pd.concat([df, macd], axis=1)

            # RSI
            df['rsi_14'] = ta.rsi(df['close'])

            # Bollinger Bands
            bb = ta.bbands(df['close'], length=20)
            df = pd.concat([df, bb], axis=1)

            # ATR
            df['atr_14'] = ta.atr(df['high'], df['low'], df['close'])

            # Stochastic
            stoch = ta.stoch(df['high'], df['low'], df['close'])
            df = pd.concat([df, stoch], axis=1)

            # OBV
            df['obv'] = ta.obv(df['close'], df['volume'])

            # ADX
            adx = ta.adx(df['high'], df['low'], df['close'])
            df = pd.concat([df, adx], axis=1)

            # Williams %R
            df['willr_14'] = ta.willr(df['high'], df['low'], df['close'])

            # CCI
            df['cci_20'] = ta.cci(df['high'], df['low'], df['close'])

            # CMF
            df['cmf_20'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'])

            # MFI
            df['mfi_14'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'])

            # VWAP
            df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])

            df.dropna(inplace=True)

            indicators = df.iloc[-1].to_dict()

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")

        return indicators

    def calculate_scores(self, indicators: Dict) -> TechnicalMetrics:
        """Calculate technical scores"""
        trend_score = 0
        momentum_score = 0
        volatility_score = 0
        volume_score = 0

        # Trend
        if indicators.get('ema_12') > indicators.get('ema_26'):
            trend_score += 1
        if indicators.get('hma_50') and indicators['close'] > indicators['hma_50']:
            trend_score += 1
        if indicators.get('ADX_14') > Config.ADX_STRONG_TREND:
            trend_score += 1 if indicators.get('DMP_14') > indicators.get('DMN_14') else -1

        # Momentum
        if indicators.get('rsi_14') < Config.RSI_OVERSOLD:
            momentum_score += 1
        elif indicators.get('rsi_14') > Config.RSI_OVERBOUGHT:
            momentum_score -= 1
        if indicators.get('MACDh_12_26_9') > 0:
            momentum_score += 1
        if indicators.get('STOCHk_14_3_3') < 20:
            momentum_score += 1
        elif indicators.get('STOCHk_14_3_3') > 80:
            momentum_score -= 1

        # Volatility
        if indicators['close'] < indicators.get('BBL_20_2.0'):
            volatility_score += 1
        elif indicators['close'] > indicators.get('BBU_20_2.0'):
            volatility_score -= 1

        # Volume
        if indicators.get('obv') > indicators.get('obv', 0):  # Simplified
            volume_score += 1
        if indicators.get('mfi_14') < 20:
            volume_score += 1
        elif indicators.get('mfi_14') > 80:
            volume_score -= 1

        overall_score = (trend_score + momentum_score + volatility_score + volume_score) / 4.0

        if overall_score > 0.5:
            signal = Signal.STRONG_BUY if overall_score > 1.5 else Signal.BUY
        elif overall_score < -0.5:
            signal = Signal.STRONG_SELL if overall_score < -1.5 else Signal.SELL
        else:
            signal = Signal.HOLD

        return TechnicalMetrics(
            trend_score=trend_score,
            momentum_score=momentum_score,
            volatility_score=volatility_score,
            volume_score=volume_score,
            overall_score=overall_score,
            signal=signal,
            indicators=indicators
        )

# ==================== FUNDAMENTAL ANALYZER ====================

class FundamentalAnalyzer:
    """Performs fundamental analysis"""

    def calculate_scores(self, fundamentals: Dict) -> FundamentalMetrics:
        """Calculate fundamental scores"""
        valuation_score = 0
        profitability_score = 0
        growth_score = 0
        health_score = 0

        # Valuation
        if fundamentals.get('pe_ratio') and fundamentals['pe_ratio'] < 15:
            valuation_score += 1
        if fundamentals.get('forward_pe') and fundamentals['forward_pe'] < 15:
            valuation_score += 1
        if fundamentals.get('peg_ratio') and fundamentals['peg_ratio'] < 1:
            valuation_score += 1

        # Profitability
        if fundamentals.get('profit_margin') and fundamentals['profit_margin'] > 0.2:
            profitability_score += 1
        if fundamentals.get('return_on_equity') and fundamentals['return_on_equity'] > 0.15:
            profitability_score += 1

        # Growth
        if fundamentals.get('revenue_growth') and fundamentals['revenue_growth'] > 0.1:
            growth_score += 1
        if fundamentals.get('earnings_growth') and fundamentals['earnings_growth'] > 0.1:
            growth_score += 1

        # Health
        if fundamentals.get('current_ratio') and fundamentals['current_ratio'] > 1.5:
            health_score += 1
        if fundamentals.get('debt_to_equity') and fundamentals['debt_to_equity'] < 1:
            health_score += 1

        overall_score = (valuation_score + profitability_score + growth_score + health_score) / 4.0

        return FundamentalMetrics(
            valuation_score=valuation_score,
            profitability_score=profitability_score,
            growth_score=growth_score,
            health_score=health_score,
            overall_score=overall_score,
            metrics=fundamentals
        )

# ==================== STRATEGY ENGINE ====================

class StrategyEngine:
    """Generates trading recommendations"""

    def generate_recommendation(self, ticker: str, technical: TechnicalMetrics, fundamental: FundamentalMetrics,
                                sentiment: SentimentMetrics, current_price: float, timeframe: str, predicted_price: float) -> TradeRecommendation:
        """Generate trade recommendation with ML prediction"""
        confidence = (technical.overall_score + fundamental.overall_score + sentiment.news_sentiment) / 3.0

        combined_score = technical.overall_score * 0.4 + fundamental.overall_score * 0.3 + sentiment.news_sentiment * 0.2

        # Incorporate ML prediction
        predicted_change = (predicted_price - current_price) / current_price
        if predicted_change > Config.TARGET_GAIN_PCT:
            combined_score += 1.0
        elif predicted_change > 0.05:
            combined_score += 0.5
        elif predicted_change < -0.05:
            combined_score -= 0.5

        if combined_score > 0.5:
            signal = Signal.STRONG_BUY if combined_score > 1.5 else Signal.BUY
        elif combined_score < -0.5:
            signal = Signal.STRONG_SELL if combined_score < -1.5 else Signal.SELL
        else:
            signal = Signal.HOLD

        # Adjust for timeframe: short-term less aggressive
        if timeframe == "short" and signal in [Signal.STRONG_BUY, Signal.BUY]:
            signal = Signal.BUY

        # Risk management
        stop_loss_pct = Config.DEFAULT_STOP_LOSS_PCT
        target_gain_pct = max(Config.TARGET_GAIN_PCT, predicted_change) if predicted_change > Config.TARGET_GAIN_PCT else Config.TARGET_GAIN_PCT

        if signal in [Signal.BUY, Signal.STRONG_BUY]:
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + target_gain_pct)
        elif signal in [Signal.SELL, Signal.STRONG_SELL]:
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 - target_gain_pct)
        else:
            stop_loss = take_profit = current_price

        rr_ratio = target_gain_pct / stop_loss_pct if stop_loss_pct > 0 else 0

        reasoning = f"Technical: {technical.signal}, Fundamental: {fundamental.overall_score:.2f}, Sentiment: {sentiment.sentiment_label}. "
        reasoning += f"ML Predicted Price: ${predicted_price:.2f} (change: {predicted_change*100:.1f}%). "
        reasoning += f"Target 25% gain in <24h for short-term (high risk)."

        return TradeRecommendation(
            signal=signal,
            confidence=abs(confidence),
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=Config.MAX_POSITION_SIZE,
            risk_reward_ratio=rr_ratio,
            timeframe=timeframe,
            reasoning=reasoning,
            predicted_price=predicted_price
        )

# ==================== NARRATIVE GENERATOR ====================

class NarrativeGenerator:
    """Generates investment narratives"""

    def generate(self, ticker: str, technical: TechnicalMetrics, fundamental: FundamentalMetrics,
                 sentiment: SentimentMetrics, recommendation: TradeRecommendation) -> str:
        """Generate narrative"""
        narrative = f"### Investment Analysis for {ticker} ({recommendation.timeframe.capitalize()} Timeframe)\n\n"

        narrative += "#### Technical Summary\n"
        narrative += f"Overall Score: {technical.overall_score:.2f}\n"
        narrative += f"Signal: {technical.signal.name}\n"
        narrative += f"Key Indicators: RSI {technical.indicators.get('rsi_14', 'N/A'):.2f}, "
        narrative += f"MACD Histogram {technical.indicators.get('MACDh_12_26_9', 'N/A'):.2f}\n\n"

        narrative += "#### Fundamental Summary\n"
        narrative += f"Overall Score: {fundamental.overall_score:.2f}\n"
        narrative += f"PE Ratio: {fundamental.metrics.get('pe_ratio', 'N/A')}\n"
        narrative += f"Revenue Growth: {fundamental.metrics.get('revenue_growth', 'N/A')*100:.1f}%\n\n"

        narrative += "#### Sentiment Summary\n"
        narrative += f"Sentiment Score: {sentiment.news_sentiment:.2f} ({sentiment.sentiment_label.name})\n"
        narrative += f"Based on {sentiment.article_count} recent articles.\n\n"

        narrative += "#### ML Prediction\n"
        narrative += f"Predicted Next Price: ${recommendation.predicted_price:.2f}\n\n"

        narrative += "#### Trade Recommendation\n"
        narrative += f"Signal: {recommendation.signal.name}\n"
        narrative += f"Confidence: {recommendation.confidence:.2f}\n"
        narrative += f"Entry: ${recommendation.entry_price:.2f}\n"
        narrative += f"Stop Loss: ${recommendation.stop_loss:.2f} (2%)\n"
        narrative += f"Take Profit: ${recommendation.take_profit:.2f} (25%+ gain)\n"
        narrative += f"Risk/Reward: {recommendation.risk_reward_ratio:.1f}:1\n"
        narrative += f"Reasoning: {recommendation.reasoning}\n"

        if recommendation.timeframe == "short":
            narrative += "\nNote: Short-term trade (<24h) is high-risk; monitor closely for 25% target."

        return narrative

# ==================== VISUALIZER ====================

class Visualizer:
    """Handles visualizations"""

    def plot_technical_chart(self, df: pd.DataFrame, ticker: str):
        """Plot technical chart"""
        try:
            ap = [
                mpf.make_addplot(df['sma_20'], color='blue'),
                mpf.make_addplot(df['sma_50'], color='orange'),
                mpf.make_addplot(df['rsi_14'], panel=1, color='purple', ylabel='RSI'),
                mpf.make_addplot([70] * len(df), panel=1, color='r', linestyle='--'),
                mpf.make_addplot([30] * len(df), panel=1, color='g', linestyle='--'),
                mpf.make_addplot(df['MACD_12_26_9'], panel=2, color='fuchsia', ylabel='MACD'),
                mpf.make_addplot(df['MACDs_12_26_9'], panel=2, color='c', linestyle='--'),
                mpf.make_addplot(df['MACDh_12_26_9'], panel=2, type='bar', color='gray', alpha=0.5)
            ]

            mpf.plot(df, type='candle', style='yahoo', title=f'{ticker} Chart',
                     ylabel='Price', volume=True, addplot=ap, panel_ratios=(6,2,2))
        except Exception as e:
            logger.error(f"Error plotting chart: {e}")

# ==================== EXPORTER ====================

class Exporter:
    """Handles exports"""

    def export_json(self, data: Dict, ticker: str, timestamp: str):
        """Export to JSON"""
        filename = Config.EXPORT_DIR / f"{ticker}_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Exported JSON to {filename}")

    def export_text(self, text: str, ticker: str, timestamp: str):
        """Export to text"""
        filename = Config.EXPORT_DIR / f"{ticker}_{timestamp}.txt"
        with open(filename, 'w') as f:
            f.write(text)
        logger.info(f"Exported text to {filename}")

# ==================== BACKTESTER ====================

class Backtester:
    """Performs backtesting of strategies"""

    def __init__(self, df: pd.DataFrame, initial_capital: float = 100000.0):
        self.df = df
        self.initial_capital = initial_capital
        self.positions = []
        self.trades = []

    def run(self, strategy_func):
        """Run backtest with given strategy function"""
        capital = self.initial_capital
        position = 0

        for i in range(1, len(self.df)):
            signal = strategy_func(self.df.iloc[:i])

            current_price = self.df['close'].iloc[i]

            if signal == Signal.BUY and position == 0:
                shares = capital // current_price
                if shares > 0:
                    position = shares
                    capital -= shares * current_price
                    self.trades.append({'date': self.df.index[i], 'action': 'buy', 'price': current_price, 'shares': shares})

            elif signal == Signal.SELL and position > 0:
                capital += position * current_price
                self.trades.append({'date': self.df.index[i], 'action': 'sell', 'price': current_price, 'shares': position})
                position = 0

        # Close any open position
        if position > 0:
            capital += position * self.df['close'].iloc[-1]

        return {
            'final_capital': capital,
            'return_pct': (capital - self.initial_capital) / self.initial_capital * 100,
            'trades': self.trades
        }

# ==================== MARKET ANALYZER ====================

class MarketAnalyzer:
    """Main analysis class"""

    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.technical_analyzer = TechnicalAnalyzer()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.ml_predictor = MLPredictor()
        self.strategy_engine = StrategyEngine()
        self.narrative_generator = NarrativeGenerator()
        self.visualizer = Visualizer()
        self.exporter = Exporter()

    def analyze(self, ticker: str, export: bool = True, backtest: bool = False) -> Dict:
        """Analyze single ticker across timeframes"""
        Config.initialize()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {'ticker': ticker, 'analyses': {}}

        # Fundamentals and sentiment (once)
        fundamentals = self.data_fetcher.fetch_fundamental_data(ticker) or {}
        fundamental = self.fundamental_analyzer.calculate_scores(fundamentals)
        sentiment = self.data_fetcher.fetch_news_sentiment(ticker)

        for tf_name, tf_params in Config.TIMEFRAMES.items():
            period = tf_params["period"]
            interval = tf_params["interval"]

            df = self.data_fetcher.fetch_price_data(ticker, period, interval)
            if df is None or df.empty:
                continue

            # Train ML on data
            if len(df) > Config.SEQ_LENGTH * 2:  # Enough data
                self.ml_predictor.train(df['close'])
                predicted_price = self.ml_predictor.predict(df['close'])
            else:
                predicted_price = df['close'].iloc[-1]
                logger.warning(f"Insufficient data for ML prediction in {tf_name}")

            indicators = self.technical_analyzer.calculate_indicators(df)
            technical = self.technical_analyzer.calculate_scores(indicators)

            try:
                current_price = float(df['close'].iloc[-1])
            except:
                continue

            recommendation = self.strategy_engine.generate_recommendation(
                ticker, technical, fundamental, sentiment, current_price, tf_name, predicted_price
            )

            narrative = self.narrative_generator.generate(
                ticker, technical, fundamental, sentiment, recommendation
            )

            results['analyses'][tf_name] = {
                'technical': asdict(technical),
                'fundamental': asdict(fundamental),
                'sentiment': asdict(sentiment),
                'recommendation': asdict(recommendation),
                'narrative': narrative
            }

            print(f"\n--- {tf_name.upper()} Timeframe ---\n")
            print(narrative)

            if len(df) > 0:
                self.visualizer.plot_technical_chart(df.tail(126), ticker)

            if backtest:
                backtester = Backtester(df)
                def simple_strategy(history_df):
                    last_indicators = self.technical_analyzer.calculate_indicators(history_df).get('close', history_df['close'].iloc[-1])
                    metrics = self.technical_analyzer.calculate_scores(last_indicators)
                    return metrics.signal

                bt_results = backtester.run(simple_strategy)
                print(f"\nBacktest Results for {tf_name}:")
                print(f"Final Capital: ${bt_results['final_capital']:.2f}")
                print(f"Return: {bt_results['return_pct']:.2f}%")
                results['analyses'][tf_name]['backtest'] = bt_results

        if export:
            self.exporter.export_json(results, ticker, timestamp)

        return results

    def batch_analyze(self, tickers: List[str]) -> List[Dict]:
        """Analyze multiple tickers"""
        results = []
        for ticker in tickers:
            result = self.analyze(ticker)
            results.append(result)
        return results

# ==================== CLI INTERFACE ====================

def main():
    """Main CLI interface"""
    print("=" * 80)
    print("PROFESSIONAL TRADING ANALYSIS SYSTEM v3.0 with ML Integration")
    print("=" * 80)
    print("\nFeatures:")
    print("  • Multi-timeframe analysis (short <24h, medium, long)")
    print("  • Technical, fundamental, and sentiment analysis")
    print("  • ML price prediction with LSTM")
    print("  • 2% stop loss, 25% target gain (adjusted by ML)")
    print("  • Backtesting option")
    print("  • Automated report generation")
    print("=" * 80)

    analyzer = MarketAnalyzer()

    while True:
        print("\nOptions:")
        print("1. Analyze single ticker")
        print("2. Analyze multiple tickers")
        print("3. Backtest single ticker")
        print("4. Exit")

        choice = input("\nSelect option (1-4): ").strip()

        if choice == '1':
            ticker = input("Enter ticker symbol (e.g., AAPL, BTC-USD): ").strip().upper()
            if ticker:
                analyzer.analyze(ticker)

        elif choice == '2':
            tickers_input = input("Enter ticker symbols (comma-separated): ").strip().upper()
            if tickers_input:
                tickers = [t.strip() for t in tickers_input.split(',')]
                analyzer.batch_analyze(tickers)

        elif choice == '3':
            ticker = input("Enter ticker symbol for backtest: ").strip().upper()
            if ticker:
                analyzer.analyze(ticker, backtest=True)

        elif choice == '4':
            print("\nThank you for using Professional Trading Analysis System!")
            break

        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
```
