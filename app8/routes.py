import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
from flask import Blueprint, render_template, session, redirect, url_for, request
from functools import wraps
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
import pickle
from torch.cuda.amp import autocast, GradScaler
import time
import random
import yfinance as yf
import seaborn as sns
import uuid
from io import StringIO
from contextlib import redirect_stdout

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

bp = Blueprint('app8', __name__, template_folder='templates')

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('app2.login'))
        return f(*args, **kwargs)
    return decorated_function

@bp.route('/', methods=['GET', 'POST'])
@login_required
def page():
    if request.method == 'POST':
        ticker = request.form.get('ticker', 'AAPL').upper()
        buf = StringIO()
        with redirect_stdout(buf):
            results = run_prediction(ticker)
        output = buf.getvalue().replace('\n', '<br>')
        return render_template(
            'home8.html',
            output=output,
            images=results['image_paths'],
            metrics=results['metrics'],
            ticker=ticker
        )
    return render_template('home8.html', output='', images=[], metrics=None, ticker='')

# --------------------------
# Reproducibility helpers
# --------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# --------------------------
# Configuration
# --------------------------
class Config:
    def __init__(self, ticker_symbol: str):
        self.ticker = ticker_symbol.upper()
        self.start_date = "2022-01-01"
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.features = ['Close', 'High', 'Low']
        self.technical_indicators = [
            'SMA_10', 'RSI', 'MACD',
            'BB_Upper', 'BB_Lower', 'ATR',
            'Price_Change', 'Log_Close', 'Volatility_10',
            'Momentum_5', 'Momentum_10', 'EMA_10', 'EMA_20', 'Volatility_20', 'Volatility_50'
        ]
        self.window_size = 40
        self.train_split = 0.8
        self.forecast_days = 4
        self.past_days_plot = 14
        self.batch_size = 64
        self.epochs = 50
        self.learning_rate = 5e-4
        self.lstm_units = 192
        self.num_layers = 3
        self.dropout = 0.25
        self.loss = 'huber'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_features = None

# --------------------------
# Data handling
# --------------------------
class DataHandler:
    def __init__(self, config: Config):
        self.config = config
        self.scaler = MinMaxScaler()
        self.dates = None
        self.n_features = None

    def download_data(self):
        cache_file = f'data_{self.config.ticker}.pkl'
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                self.dates = data.index
                print(f"Loaded cached data for {self.config.ticker}")
                return data
            except Exception:
                print(f"Cache file corrupted, re-downloading data for {self.config.ticker}")

        try:
            print(f"Downloading data for {self.config.ticker}...")
            data = yf.download(
                self.config.ticker,
                start=self.config.start_date,
                end=self.config.end_date,
                interval="1d",
                progress=False
            )
            if data.empty:
                raise ValueError(f"No data retrieved for ticker {self.config.ticker}")
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            required_cols = ['Close', 'Volume', 'High', 'Low', 'Open']
            available_cols = [col for col in required_cols if col in data.columns]
            if 'Close' not in available_cols:
                raise ValueError(f"Critical error: 'Close' price not available for {self.config.ticker}")
            self.config.features = [col for col in self.config.features if col in available_cols]
            if len(data) < self.config.window_size:
                raise ValueError(f"Insufficient data: {len(data)} samples, need at least {self.config.window_size}")
            data = data[self.config.features].copy()
            self.dates = data.index
            data = self._add_technical_indicators(data)
            data = self._handle_missing_values(data)
            print(f"Data shape after preprocessing: {data.shape}")
            print(f"Date range: {data.index[0]} to {data.index[-1]}")
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            return data
        except Exception as e:
            print(f"Error downloading data for {self.config.ticker}: {e}")
            return None

    def _add_technical_indicators(self, data: pd.DataFrame):
        try:
            data['SMA_10'] = data['Close'].rolling(window=10, min_periods=1).mean()
            data['RSI'] = self._compute_rsi(data['Close'])
            data['MACD'] = self._compute_macd(data['Close'])
            data['BB_Upper'], data['BB_Lower'] = self._compute_bollinger_bands(data['Close'])
            data['ATR'] = self._compute_atr(data)
            data['Price_Change'] = data['Close'].pct_change()
            data['Log_Close'] = np.log(data['Close'] + 1e-8)
            data['Volatility_10'] = data['Close'].pct_change().rolling(10, min_periods=1).std()
            data['Momentum_5'] = data['Close'] - data['Close'].shift(5)
            data['Momentum_10'] = data['Close'] - data['Close'].shift(10)
            data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
            data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
            data['Volatility_20'] = data['Close'].pct_change().rolling(20, min_periods=1).std()
            data['Volatility_50'] = data['Close'].pct_change().rolling(50, min_periods=1).std()
        except Exception as e:
            print(f"Error computing technical indicators: {e}")
            for col in self.config.technical_indicators:
                if col not in data.columns:
                    data[col] = data['Close']
        return data

    def _handle_missing_values(self, data: pd.DataFrame):
        data = data.fillna(method='ffill')
        for col in data.columns:
            if data[col].isna().any():
                data[col] = data[col].fillna(data[col].mean())
        data = data.replace([np.inf, -np.inf], np.nan).fillna(data.mean())
        data = data.fillna(0)
        return data

    def _compute_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _compute_macd(self, prices, slow=26, fast=12, signal=9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        return macd.fillna(0)

    def _compute_bollinger_bands(self, prices, window=20, num_std=2):
        sma = prices.rolling(window=window, min_periods=1).mean()
        std = prices.rolling(window=window, min_periods=1).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper.fillna(sma), lower.fillna(sma)

    def _compute_atr(self, data, period=14):
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        return atr.fillna(atr.mean())

    def prepare_data(self, data: pd.DataFrame):
        feature_list = list(dict.fromkeys(self.config.features + self.config.technical_indicators))
        available_features = [f for f in feature_list if f in data.columns]
        print(f"Using features: {available_features}")
        self.n_features = len(available_features)
        features_vals = data[available_features].values
        scaled_features = self.scaler.fit_transform(features_vals)
        X, y = [], []
        W = self.config.window_size
        for i in range(len(scaled_features) - W):
            X.append(scaled_features[i:i + W])
            y.append(scaled_features[i + W, available_features.index('Close')])
        X, y = np.array(X), np.array(y)
        train_size = int(len(X) * self.config.train_split)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        return (X_train, y_train), (X_test, y_test), scaled_features

    def inverse_target_transform(self, scaled_data):
        scaled = np.asarray(scaled_data).reshape(-1, 1)
        dummy = np.zeros((len(scaled), self.n_features))
        dummy[:, 0] = scaled.flatten()
        return self.scaler.inverse_transform(dummy)[:, 0]

# --------------------------
# Dataset
# --------------------------
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --------------------------
# Attention
# --------------------------
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_out):
        score = self.tanh(self.attn(lstm_out))
        attn_weights = self.softmax(torch.matmul(score, self.v))
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out)
        return context.squeeze(1), attn_weights

# --------------------------
# Enhanced Model (BiGRU + Attention)
# --------------------------
class EnhancedGRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_size = config.n_features
        self.conv1 = nn.Conv1d(input_size, input_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_size, input_size, kernel_size=5, padding=2)
        self.act = nn.ReLU()
        self.gru = nn.GRU(
            input_size=input_size*2,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attn = nn.Linear(256, 256)
        self.fc = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.15)
        self.ln = nn.LayerNorm(256)

    def forward(self, x):
        conv_out1 = self.act(self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1))
        conv_out2 = self.act(self.conv2(x.permute(0, 2, 1)).permute(0, 2, 1))
        conv_out = torch.cat([conv_out1, conv_out2], dim=2)
        gru_out, _ = self.gru(conv_out)
        scores = torch.matmul(self.attn(gru_out), gru_out.transpose(1,2)) / (gru_out.size(-1)**0.5)
        weights = torch.softmax(scores.mean(dim=1), dim=1).unsqueeze(-1)
        context = (weights * gru_out).sum(dim=1)
        context = self.ln(context)
        context = self.dropout(context)
        out = self.fc(context)
        return out.squeeze(-1)

# --------------------------
# Trainer
# --------------------------
class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)

    def train(self, model, train_loader, test_loader):
        if self.config.loss == 'huber':
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        scaler = GradScaler() if torch.cuda.is_available() else None
        best_loss = float('inf')
        train_losses, test_losses = [], []
        patience, early_stopping = 15, False
        patience_counter = 0
        start_time = time.time()
        os.makedirs("models", exist_ok=True)
        for epoch in range(self.config.epochs):
            if early_stopping:
                break
            epoch_start = time.time()
            model.train()
            train_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                if scaler:
                    with autocast():
                        output = model(X)
                        loss = criterion(output, y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(X)
                    loss = criterion(output, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                train_loss += loss.item() * X.size(0)
            train_loss /= len(train_loader.dataset)
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    if scaler:
                        with autocast():
                            output = model(X)
                            batch_loss = criterion(output, y).item()
                    else:
                        output = model(X)
                        batch_loss = criterion(output, y).item()
                    test_loss += batch_loss * X.size(0)
            test_loss /= len(test_loader.dataset)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            scheduler.step(test_loss)
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(model.state_dict(), f'models/{self.config.ticker}_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    early_stopping = True
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | "
                    f"Time: {time.time() - epoch_start:.2f}s"
                )
        print(f"Total training time: {(time.time() - start_time)/60:.2f} minutes")
        return train_losses, test_losses

    def evaluate(self, model, loader, data_handler: DataHandler):
        model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                output = model(X).cpu().numpy()
                preds.extend(data_handler.inverse_target_transform(output))
                actuals.extend(data_handler.inverse_target_transform(y.cpu().numpy()))
        return np.array(preds), np.array(actuals)

    def directional_accuracy(self, actuals, preds):
        if len(actuals) <= 1 or len(preds) <= 1:
            return 0.0
        actual_diff = np.diff(actuals)
        pred_diff = np.diff(preds)
        correct = np.sum((actual_diff > 0) == (pred_diff > 0))
        return correct / len(actual_diff) * 100.0

# --------------------------
# Visualization
# --------------------------
def plot_results(dates, actuals, preds, title, config, save=False, static_dir='static'):
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 6), facecolor='#1c2526')
    plt.plot(dates, actuals, label='Actual', color='#00ff00', linewidth=2)
    plt.plot(dates, preds, '--', label='Predicted', color='#00f7ff', linewidth=2)
    plt.title(f"{title} - {config.ticker}", fontsize=14, color='white')
    plt.xlabel('Date', fontsize=12, color='white')
    plt.ylabel('Price (USD)', fontsize=12, color='white')
    plt.legend(fontsize=10, facecolor='#1c2526', edgecolor='white', labelcolor='white')
    plt.grid(True, alpha=0.3, color='#cccccc')
    plt.gcf().autofmt_xdate()
    plt.tick_params(colors='white')
    plt.tight_layout()
    if save:
        filename = str(uuid.uuid4()) + '_results.png'
        filepath = os.path.join(static_dir, filename)
        plt.savefig(filepath)
        plt.close()
        return filename
    else:
        plt.show()
        return None

def plot_forecast(dates, prices, forecast_dates, forecast_prices, std, config, save=False, static_dir='static'):
    plt.style.use('dark_background')
    plt.figure(figsize=(14, 8), facecolor='#1c2526')
    past_days = config.past_days_plot
    historical_dates = dates[-past_days:]
    historical_prices = prices[-past_days:]
    plt.plot(historical_dates, historical_prices, 'o-', label='Historical (Past 60 Days)', color='#ff00ff', linewidth=2)
    plt.plot(forecast_dates, forecast_prices, 'o-', label=f'Forecast (Next {config.forecast_days} Days)', color='#ffff00', linewidth=2, markersize=6)
    plt.fill_between(forecast_dates, forecast_prices - std, forecast_prices + std, alpha=0.2, color='#ffff00', label='Confidence Interval')
    plt.axvline(x=dates[-1], color='#cccccc', linestyle='--', alpha=0.7, label='Today')
    plt.title(f"{config.ticker} - Past {past_days} Days & {config.forecast_days}-Day Forecast", fontsize=14, color='white')
    plt.xlabel('Date', fontsize=12, color='white')
    plt.ylabel('Price (USD)', fontsize=12, color='white')
    plt.legend(fontsize=10, facecolor='#1c2526', edgecolor='white', labelcolor='white')
    plt.grid(True, alpha=0.3, color='#cccccc')
    plt.gcf().autofmt_xdate()
    plt.tick_params(colors='white')
    plt.tight_layout()
    forecast_data = []
    for date, price in zip(forecast_dates, forecast_prices):
        forecast_data.append(f"{date.strftime('%Y-%m-%d (%A)')}: ${price:.2f}")
    if save:
        filename = str(uuid.uuid4()) + '_forecast.png'
        filepath = os.path.join(static_dir, filename)
        plt.savefig(filepath)
        plt.close()
        return filename, forecast_data
    else:
        print(f"\n{config.ticker} - {config.forecast_days} Day Forecast:")
        print("-" * 50)
        for item in forecast_data:
            print(item)
        plt.show()
        return None, forecast_data

def plot_frequency_and_heatmap(data: pd.DataFrame, ticker: str, features: list, technical_indicators: list, save=False, static_dir='static'):
    plt.style.use('dark_background')
    all_features = list(dict.fromkeys(features + technical_indicators))
    available_features = [f for f in all_features if f in data.columns]
    hist_features = ['Close', 'RSI', 'SMA_10', 'BB_Upper', 'BB_Lower', 'Volatility_10', 'Volatility_20', 'Price_Change', 'MACD']
    hist_features = [f for f in hist_features if f in available_features]
    paths = []
    if hist_features:
        plt.figure(figsize=(15, 10), facecolor='#1c2526')
        colors = ['#00ffab', '#ff6f61', '#ffd700', '#6ab04c', '#ff85ff', '#00b7eb', '#ff9f43', '#5c5c8a', '#ff4f81']
        for i, feature in enumerate(hist_features, 1):
            plt.subplot(3, 3, i)
            data_clean = data[feature].replace([np.inf, -np.inf], np.nan).dropna()
            if data_clean.empty:
                print(f"Warning: No valid data for {feature}. Skipping histogram.")
                continue
            sns.histplot(data_clean, bins=20, kde=True, color=colors[i-1], edgecolor='white', alpha=0.7)
            plt.title(f'{feature} Distribution', fontsize=10, color='white')
            plt.xlabel(feature, fontsize=8, color='white')
            plt.ylabel('Frequency', fontsize=8, color='white')
            plt.grid(True, alpha=0.3, color='gray')
            plt.tick_params(colors='white')
        plt.suptitle(f'{ticker} Feature Distributions', fontsize=14, color='white')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save:
            filename1 = str(uuid.uuid4()) + '_frequency.png'
            filepath1 = os.path.join(static_dir, filename1)
            plt.savefig(filepath1)
            plt.close()
            paths.append(filename1)
        else:
            plt.show()
    plt.figure(figsize=(12, 10), facecolor='#1c2526')
    correlation_matrix = data[available_features].corr()
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='Spectral',
        center=0,
        vmin=-1,
        vmax=1,
        fmt='.2f',
        square=True,
        cbar_kws={'label': 'Correlation', 'shrink': 0.8},
        annot_kws={'size': 8, 'color': 'white'},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title(f'{ticker} Feature Correlation Heatmap', fontsize=14, color='white')
    plt.xticks(rotation=45, ha='right', color='white')
    plt.yticks(rotation=0, color='white')
    plt.tight_layout()
    if save:
        filename2 = str(uuid.uuid4()) + '_heatmap.png'
        filepath2 = os.path.join(static_dir, filename2)
        plt.savefig(filepath2)
        plt.close()
        paths.append(filename2)
    else:
        plt.show()
    return paths

# --------------------------
# Future prediction (recursive)
# --------------------------
def predict_future(model, last_window, num_days, data_handler: DataHandler, config: Config):
    model.eval()
    predictions_scaled = []
    current_window = last_window.copy()
    for _ in range(num_days):
        input_tensor = torch.FloatTensor(current_window).unsqueeze(0).to(config.device)
        with torch.no_grad():
            pred_scaled = model(input_tensor).cpu().item()
        predictions_scaled.append(pred_scaled)
        current_window = np.roll(current_window, -1, axis=0)
        current_window[-1, 0] = pred_scaled
    return data_handler.inverse_target_transform(np.array(predictions_scaled))

# --------------------------
# Main logic
# --------------------------
def run_prediction(ticker):
    results = {'image_paths': [], 'metrics': None}
    print("=" * 60)
    print("Enhanced Stock Price Prediction System (Conv1D + BiLSTM + Attention)")
    print("=" * 60)
    if not ticker:
        ticker = "AAPL"
    config = Config(ticker)
    os.makedirs('models', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    data_handler = DataHandler(config)
    data = data_handler.download_data()
    if data is None or data.empty:
        print(f"Failed to download data for {ticker}. Please check the ticker symbol.")
        return results
    if data_handler.dates is None:
        print(f"Date index not set for {ticker}. Data download failed.")
        return results
    freq_paths = plot_frequency_and_heatmap(data, ticker, config.features, config.technical_indicators, save=True)
    results['image_paths'].extend(freq_paths)
    (X_train, y_train), (X_test, y_test), scaled_data = data_handler.prepare_data(data)
    config.n_features = data_handler.n_features
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)
    print(f"\nInitializing model on device: {config.device}")
    model = EnhancedGRU(config).to(config.device)
    trainer = Trainer(config)
    print("Starting model training...")
    train_losses, test_losses = trainer.train(model, train_loader, test_loader)
    try:
        model.load_state_dict(torch.load(f'models/{ticker}_model.pth', map_location=config.device))
        print(f"Loaded best model for {ticker}")
    except FileNotFoundError:
        print("Warning: Model file not found. Using current model state.")
    test_preds, test_actuals = trainer.evaluate(model, test_loader, data_handler)
    metrics = {}
    if len(test_actuals) > 0 and len(test_preds) > 0:
        metrics['rmse'] = np.sqrt(mean_squared_error(test_actuals, test_preds))
        metrics['mae'] = mean_absolute_error(test_actuals, test_preds)
        metrics['r2'] = r2_score(test_actuals, test_preds)
        metrics['directional_acc'] = trainer.directional_accuracy(test_actuals, test_preds)
        print(f"\n{ticker} Model Performance:")
        print("-" * 40)
        print(f"RMSE: ${metrics['rmse']:.2f}")
        print(f"MAE: ${metrics['mae']:.2f}")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"Directional Accuracy: {metrics['directional_acc']:.2f}%")
        test_dates = data_handler.dates[len(X_train) + config.window_size:]
        if len(test_dates) >= len(test_actuals):
            results_path = plot_results(test_dates[:len(test_actuals)], test_actuals, test_preds, f"Test Predictions", config, save=True)
            if results_path:
                results['image_paths'].append(results_path)
    results['metrics'] = metrics
    print(f"\nGenerating {config.forecast_days}-day forecast...")
    last_window = scaled_data[-config.window_size:]
    future_prices = predict_future(model, last_window, config.forecast_days, data_handler, config)
    start_date = datetime.now() + timedelta(days=1)
    future_dates = [start_date + timedelta(days=i) for i in range(config.forecast_days)]
    if len(test_preds) > 0 and len(test_actuals) > 0:
        std = np.std(test_preds - test_actuals)
    else:
        std = np.std(future_prices) * 0.1
    historical_prices = data_handler.inverse_target_transform(scaled_data[:, 0])
    forecast_path, forecast_data = plot_forecast(data_handler.dates, historical_prices, future_dates, future_prices, std, config, save=True)
    if forecast_path:
        results['image_paths'].append(forecast_path)
    metrics['forecast_data'] = forecast_data
    print(f"\nAnalysis completed for {ticker}")
    print("=" * 60)
    return results