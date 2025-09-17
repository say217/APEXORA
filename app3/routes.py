from flask import Blueprint, render_template, request, session, redirect, url_for
from functools import wraps
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

bp = Blueprint("app3", __name__, template_folder="templates")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("app2.login"))
        return f(*args, **kwargs)
    return decorated_function

def calculate_metrics(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None, None, None, None, None, None, "Invalid ticker or no data available."

        # Price Changes
        df["Daily_Change"] = df["Close"].diff()
        df["Pct_Change"] = df["Close"].pct_change() * 100
        df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))

        # Moving Averages
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["SMA_200"] = df["Close"].rolling(window=200).mean()
        df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

        # Volatility
        df["Std_Dev"] = df["Close"].rolling(window=20).std()
        annualized_vol = df["Std_Dev"].iloc[-1] * np.sqrt(252) if not df["Std_Dev"].empty else np.nan
        df["True_Range"] = df["High"] - df["Low"]
        df["ATR"] = df["True_Range"].rolling(window=14).mean()
        df["Daily_Range"] = df["High"] - df["Low"]
        avg_daily_range = df["Daily_Range"].mean()

        # Bollinger Bands
        df["BB_Mid"] = df["Close"].rolling(window=20).mean()
        df["BB_Std"] = df["Close"].rolling(window=20).std()
        df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
        df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]

        # RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # Stochastic Oscillator
        df["L14"] = df["Low"].rolling(window=14).min()
        df["H14"] = df["High"].rolling(window=14).max()
        df["%K"] = 100 * ((df["Close"] - df["L14"]) / (df["H14"] - df["L14"]))
        df["%D"] = df["%K"].rolling(window=3).mean()

        # VWAP
        df["Typical_Price"] = (df["High"] + df["Low"] + df["Close"]) / 3
        df["VWAP"] = (df["Typical_Price"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

        # Support/Resistance (Pivot Points)
        pivot = (df["High"].iloc[-1] + df["Low"].iloc[-1] + df["Close"].iloc[-1]) / 3
        support1 = (2 * pivot) - df["High"].iloc[-1]
        resistance1 = (2 * pivot) - df["Low"].iloc[-1]

        # Fibonacci Levels
        max_price = df["Close"].max()
        min_price = df["Close"].min()
        diff = max_price - min_price
        fib_levels = {
            "0%": max_price,
            "23.6%": max_price - 0.236 * diff,
            "38.2%": max_price - 0.382 * diff,
            "50%": max_price - 0.5 * diff,
            "61.8%": max_price - 0.618 * diff,
            "100%": min_price
        }

        # Returns
        initial_price = df["Close"].iloc[0]
        final_price = df["Close"].iloc[-1]
        cumulative_return = ((final_price / initial_price) - 1) * 100
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = ((final_price / initial_price) ** (1 / years) - 1) * 100 if years > 0 else np.nan

        # Risk Metrics
        market = yf.Ticker("^GSPC").history(period=period)
        df["Stock_Returns"] = df["Close"].pct_change()
        market["Market_Returns"] = market["Close"].pct_change()
        cov = df["Stock_Returns"].cov(market["Market_Returns"])
        var = market["Market_Returns"].var()
        beta = cov / var if var != 0 else np.nan
        sharpe_ratio = (df["Stock_Returns"].mean() * 252 - 0.02) / (df["Stock_Returns"].std() * np.sqrt(252)) if df["Stock_Returns"].std() != 0 else np.nan
        max_drawdown = ((df["Close"].cummax() - df["Close"]) / df["Close"].cummax()).max() * 100
        correlation = df["Stock_Returns"].corr(market["Market_Returns"])

        # ADX Calculation
        def calculate_adx(df, period=14):
            df = df.copy()
            df["H-L"] = df["High"] - df["Low"]
            df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
            df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
            df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
            df["DM+"] = np.where((df["High"] - df["High"].shift(1)) > (df["Low"].shift(1) - df["Low"]),
                                 df["High"] - df["High"].shift(1), 0)
            df["DM-"] = np.where((df["Low"].shift(1) - df["Low"]) > (df["High"] - df["High"].shift(1)),
                                 df["Low"].shift(1) - df["Low"], 0)
            df["TR14"] = df["TR"].rolling(window=period).sum()
            df["DM+14"] = df["DM+"].rolling(window=period).sum()
            df["DM-14"] = df["DM-"].rolling(window=period).sum()
            df["DI+"] = (df["DM+14"] / df["TR14"]) * 100
            df["DI-"] = (df["DM-14"] / df["TR14"]) * 100
            df["DX"] = (abs(df["DI+"] - df["DI-"]) / (df["DI+"] + df["DI-"])) * 100
            df["ADX"] = df["DX"].rolling(window=period).mean()
            return df["ADX"].iloc[-1] if not df["ADX"].empty else np.nan

        adx = calculate_adx(df)

        # Trend Analysis
        trend = "Sideways"
        if df["SMA_50"].iloc[-1] > df["SMA_200"].iloc[-1] and df["Close"].iloc[-1] > df["SMA_50"].iloc[-1]:
            trend = "Uptrend"
        elif df["SMA_50"].iloc[-1] < df["SMA_200"].iloc[-1] and df["Close"].iloc[-1] < df["SMA_50"].iloc[-1]:
            trend = "Downtrend"

        valid = df["SMA_50"].notnull() & df["SMA_200"].notnull()
        trend_duration_days = (df.loc[valid, "SMA_50"] > df.loc[valid, "SMA_200"]).astype(int).diff().ne(0).cumsum().value_counts().max()

        # Replace NaN with None for JSON serialization
        df = df.where(df.notna(), None)
        dates = df.index[valid].strftime("%Y-%m-%d").tolist()
        
        def filter_valid(series):
            return [x for x in series[valid].tolist() if x is not None]

        # Chart Data
        price_data = {
            "dates": dates,
            "close": filter_valid(df["Close"]),
            "sma_50": filter_valid(df["SMA_50"]),
            "sma_200": filter_valid(df["SMA_200"]),
            "bb_upper": filter_valid(df["BB_Upper"]),
            "bb_lower": filter_valid(df["BB_Lower"]),
            "vwap": filter_valid(df["VWAP"]),
            "fib_levels": {level: round(price, 2) if price is not None else None for level, price in fib_levels.items()},
            "support": round(support1, 2) if support1 is not None else None,
            "resistance": round(resistance1, 2) if resistance1 is not None else None
        }
        volume_data = {"dates": dates, "volume": filter_valid(df["Volume"])}
        rsi_data = {"dates": dates, "rsi": filter_valid(df["RSI"])}
        macd_data = {"dates": dates, "macd": filter_valid(df["MACD"]), "signal": filter_valid(df["Signal_Line"])}

        # Pie Chart Data (using RSI, Stochastic %K, %D, and ADX)
        pie_data = {
            "labels": ["RSI", "Stochastic %K", "Stochastic %D", "ADX"],
            "values": [
                round(df["RSI"].iloc[-1], 2) if df["RSI"].iloc[-1] is not None else 0,
                round(df["%K"].iloc[-1], 2) if df["%K"].iloc[-1] is not None else 0,
                round(df["%D"].iloc[-1], 2) if df["%D"].iloc[-1] is not None else 0,
                round(adx, 2) if not np.isnan(adx) else 0
            ],
            "colors": ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0"]
        }

        # Metrics
        metrics = {
            "Latest Close": {"value": round(df["Close"].iloc[-1], 2), "desc": "Most recent closing price."},
            "Daily Change": {"value": round(df["Daily_Change"].iloc[-1], 2) if df["Daily_Change"].iloc[-1] is not None else None, "desc": "Change in price from previous day."},
            "Pct Change": {"value": round(df["Pct_Change"].iloc[-1], 2) if df["Pct_Change"].iloc[-1] is not None else None, "desc": "Percentage change from previous day."},
            "Cumulative Return": {"value": round(cumulative_return, 2), "desc": "Total return over the period (%)."},
            "CAGR": {"value": round(cagr, 2) if not np.isnan(cagr) else None, "desc": "Compound annual growth rate (%)."},
            "Annualized Volatility": {"value": round(annualized_vol, 2) if not np.isnan(annualized_vol) else None, "desc": "Volatility of returns annualized (%)."},
            "Average Daily Range": {"value": round(avg_daily_range, 2), "desc": "Average daily high-low range."},
            "ATR": {"value": round(df["ATR"].iloc[-1], 2) if df["ATR"].iloc[-1] is not None else None, "desc": "Average True Range (14-day)."},
            "RSI": {"value": round(df["RSI"].iloc[-1], 2) if df["RSI"].iloc[-1] is not None else None, "desc": "Relative Strength Index (14-day)."},
            "Stochastic %K": {"value": round(df["%K"].iloc[-1], 2) if df["%K"].iloc[-1] is not None else None, "desc": "Stochastic Oscillator %K (14-day)."},
            "Stochastic %D": {"value": round(df["%D"].iloc[-1], 2) if df["%D"].iloc[-1] is not None else None, "desc": "Stochastic Oscillator %D (3-day SMA of %K)."},
            "ADX": {"value": round(adx, 2) if not np.isnan(adx) else None, "desc": "Average Directional Index (>25 indicates strong trend)."},
            "Beta": {"value": round(beta, 2) if not np.isnan(beta) else None, "desc": "Stock volatility relative to S&P 500."},
            "Sharpe Ratio": {"value": round(sharpe_ratio, 2) if not np.isnan(sharpe_ratio) else None, "desc": "Risk-adjusted return (higher is better)."},
            "Max Drawdown": {"value": round(max_drawdown, 2), "desc": "Largest peak-to-trough decline (%)."},
            "Correlation with S&P 500": {"value": round(correlation, 2) if not np.isnan(correlation) else None, "desc": "Correlation of returns with S&P 500."},
            "Support Level": {"value": round(support1, 2), "desc": "Pivot-based support level."},
            "Resistance Level": {"value": round(resistance1, 2), "desc": "Pivot-based resistance level."},
            "VWAP": {"value": round(df["VWAP"].iloc[-1], 2) if df["VWAP"].iloc[-1] is not None else None, "desc": "Volume-Weighted Average Price."},
            "Trend": {"value": trend, "desc": "Current trend direction."},
            "Trend Duration (Days)": {"value": trend_duration_days, "desc": "Estimated days in current trend."}
        }
        metrics.update({f"Fibonacci {level}": {"value": round(price, 2), "desc": f"Fibonacci retracement level at {level}."} for level, price in fib_levels.items()})

        return (json.dumps(price_data, ensure_ascii=False), json.dumps(volume_data, ensure_ascii=False),
                json.dumps(rsi_data, ensure_ascii=False), json.dumps(macd_data, ensure_ascii=False),
                metrics, json.dumps(pie_data, ensure_ascii=False), None)

    except Exception as e:
        return None, None, None, None, None, None, f"Error calculating metrics: {str(e)}"

@bp.route("/", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":
        ticker = request.form.get("ticker", "").upper()
        if not ticker:
            return render_template("index.html", error="Please enter a ticker symbol.")
        
        price_data, volume_data, rsi_data, macd_data, metrics, pie_data, error = calculate_metrics(ticker)
        if error:
            return render_template("index.html", error=error)

        return render_template(
            "results.html",
            ticker=ticker,
            price_plot=price_data,
            volume_plot=volume_data,
            rsi_plot=rsi_data,
            macd_plot=macd_data,
            metrics=metrics,
            pie_plot=pie_data
        )
    return render_template("index.html")