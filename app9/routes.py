from flask import Blueprint, request, jsonify, session, redirect, url_for, render_template
from functools import wraps
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import google.generativeai as genai
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from cachetools import TTLCache
import yfinance.exceptions as yf_exceptions
import re
from ratelimit import limits, sleep_and_retry

# Define Blueprint for app1
bp = Blueprint('app9', __name__, template_folder='templates')

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Gemini API key
genai.configure(api_key="AIzaSyAVI4DwPzQ7vl5J19vQYF-Ywg9BtChj1fk")

# Initialize caches (TTL in seconds)
price_cache = TTLCache(maxsize=100, ttl=300)  # Cache prices for 5 minutes
history_cache = TTLCache(maxsize=50, ttl=3600)  # Cache historical data for 1 hour
stats_cache = TTLCache(maxsize=100, ttl=300)  # Cache stats for 5 minutes

# Sample stock data for sidebar
stock_data = [
    {
        "name": "3M India Ltd",
        "code": "3MINDIA.NS",
        "image": "https://logosmarcas.net/wp-content/uploads/2022/02/3M-Simbolo.png"
    },
    {
        "name": "Aarti Industries Ltd",
        "code": "AARTIIND.NS",
        "image": "https://companieslogo.com/img/orig/AARTIIND.NS-9c2cfd45.png?t=1613409100"
    },
    {
        "name": "ABB India Ltd",
        "code": "ABB.NS",
        "image": "https://seekvectors.com/storage/images/6b84dc4fb705ba7df8405f6180c6d5d5.jpg"
    }
]

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('app2.login'))
        return f(*args, **kwargs)
    return decorated_function

# Check if market is open based on ticker
def is_market_open(ticker):
    now = datetime.now()
    if ticker.endswith('.NS'):  # Indian market (NSE)
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = now.astimezone(ist)
        hour = now_ist.hour
        minute = now_ist.minute
        weekday = now_ist.weekday()
        if weekday < 5:  # Monday-Friday
            if (hour == 9 and minute >= 15) or (9 < hour < 15) or (hour == 15 and minute <= 30):
                return True
    else:  # US market (default)
        est = pytz.timezone('America/New_York')
        now_est = now.astimezone(est)
        hour = now_est.hour
        minute = now_est.minute
        weekday = now_est.weekday()
        if weekday < 5:  # Monday-Friday
            if (hour == 9 and minute >= 30) or (9 < hour < 16) or (hour == 16 and minute == 0):
                return True
    return False

# Rate limit for yfinance API calls (e.g., 60 calls per minute)
@sleep_and_retry
@limits(calls=60, period=60)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=5, max=30),
    retry=retry_if_exception_type(yf_exceptions.YFRateLimitError)
)
def get_stock_price(ticker):
    try:
        if ticker in price_cache:
            logger.info(f"Returning cached price for {ticker}")
            return price_cache[ticker]
        
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get('regularMarketPrice')
        if price is None:
            logger.error(f"No regularMarketPrice found for ticker {ticker}")
            return None
        
        price_cache[ticker] = price
        return price
    except yf_exceptions.YFInvalidTickerError:
        logger.error(f"Invalid ticker: {ticker}")
        return None
    except yf_exceptions.YFRateLimitError as e:
        logger.error(f"Rate limit error fetching price for {ticker}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error fetching price for {ticker}: {str(e)}")
        return None

@sleep_and_retry
@limits(calls=60, period=60)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=5, max=30),
    retry=retry_if_exception_type(yf_exceptions.YFRateLimitError)
)
def get_key_stats(ticker):
    try:
        if ticker in stats_cache:
            logger.info(f"Returning cached stats for {ticker}")
            return stats_cache[ticker]
        
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period='1d')
        
        stats = {
            'open': info.get('regularMarketOpen'),
            'high': info.get('regularMarketDayHigh'),
            'low': info.get('regularMarketDayLow'),
            'close': info.get('regularMarketPreviousClose'),
            'volume': info.get('regularMarketVolume')
        }
        
        if not history.empty:
            latest = history.iloc[-1]
            stats.update({
                'open': latest.get('Open', stats['open']),
                'high': latest.get('High', stats['high']),
                'low': latest.get('Low', stats['low']),
                'close': latest.get('Close', stats['close']),
                'volume': latest.get('Volume', stats['volume'])
            })
        
        stats_cache[ticker] = stats
        return stats
    except yf_exceptions.YFInvalidTickerError:
        logger.error(f"Invalid ticker: {ticker}")
        return None
    except yf_exceptions.YFRateLimitError as e:
        logger.error(f"Rate limit error fetching stats for {ticker}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error fetching stats for {ticker}: {str(e)}")
        return None

@sleep_and_retry
@limits(calls=60, period=60)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=5, max=30),
    retry=retry_if_exception_type(yf_exceptions.YFRateLimitError)
)
def get_historical_data(ticker):
    try:
        if ticker in history_cache:
            logger.info(f"Returning cached historical data for {ticker}")
            return history_cache[ticker]
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10*365)  # Last 10 years
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, interval='1wk')  # Weekly data
        
        if data.empty:
            logger.error(f"No historical data found for ticker {ticker}")
            return None
        
        historical_data = {
            'labels': [index.strftime('%Y-%m-%d') for index in data.index],
            'prices': data['Close'].tolist(),
            'volumes': data['Volume'].tolist()
        }
        
        history_cache[ticker] = historical_data
        return historical_data
    except yf_exceptions.YFInvalidTickerError:
        logger.error(f"Invalid ticker: {ticker}")
        return None
    except yf_exceptions.YFRateLimitError as e:
        logger.error(f"Rate limit error fetching historical data for {ticker}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error fetching historical data for {ticker}: {str(e)}")
        return None

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
def get_stock_details(ticker):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Provide details about the stock {ticker}. Details about this company in 5 lines."
        response = model.generate_content(prompt)
        formatted_response = response.text.replace('\n', '\n\n')
        return formatted_response
    except Exception as e:
        logger.error(f"Error fetching Gemini details for {ticker}: {str(e)}")
        return f"Error fetching details: {str(e)}"

# Validate ticker format
def is_valid_ticker(ticker):
    return bool(re.match(r'^[A-Z0-9.-]+$', ticker))

# Home route
@bp.route('/')
@login_required
def home():
    return render_template('home9.html', stocks=stock_data)

# API route to fetch stock price, historical data, stats, or Gemini details
@bp.route('/get_price', methods=['POST'])
@login_required
def get_price():
    ticker = request.form.get('ticker', '').upper()
    if not ticker or not is_valid_ticker(ticker):
        logger.warning(f"Invalid or missing ticker: {ticker}")
        return jsonify({'error': 'Invalid or missing ticker'}), 400
    
    gemini_details = get_stock_details(ticker)
    response_data = {
        'ticker': ticker,
        'details': gemini_details,
        'price': None,
        'historical_data': None,
        'stats': None,
        'market_open': False,
        'time': datetime.now(pytz.timezone('America/New_York')).strftime("%H:%M:%S")
    }
    
    try:
        if is_market_open(ticker):
            response_data['market_open'] = True
            price = get_stock_price(ticker)
            historical_data = get_historical_data(ticker)
            stats = get_key_stats(ticker)
            
            response_data['price'] = price
            response_data['historical_data'] = historical_data
            response_data['stats'] = stats
            
            if price is None or historical_data is None or stats is None:
                logger.error(f"Partial data fetch failure for {ticker}")
                response_data['error'] = 'Unable to fetch some data'
                return jsonify(response_data), 500
            
            logger.info(f"Successfully fetched data for {ticker}: price={price}")
            return jsonify(response_data)
        else:
            response_data['error'] = 'Market is closed'
            # Try to return cached data if available
            response_data['price'] = price_cache.get(ticker)
            response_data['historical_data'] = history_cache.get(ticker)
            response_data['stats'] = stats_cache.get(ticker)
            
            if response_data['historical_data'] is None:
                logger.error(f"No historical data available for {ticker}")
                response_data['error'] = f'No historical data available for {ticker}'
                return jsonify(response_data), 500
            
            logger.info(f"Market closed for {ticker}, returned cached data")
            return jsonify(response_data)
    
    except yf_exceptions.YFRateLimitError:
        logger.error(f"Rate limit exceeded for {ticker} request")
        # Return cached data if available
        response_data['price'] = price_cache.get(ticker)
        response_data['historical_data'] = history_cache.get(ticker)
        response_data['stats'] = stats_cache.get(ticker)
        response_data['error'] = 'Rate limit exceeded. Returning cached data if available.'
        return jsonify(response_data), 429
    except Exception as e:
       logger.exception(f"Unexpected error for {ticker}: {str(e)}")  # Use logger.exception for stack trace
       response_data['error'] = f'Unexpected error: {str(e)}'
       return jsonify(response_data), 500