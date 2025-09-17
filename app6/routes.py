from flask import Blueprint, render_template, request, session, redirect, url_for, flash, send_file
from functools import wraps
import yfinance as yf
import json
from datetime import datetime
import pytz
import os
import logging

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),  # Log to a file for persistence
        logging.StreamHandler()  # Also log to console
    ]
)

bp = Blueprint('app6', __name__, template_folder='templates')

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            logging.info("User not logged in, redirecting to login")
            return redirect(url_for('app2.login'))
        return f(*args, **kwargs)
    return decorated_function

# Check if Indian market is open (9:15 AM to 3:30 PM IST, Mon-Fri)
def is_market_open():
    now = datetime.now(pytz.timezone('Asia/Kolkata'))
    hour = now.hour
    minute = now.minute
    weekday = now.weekday()
    
    if weekday < 5:  # 0-4 is Monday-Friday
        if (hour == 9 and minute >= 15) or (10 <= hour < 15) or (hour == 15 and minute <= 30):
            return True
    return False

# Get company data
def get_company_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        logging.debug(f"Stock info for {ticker}: {info}")
        
        data = {
            'Company Name': info.get('longName', 'N/A'),
            'Ticker': ticker,
            'Current Price': info.get('regularMarketPrice', info.get('previousClose', 'N/A')),
            'Currency': info.get('currency', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A'),
            'P/E Ratio': info.get('trailingPE', 'N/A'),
            '52 Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52 Week Low': info.get('fiftyTwoWeekLow', 'N/A'),
            'Average Volume': info.get('averageVolume', 'N/A'),
            'Dividend Yield': info.get('dividendYield', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Market Status': 'Open (Real-time)' if is_market_open() else 'Closed (Last Close)',
            'Date': datetime.now().strftime("%Y-%m-%d"),
            'Time': datetime.now().strftime("%H:%M:%S"),
            'Free Cash Flow': info.get('freeCashflow', 'N/A'),
            'Total Debt': info.get('totalDebt', 'N/A'),
            'Revenue': info.get('totalRevenue', 'N/A'),
            'Net Income': info.get('netIncomeToCommon', 'N/A'),
            'EPS': info.get('trailingEps', 'N/A'),
            'Beta': info.get('beta', 'N/A'),
            'Price to Book Ratio': info.get('priceToBook', 'N/A'),
            'Return on Equity': info.get('returnOnEquity', 'N/A'),
            'Operating Margin': info.get('operatingMargins', 'N/A'),
            'Shares Outstanding': info.get('sharesOutstanding', 'N/A')
        }
        return data
    except Exception as e:
        logging.error(f"Error fetching data for ticker {ticker}: {str(e)}")
        return None

# Save data to JSON file
def save_to_json_file(data, ticker):
    if not data:
        logging.error(f"No data provided to save for ticker {ticker}")
        return False
    filename = os.path.join('static', 'json', f"{ticker}_info_{data['Date']}.json")
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)
        logging.info(f"Successfully saved data to {filename}")
        return filename
    except Exception as e:
        logging.error(f"Error saving JSON file {filename}: {str(e)}")
        return False

# Format data for display
def format_data(data):
    formatted_data = {}
    for key, value in data.items():
        if isinstance(value, (int, float)) and key not in ['Dividend Yield', 'P/E Ratio', 'Beta', 'Price to Book Ratio', 'Return on Equity', 'Operating Margin', 'EPS']:
            formatted_data[key] = f"{value:,.2f}" if value != 'N/A' else 'N/A'
        elif key == 'Dividend Yield' and value != 'N/A':
            formatted_data[key] = f"{value:.2%}"
        elif key in ['Return on Equity', 'Operating Margin'] and value != 'N/A':
            formatted_data[key] = f"{value:.2%}"
        elif isinstance(value, float) and value != 'N/A':
            formatted_data[key] = f"{value:.2f}"
        else:
            formatted_data[key] = value
    return formatted_data

@bp.route('/', methods=['GET', 'POST'])
@login_required
def page():
    data = None
    filename = None
    error = None
    
    if request.method == 'POST':
        ticker = request.form.get('ticker', '').strip()
        logging.info(f"Received ticker: {ticker}")
        if ticker:
            company_data = get_company_data(ticker)
            if company_data:
                filename = save_to_json_file(company_data, ticker)
                if filename:
                    data = format_data(company_data)
                else:
                    error = "Error: Could not save data to JSON file."
            else:
                error = f"Error: Could not fetch data for {ticker}. Check ticker or internet."
        else:
            error = "Please enter a valid ticker symbol."
    
    return render_template('home6.html', data=data, filename=filename, error=error)

@bp.route('/download/<path:filename>')
@login_required
def download_file(filename):
    try:
        file_path = os.path.join('static', 'json', filename)
        logging.debug(f"Attempting to download file: {file_path}")
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            flash(f"Error: File {filename} not found.", "error")
            return redirect(url_for('app3.page'))
        if not os.access(file_path, os.R_OK):
            logging.error(f"No read permission for file: {file_path}")
            flash(f"Error: Cannot access file {filename}.", "error")
            return redirect(url_for('app3.page'))
        logging.info(f"Serving file for download: {file_path}")
        response = send_file(file_path, as_attachment=True, download_name=filename)
        flash("Download Complete!", "success")
        return response
    except Exception as e:
        logging.error(f"Error downloading file {filename}: {str(e)}")
        flash(f"Error: Could not download file {filename}.", "error")
        return redirect(url_for('app6.page'))