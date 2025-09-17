from flask import Blueprint, render_template, session, redirect, url_for, request, flash, send_file
from functools import wraps
import yfinance as yf
import json
from datetime import datetime
from dateutil.parser import parse
from pathlib import Path
import os
import uuid

bp = Blueprint('app5', __name__, template_folder='templates')

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('app2.login'))
        return f(*args, **kwargs)
    return decorated_function

def validate_date(date_str):
    try:
        parsed_date = parse(date_str, fuzzy=False)
        current_date = datetime.now().date()
        if parsed_date.date() > current_date:
            return None
        return parsed_date.strftime("%Y-%m-%d")
    except:
        return None

def get_historical_data(ticker, start_date, end_date, interval="1d"):
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(start=start_date, end=end_date, interval=interval)
        if history.empty:
            return None
        history_data = []
        for date, row in history.iterrows():
            history_data.append({
                'Date': date.strftime("%Y-%m-%d"),
                'Open': row['Open'],
                'High': row['High'],
                'Low': row['Low'],
                'Close': row['Close'],
                'Volume': row['Volume'],
                'Dividends': row['Dividends'],
                'Stock Splits': row['Stock Splits']
            })
        return {'Ticker': ticker, 'Historical Data': history_data}
    except yf.YFinanceError:
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def save_to_json_file(data, ticker, start_date, end_date):
    if not data:
        return None
    safe_ticker = ticker.replace(".", "_")
    filename = f"{safe_ticker}_history_{start_date}_{end_date}_{uuid.uuid4().hex}.json"
    try:
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)
        return filename
    except IOError:
        return None

@bp.route('/', methods=['GET', 'POST'])
@login_required
def page():
    form_data = {}
    historical_data = None
    filename = None
    errors = []

    if request.method == 'POST':
        ticker = request.form.get('ticker', '').strip()
        start_date = request.form.get('start_date', '').strip()
        end_date = request.form.get('end_date', '').strip()
        interval = request.form.get('interval', '1d').strip()

        form_data = {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'interval': interval
        }

        # Validate inputs
        if not ticker or (not ticker.isalnum() and "." not in ticker):
            errors.append("Invalid ticker format.")
        else:
            start_date_valid = validate_date(start_date)
            end_date_valid = validate_date(end_date)
            if not start_date_valid or not end_date_valid:
                errors.append("Invalid date format. Use YYYY-MM-DD.")
            elif start_date_valid >= end_date_valid:
                errors.append("Start date must be before end date.")
            else:
                historical_data = get_historical_data(ticker, start_date_valid, end_date_valid, interval)
                if not historical_data:
                    errors.append(f"Could not fetch data for {ticker}. Check ticker, dates, or internet.")
                else:
                    filename = save_to_json_file(historical_data, ticker, start_date_valid, end_date_valid)
                    if not filename:
                        errors.append("Could not save data to JSON file.")

    return render_template('home4.html', 
                         form_data=form_data, 
                         historical_data=historical_data, 
                         filename=filename, 
                         errors=errors)

@bp.route('/download/<filename>')
@login_required
def download_file(filename):
    try:
        file_path = Path(filename)
        if not file_path.exists():
            flash("File not found.", "error")
            return redirect(url_for('app5.page'))
        response = send_file(file_path, as_attachment=True)
        # Clean up the file after sending
        os.remove(file_path)
        return response
    except Exception as e:
        flash(f"Error downloading file: {e}", "error")
        return redirect(url_for('app5.page'))