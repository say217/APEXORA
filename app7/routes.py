from flask import Blueprint, render_template, session, redirect, url_for
from functools import wraps
import requests

bp = Blueprint('app7', __name__, template_folder='templates')

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('app2.login'))
        return f(*args, **kwargs)
    return decorated_function

@bp.route('/')
@login_required
def page():
    # NewsAPI key
    api_key = "597d986c60374539aeaf83ca1da405f6"
    url = (
        "https://newsapi.org/v2/everything?"
        "q=Indian+stock+market+OR+BSE+OR+NSE&"
        "language=en&"
        "sortBy=publishedAt&"
        f"apiKey={api_key}"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get("articles", [])[:12]  # Get top 10 articles
    except Exception as e:
        articles = []
    
    return render_template('home7.html', articles=articles)