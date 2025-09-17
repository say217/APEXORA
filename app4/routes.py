from flask import Blueprint, render_template, session, redirect, url_for, request
from functools import wraps
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os
from uuid import uuid4
import pandas as pd
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import praw
from datetime import datetime, timedelta

# Download NLTK resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

bp = Blueprint('app4', __name__, template_folder='templates', static_folder='static')

# API Keys (replace with your actual keys)
NEWS_API_KEY = '597d986c60374539aeaf83ca1da405f6'
MEDIASTACK_API_KEY = '46ba7d0a470a289131188e093a9ebcc8'  # Obtain from mediastack.com
REDDIT_CLIENT_ID = 'ZMX2YDBKCC6XNpJql9vAmg'
REDDIT_CLIENT_SECRET = 'OzknSOPczqITEd6gF3RUuRTUshtqMQ'
REDDIT_USER_AGENT = 'SentimentAnalysisBot/1.0 by Beneficial-While-616'

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# Initialize sentiment analysis models
vader_analyzer = SentimentIntensityAnalyzer()
transformer_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('app2.login'))
        return f(*args, **kwargs)
    return decorated_function

def fetch_news(ticker):
    """
    Fetch top 5 recent news headlines from NewsAPI and Mediastack.
    """
    headlines = []
    
    # NewsAPI sources
    newsapi_sources = [
        {'url': 'https://newsapi.org/v2/everything', 'params': {
            'q': ticker,
            'sortBy': 'publishedAt',
            'language': 'en',
            'pageSize': 7,
            'apiKey': NEWS_API_KEY
        }},
        {'url': 'https://newsapi.org/v2/top-headlines', 'params': {
            'q': ticker,
            'language': 'en',
            'pageSize': 7,
            'apiKey': NEWS_API_KEY
        }}
    ]
    
    for source in newsapi_sources:
        try:
            response = requests.get(source['url'], params=source['params'])
            response.raise_for_status()
            articles = response.json().get('articles', [])
            headlines.extend([article['title'] for article in articles if 'title' in article])
        except Exception as e:
            continue
    
    # Mediastack source
    mediastack_url = 'http://api.mediastack.com/v1/news'
    mediastack_params = {
        'access_key': MEDIASTACK_API_KEY,
        'keywords': ticker,
        'languages': 'en',
        'limit': 7,
        'sort': 'published_desc'
    }
    try:
        response = requests.get(mediastack_url, params=mediastack_params)
        response.raise_for_status()
        articles = response.json().get('data', [])
        headlines.extend([article['title'] for article in articles if 'title' in article])
    except Exception as e:
        pass
    
    return list(dict.fromkeys(headlines))[:7]  # Remove duplicates and limit to 5

def fetch_reddit_posts(ticker):
    """
    Fetch recent posts from Reddit subreddits related to the ticker.
    """
    subreddits = ['wallstreetbets', 'stocks', 'investing']
    posts = []
    try:
        for subreddit in subreddits:
            for submission in reddit.subreddit(subreddit).search(ticker, limit=5):
                posts.append(submission.title)
        return list(dict.fromkeys(posts))[:7]  # Remove duplicates and limit to 5
    except Exception as e:
        return []

def analyze_sentiment(texts):
    """
    Analyze sentiment using VADER and Transformer models.
    """
    if not texts:
        return []
    results = []
    for text in texts:
        vader_score = vader_analyzer.polarity_scores(text)['compound']
        transformer_result = transformer_analyzer(text)[0]
        transformer_score = transformer_result['score'] if transformer_result['label'] == 'POSITIVE' else -transformer_result['score']
        avg_score = (vader_score + transformer_score) / 2
        results.append((text, avg_score, vader_score, transformer_score))
    return results

def compute_shap_values(texts, scores):
    """
    Compute SHAP values for sentiment analysis using a RandomForest model.
    """
    if not texts or len(texts) < 2:
        return None
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(texts).toarray()
    y = scores
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    feature_names = vectorizer.get_feature_names_out()
    return shap_values, feature_names

def plot_sentiments(news_results, reddit_results, ticker, static_dir):
    """
    Generate multiple plots: Sentiment Bar (News and Reddit), SHAP Summary, and Source Comparison.
    """
    plots = {}
    
    # 1. Sentiment Bar Plot for News
    if news_results:
        headlines = [h[:50] + "..." if len(h) > 50 else h for h, _, _, _ in news_results]
        scores = [s for _, s, _, _ in news_results]
        colors = ['green' if s > 0.05 else 'red' if s < -0.05 else 'gray' for s in scores]
        
        plt.figure(figsize=(10, max(6, len(news_results) * 0.5)))
        bars = plt.barh(headlines, scores, color=colors)
        plt.title(f'News Sentiment for {ticker}')
        plt.xlabel('Sentiment Score')
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        for bar in bars:
            width = bar.get_width()
            label_x = width + 0.02 if width > 0 else width - 0.02
            plt.text(label_x, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                     ha='left' if width > 0 else 'right', va='center')
        plot_filename = f'news_plot_{uuid4().hex}.png'
        plot_path = os.path.join(static_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        plots['news_plot'] = plot_filename
    
    # 2. Sentiment Bar Plot for Reddit Posts
    if reddit_results:
        posts = [p[:50] + "..." if len(p) > 50 else p for p, _, _, _ in reddit_results]
        scores = [s for _, s, _, _ in reddit_results]
        colors = ['green' if s > 0.05 else 'red' if s < -0.05 else 'gray' for s in scores]
        
        plt.figure(figsize=(10, max(6, len(reddit_results) * 0.5)))
        bars = plt.barh(posts, scores, color=colors)
        plt.title(f'Reddit Posts Sentiment for {ticker}')
        plt.xlabel('Sentiment Score')
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        for bar in bars:
            width = bar.get_width()
            label_x = width + 0.02 if width > 0 else width - 0.02
            plt.text(label_x, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                     ha='left' if width > 0 else 'right', va='center')
        plot_filename = f'reddit_plot_{uuid4().hex}.png'
        plot_path = os.path.join(static_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        plots['reddit_plot'] = plot_filename
    
    # 3. SHAP Summary Plot for News
    if news_results:
        texts = [t for t, _, _, _ in news_results]
        scores = [s for _, s, _, _ in news_results]
        shap_data = compute_shap_values(texts, scores)
        if shap_data:
            shap_values, feature_names = shap_data
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, features=pd.DataFrame(np.zeros((len(texts), len(feature_names))), columns=feature_names), feature_names=feature_names, show=False)
            plot_filename = f'shap_plot_{uuid4().hex}.png'
            plot_path = os.path.join(static_dir, plot_filename)
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            plots['shap_plot'] = plot_filename
    
    # 4. Source Comparison Plot
    if news_results or reddit_results:
        sources = ['News', 'Reddit']
        avg_scores = [
            np.mean([s for _, s, _, _ in news_results]) if news_results else 0,
            np.mean([s for _, s, _, _ in reddit_results]) if reddit_results else 0
        ]
        plt.figure(figsize=(6, 4))
        sns.barplot(x=sources, y=avg_scores, palette='viridis')
        plt.title('Average Sentiment by Source')
        plt.ylabel('Average Sentiment Score')
        plot_filename = f'comparison_plot_{uuid4().hex}.png'
        plot_path = os.path.join(static_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        plots['comparison_plot'] = plot_filename
    
    return plots
def fetch_news(ticker):
    """
    Fetch top 5 recent news headlines and their URLs from NewsAPI and Mediastack.
    """
    headlines = []
    
    # NewsAPI sources
    newsapi_sources = [
        {'url': 'https://newsapi.org/v2/everything', 'params': {
            'q': ticker,
            'sortBy': 'publishedAt',
            'language': 'en',
            'pageSize': 7,
            'apiKey': NEWS_API_KEY
        }},
        {'url': 'https://newsapi.org/v2/top-headlines', 'params': {
            'q': ticker,
            'language': 'en',
            'pageSize':7,
            'apiKey': NEWS_API_KEY
        }}
    ]
    
    for source in newsapi_sources:
        try:
            response = requests.get(source['url'], params=source['params'])
            response.raise_for_status()
            articles = response.json().get('articles', [])
            headlines.extend([(article['title'], article.get('url', '#')) for article in articles if 'title' in article])
        except Exception as e:
            continue
    
    # Mediastack source
    mediastack_url = 'http://api.mediastack.com/v1/news'
    mediastack_params = {
        'access_key': MEDIASTACK_API_KEY,
        'keywords': ticker,
        'languages': 'en',
        'limit': 5,
        'sort': 'published_desc'
    }
    try:
        response = requests.get(mediastack_url, params=mediastack_params)
        response.raise_for_status()
        articles = response.json().get('data', [])
        headlines.extend([(article['title'], article.get('url', '#')) for article in articles if 'title' in article])
    except Exception as e:
        pass
    
    return list(dict.fromkeys(headlines))[:7]  # Remove duplicates and limit to 5
def fetch_reddit_posts(ticker):
    """
    Fetch recent posts and their URLs from Reddit subreddits related to the ticker.
    """
    subreddits = ['wallstreetbets', 'stocks', 'investing']
    posts = []
    try:
        for subreddit in subreddits:
            for submission in reddit.subreddit(subreddit).search(ticker, limit=5):
                posts.append((submission.title, submission.url))
        return list(dict.fromkeys(posts))[:7]  # Remove duplicates and limit to 5
    except Exception as e:
        return []
    
@bp.route('/', methods=['GET', 'POST'])
@login_required
def page():
    plot_urls = {}
    news_headlines = []
    reddit_posts = []
    error = None
    ticker = None  # Initialize ticker for template context

    if request.method == 'POST':
        ticker = request.form.get('ticker', '').strip()
        if ticker:
            # Fetch and analyze news
            news_headlines = fetch_news(ticker)
            if news_headlines:
                news_sentiments = analyze_sentiment([headline[0] for headline in news_headlines])
            else:
                news_sentiments = []
            
            # Fetch and analyze Reddit posts
            reddit_posts = fetch_reddit_posts(ticker)
            if reddit_posts:
                reddit_sentiments = analyze_sentiment([post[0] for post in reddit_posts])
            else:
                reddit_sentiments = []
            
            if news_headlines or reddit_posts:
                static_dir = bp.static_folder
                plots = plot_sentiments(news_sentiments, reddit_sentiments, ticker, static_dir)
                for plot_type, filename in plots.items():
                    plot_urls[plot_type] = url_for('app4.static', filename=filename)
            else:
                error = f"No data found for '{ticker}'."
        else:
            error = "Please enter a valid ticker."

    return render_template('home3.html', 
                         plot_urls=plot_urls, 
                         news_headlines=news_headlines, 
                         reddit_posts=reddit_posts, 
                         error=error,
                         ticker=ticker)