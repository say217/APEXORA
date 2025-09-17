from flask import Blueprint, render_template, session, redirect, url_for
from functools import wraps

bp = Blueprint('app1', __name__, template_folder='templates')

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('app2.login'))
        return f(*args, **kwargs)
    return decorated_function

@bp.route('/')
@login_required
def home():
    return render_template('home.html')

@bp.route('/description')
@login_required
def description():
    return render_template('description.html')
