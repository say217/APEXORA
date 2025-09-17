from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from .models import db, User
from werkzeug.security import generate_password_hash, check_password_hash

bp = Blueprint('app2', __name__, template_folder='templates')

@bp.record
def record_params(setup_state):
    app = setup_state.app
    db.init_app(app)
    with app.app_context():
        db.create_all()

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            return redirect(url_for('app1.home'))
        flash('Invalid credentials')
    
    return render_template('login.html')

@bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return render_template('signup.html')
        
        user = User(
            name=name,
            email=email,
            password=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        
        session['user_id'] = user.id
        return redirect(url_for('app1.home'))
    
    return render_template('signup.html')

@bp.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('app2.login'))