 from app1 import create_app as create_app1
from app2 import create_app as create_app2
from app3 import create_app as create_app3
from app4 import create_app as create_app4
from app5 import create_app as create_app5
from app6 import create_app as create_app6
from app7 import create_app as create_app7
from app8 import create_app as create_app8    
from app9 import create_app as create_app9
from flask import Flask, redirect, url_for, session

app = Flask(__name__)

# Configuration settings
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key'
# Add to run.py before app.run()

# Register blueprints
app.register_blueprint(create_app1(), url_prefix='/app1')
app.register_blueprint(create_app2(), url_prefix='/app2')
app.register_blueprint(create_app3(), url_prefix='/app3')
app.register_blueprint(create_app4(), url_prefix='/app4')
app.register_blueprint(create_app5(), url_prefix='/app5')
app.register_blueprint(create_app6(), url_prefix='/app6')
app.register_blueprint(create_app7(), url_prefix='/app7')
app.register_blueprint(create_app8(), url_prefix='/app8')
app.register_blueprint(create_app9(), url_prefix='/app9')
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('app1.home'))
    return redirect(url_for('app2.login'))

if __name__ == '__main__':
    app.run(debug=True)
    
print(app.blueprints)
