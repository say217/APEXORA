from flask import Blueprint
from .routes import bp as routes_bp

def create_app():
    return routes_bp