from flask import Flask
from flask_cors import CORS
def create_app():
    app = Flask(__name__)
    CORS(app)
    
    from app.controllers.parsing_controller import parsing_bp
    app.register_blueprint(parsing_bp, url_prefix='/api')
    
    return app