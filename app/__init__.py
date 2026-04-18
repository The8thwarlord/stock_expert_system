from flask import Flask

from app.routes import main_bp


def create_app() -> Flask:
    app = Flask(__name__, template_folder="../templates")
    app.register_blueprint(main_bp)
    return app
