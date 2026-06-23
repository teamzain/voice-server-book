"""Flask application factory for the Lens book-scanner recognition service."""

from __future__ import annotations

import logging

from flask import Flask
from flask_cors import CORS

from .config import get_config


def create_app() -> Flask:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    app = Flask(__name__)
    CORS(app)  # mobile client is a separate origin

    app.config["LENS"] = get_config()

    from .routes import bp

    app.register_blueprint(bp)
    return app
