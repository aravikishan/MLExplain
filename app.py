"""MLExplain -- Flask application entry point.

An interactive machine learning model explainer providing scikit-learn
model training, feature importance extraction, confusion matrix
computation, and model comparison utilities.
"""

import logging
import os
import sys

from flask import Flask

from config import Config, TestConfig
from models.database import init_db, get_db
from routes.api import api_bp
from routes.views import views_bp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(config_class=None) -> Flask:
    """Application factory -- create and configure the Flask app."""
    app = Flask(__name__)

    if config_class is None:
        config_class = Config
    app.config.from_object(config_class)

    # Ensure instance directory exists for SQLite
    os.makedirs(os.path.join(app.root_path, "instance"), exist_ok=True)

    # Initialize database
    init_db(app)

    # Register blueprints
    app.register_blueprint(api_bp)
    app.register_blueprint(views_bp)

    # Seed the database with sample experiments when not testing
    if not app.config.get("TESTING"):
        with app.app_context():
            try:
                from services.datasets import seed_database
                count = seed_database(get_db())
                if count:
                    logger.info("Seeded database with %d sample experiments", count)
            except Exception as exc:
                logger.warning("Could not seed database: %s", exc)

    logger.info("MLExplain application created successfully")
    return app


# -- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    app = create_app()
    run_port = int(os.environ.get("PORT", 8006))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info("Starting MLExplain on http://0.0.0.0:%d", run_port)
    app.run(host="0.0.0.0", port=run_port, debug=debug)
