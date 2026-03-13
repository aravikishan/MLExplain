"""SQLAlchemy database setup for MLExplain."""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def init_db(app):
    """Initialise the database with the Flask app."""
    db.init_app(app)
    with app.app_context():
        from models.schemas import Experiment, Dataset, ModelResult  # noqa: F401
        db.create_all()


def get_db():
    """Return the SQLAlchemy database instance."""
    return db
