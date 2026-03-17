"""Shared pytest fixtures for MLExplain tests."""

import pytest

from app import create_app
from config import TestConfig
from models.database import get_db


@pytest.fixture
def app():
    """Create a test Flask application."""
    application = create_app(config_class=TestConfig)
    yield application


@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()


@pytest.fixture
def db_session(app):
    """Provide a database session within app context."""
    with app.app_context():
        database = get_db()
        yield database
        database.session.rollback()


@pytest.fixture
def sample_train_payload():
    """Sample training request payload."""
    return {
        "dataset": "iris",
        "algorithm": "decision_tree",
        "name": "Test Experiment",
        "test_ratio": 0.2,
        "random_state": 42,
        "hyperparameters": {"max_depth": 3},
    }
