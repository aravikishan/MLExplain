"""Application configuration for MLExplain."""

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    """Default Flask configuration."""

    SECRET_KEY = os.environ.get("SECRET_KEY", "mlexplain-dev-secret-key")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{os.path.join(BASE_DIR, 'instance', 'mlexplain.db')}",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    PORT = int(os.environ.get("PORT", 8006))
    DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

    # ML defaults
    DEFAULT_TEST_RATIO = 0.2
    DEFAULT_RANDOM_STATE = 42
    MAX_FEATURE_IMPORTANCE_BARS = 20

    # Supported algorithms
    ALGORITHMS = [
        "decision_tree",
        "random_forest",
        "svm",
        "knn",
        "logistic_regression",
    ]

    # Built-in datasets
    DATASETS = ["iris", "wine", "breast_cancer", "digits"]


class TestConfig(Config):
    """Testing configuration -- uses in-memory SQLite."""

    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SECRET_KEY = "test-secret-key"
