"""SQLAlchemy ORM models for MLExplain.

Tables
------
- Experiment : a single training run
- Dataset    : metadata about loaded datasets
- ModelResult: detailed metrics for an experiment
"""

import json
from datetime import datetime, timezone

from models.database import db


class Dataset(db.Model):
    """Metadata for a loaded dataset."""

    __tablename__ = "datasets"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(64), unique=True, nullable=False)
    num_samples = db.Column(db.Integer, nullable=False, default=0)
    num_features = db.Column(db.Integer, nullable=False, default=0)
    num_classes = db.Column(db.Integer, nullable=False, default=0)
    feature_names = db.Column(db.Text, nullable=True)  # JSON list
    target_names = db.Column(db.Text, nullable=True)    # JSON list
    created_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    experiments = db.relationship(
        "Experiment", backref="dataset", lazy=True,
    )

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "num_samples": self.num_samples,
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "feature_names": json.loads(self.feature_names) if self.feature_names else [],
            "target_names": json.loads(self.target_names) if self.target_names else [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Experiment(db.Model):
    """A single model training experiment."""

    __tablename__ = "experiments"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(128), nullable=False)
    dataset_id = db.Column(
        db.Integer, db.ForeignKey("datasets.id"), nullable=False,
    )
    algorithm = db.Column(db.String(64), nullable=False)
    hyperparameters = db.Column(db.Text, nullable=True)  # JSON dict
    test_ratio = db.Column(db.Float, nullable=False, default=0.2)
    random_state = db.Column(db.Integer, nullable=False, default=42)
    created_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    result = db.relationship(
        "ModelResult", backref="experiment", uselist=False, lazy=True,
    )

    def get_hyperparameters(self):
        if self.hyperparameters:
            return json.loads(self.hyperparameters)
        return {}

    def to_dict(self):
        result_dict = self.result.to_dict() if self.result else None
        return {
            "id": self.id,
            "name": self.name,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset.name if self.dataset else None,
            "algorithm": self.algorithm,
            "hyperparameters": self.get_hyperparameters(),
            "test_ratio": self.test_ratio,
            "random_state": self.random_state,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "result": result_dict,
        }


class ModelResult(db.Model):
    """Detailed metrics and artefacts for a trained model."""

    __tablename__ = "model_results"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    experiment_id = db.Column(
        db.Integer, db.ForeignKey("experiments.id"), nullable=False, unique=True,
    )

    # Core metrics
    accuracy = db.Column(db.Float, nullable=False, default=0.0)
    precision = db.Column(db.Float, nullable=False, default=0.0)
    recall = db.Column(db.Float, nullable=False, default=0.0)
    f1_score = db.Column(db.Float, nullable=False, default=0.0)

    # Serialised artefacts (JSON)
    confusion_matrix = db.Column(db.Text, nullable=True)
    feature_importance = db.Column(db.Text, nullable=True)
    classification_report = db.Column(db.Text, nullable=True)

    # Training metadata
    train_samples = db.Column(db.Integer, nullable=False, default=0)
    test_samples = db.Column(db.Integer, nullable=False, default=0)
    training_time_ms = db.Column(db.Float, nullable=False, default=0.0)

    created_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "confusion_matrix": json.loads(self.confusion_matrix)
                if self.confusion_matrix else None,
            "feature_importance": json.loads(self.feature_importance)
                if self.feature_importance else None,
            "classification_report": json.loads(self.classification_report)
                if self.classification_report else None,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "training_time_ms": round(self.training_time_ms, 2),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
