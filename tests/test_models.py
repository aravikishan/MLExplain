"""Database model tests for MLExplain."""

import json
import pytest

from models.schemas import Dataset, Experiment, ModelResult


class TestDatasetModel:
    """Tests for the Dataset ORM model."""

    def test_create_dataset(self, db_session):
        ds = Dataset(
            name="test_ds",
            num_samples=100,
            num_features=4,
            num_classes=3,
            feature_names=json.dumps(["f1", "f2", "f3", "f4"]),
            target_names=json.dumps(["a", "b", "c"]),
        )
        db_session.session.add(ds)
        db_session.session.commit()

        fetched = Dataset.query.filter_by(name="test_ds").first()
        assert fetched is not None
        assert fetched.num_samples == 100

    def test_dataset_to_dict(self, db_session):
        ds = Dataset(
            name="dict_test",
            num_samples=50,
            num_features=2,
            num_classes=2,
            feature_names=json.dumps(["x", "y"]),
            target_names=json.dumps(["pos", "neg"]),
        )
        db_session.session.add(ds)
        db_session.session.commit()

        d = ds.to_dict()
        assert d["name"] == "dict_test"
        assert d["feature_names"] == ["x", "y"]
        assert d["target_names"] == ["pos", "neg"]


class TestExperimentModel:
    """Tests for the Experiment ORM model."""

    def test_create_experiment_with_result(self, db_session):
        ds = Dataset(
            name="exp_ds", num_samples=100, num_features=4,
            num_classes=3, feature_names="[]", target_names="[]",
        )
        db_session.session.add(ds)
        db_session.session.flush()

        exp = Experiment(
            name="Test Exp",
            dataset_id=ds.id,
            algorithm="random_forest",
            hyperparameters=json.dumps({"n_estimators": 50}),
            test_ratio=0.2,
            random_state=42,
        )
        db_session.session.add(exp)
        db_session.session.flush()

        result = ModelResult(
            experiment_id=exp.id,
            accuracy=0.95,
            precision=0.94,
            recall=0.93,
            f1_score=0.935,
            confusion_matrix=json.dumps([[30, 2], [1, 27]]),
            train_samples=80,
            test_samples=20,
            training_time_ms=12.5,
        )
        db_session.session.add(result)
        db_session.session.commit()

        fetched = Experiment.query.filter_by(name="Test Exp").first()
        assert fetched is not None
        assert fetched.result is not None
        assert fetched.result.accuracy == 0.95

    def test_experiment_to_dict(self, db_session):
        ds = Dataset(
            name="dict_exp_ds", num_samples=50, num_features=2,
            num_classes=2, feature_names="[]", target_names="[]",
        )
        db_session.session.add(ds)
        db_session.session.flush()

        exp = Experiment(
            name="Dict Exp", dataset_id=ds.id, algorithm="svm",
            hyperparameters=json.dumps({"C": 1.0}),
        )
        db_session.session.add(exp)
        db_session.session.commit()

        d = exp.to_dict()
        assert d["algorithm"] == "svm"
        assert d["hyperparameters"] == {"C": 1.0}
