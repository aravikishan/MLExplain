"""Service layer tests for MLExplain."""

import numpy as np
import pytest

from services.ml_engine import MLEngine
from services.datasets import load_dataset_by_name, get_available_datasets


class TestDatasets:
    """Tests for the dataset loading service."""

    def test_get_available_datasets(self):
        datasets = get_available_datasets()
        assert len(datasets) == 4
        names = [d["name"] for d in datasets]
        assert "iris" in names
        assert "wine" in names
        assert "breast_cancer" in names
        assert "digits" in names

    def test_load_iris(self):
        info = load_dataset_by_name("iris")
        assert info["num_samples"] == 150
        assert info["num_features"] == 4
        assert info["num_classes"] == 3
        assert len(info["feature_names"]) == 4
        assert isinstance(info["data"], np.ndarray)

    def test_load_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset_by_name("nonexistent")


class TestMLEngine:
    """Tests for the ML engine."""

    @pytest.fixture
    def iris_data(self):
        return load_dataset_by_name("iris")

    def test_train_decision_tree(self, iris_data):
        engine = MLEngine()
        result = engine.train(
            data=iris_data["data"],
            target=iris_data["target"],
            feature_names=iris_data["feature_names"],
            target_names=iris_data["target_names"],
            algorithm="decision_tree",
        )
        assert result["accuracy"] > 0.7
        assert len(result["feature_importance"]) == 4
        assert len(result["confusion_matrix"]) == 3

    def test_train_random_forest(self, iris_data):
        engine = MLEngine()
        result = engine.train(
            data=iris_data["data"],
            target=iris_data["target"],
            feature_names=iris_data["feature_names"],
            target_names=iris_data["target_names"],
            algorithm="random_forest",
        )
        assert result["accuracy"] > 0.7
        assert result["f1_score"] > 0.5

    def test_train_svm(self, iris_data):
        engine = MLEngine()
        result = engine.train(
            data=iris_data["data"],
            target=iris_data["target"],
            feature_names=iris_data["feature_names"],
            target_names=iris_data["target_names"],
            algorithm="svm",
        )
        assert result["accuracy"] > 0.7

    def test_train_knn(self, iris_data):
        engine = MLEngine()
        result = engine.train(
            data=iris_data["data"],
            target=iris_data["target"],
            feature_names=iris_data["feature_names"],
            target_names=iris_data["target_names"],
            algorithm="knn",
        )
        assert result["accuracy"] > 0.7

    def test_train_logistic_regression(self, iris_data):
        engine = MLEngine()
        result = engine.train(
            data=iris_data["data"],
            target=iris_data["target"],
            feature_names=iris_data["feature_names"],
            target_names=iris_data["target_names"],
            algorithm="logistic_regression",
        )
        assert result["accuracy"] > 0.7

    def test_predict(self, iris_data):
        engine = MLEngine()
        engine.train(
            data=iris_data["data"],
            target=iris_data["target"],
            feature_names=iris_data["feature_names"],
            target_names=iris_data["target_names"],
            algorithm="random_forest",
        )
        prediction = engine.predict([5.1, 3.5, 1.4, 0.2])
        assert "class" in prediction
        assert prediction["class"] in [0, 1, 2]

    def test_result_has_classification_report(self, iris_data):
        engine = MLEngine()
        result = engine.train(
            data=iris_data["data"],
            target=iris_data["target"],
            feature_names=iris_data["feature_names"],
            target_names=iris_data["target_names"],
            algorithm="decision_tree",
        )
        report = result["classification_report"]
        assert "accuracy" in report
        assert result["train_samples"] > 0
        assert result["test_samples"] > 0

    def test_predict_without_training_raises(self):
        engine = MLEngine()
        with pytest.raises(RuntimeError, match="No model"):
            engine.predict([1.0, 2.0, 3.0, 4.0])
