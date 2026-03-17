"""API endpoint tests for MLExplain."""

import json
import pytest


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert data["service"] == "mlexplain"


class TestDatasetEndpoints:
    """Tests for dataset listing and detail endpoints."""

    def test_list_datasets(self, client):
        resp = client.get("/api/datasets")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "datasets" in data
        names = [d["name"] for d in data["datasets"]]
        assert "iris" in names
        assert "wine" in names

    def test_get_dataset_info(self, client):
        resp = client.get("/api/datasets/iris")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["num_samples"] == 150
        assert data["num_features"] == 4
        assert data["num_classes"] == 3

    def test_get_unknown_dataset_404(self, client):
        resp = client.get("/api/datasets/nonexistent")
        assert resp.status_code == 404


class TestTrainEndpoint:
    """Tests for model training."""

    def test_train_decision_tree(self, client, sample_train_payload):
        resp = client.post(
            "/api/train",
            data=json.dumps(sample_train_payload),
            content_type="application/json",
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["algorithm"] == "decision_tree"
        assert data["result"] is not None
        assert data["result"]["accuracy"] > 0.5

    def test_train_random_forest(self, client):
        payload = {
            "dataset": "iris",
            "algorithm": "random_forest",
            "name": "RF Test",
            "test_ratio": 0.3,
        }
        resp = client.post(
            "/api/train",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["result"]["accuracy"] > 0.5
        assert data["result"]["feature_importance"] is not None

    def test_train_invalid_algorithm(self, client):
        payload = {"dataset": "iris", "algorithm": "invalid_algo"}
        resp = client.post(
            "/api/train",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_train_invalid_test_ratio(self, client):
        payload = {"dataset": "iris", "algorithm": "svm", "test_ratio": 1.5}
        resp = client.post(
            "/api/train",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 400


class TestExperimentEndpoints:
    """Tests for experiment CRUD."""

    def test_list_experiments(self, client, sample_train_payload):
        # Train first
        client.post(
            "/api/train",
            data=json.dumps(sample_train_payload),
            content_type="application/json",
        )
        resp = client.get("/api/experiments")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["experiments"]) >= 1

    def test_delete_experiment(self, client, sample_train_payload):
        # Train
        train_resp = client.post(
            "/api/train",
            data=json.dumps(sample_train_payload),
            content_type="application/json",
        )
        exp_id = train_resp.get_json()["id"]
        # Delete
        del_resp = client.delete(f"/api/experiments/{exp_id}")
        assert del_resp.status_code == 200


class TestCompareEndpoint:
    """Tests for model comparison."""

    def test_compare_models(self, client):
        payload = {
            "dataset": "iris",
            "algorithms": ["decision_tree", "logistic_regression"],
            "test_ratio": 0.2,
        }
        resp = client.post(
            "/api/compare",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["comparisons"]) == 2
        for comp in data["comparisons"]:
            assert "accuracy" in comp
