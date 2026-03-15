"""REST API endpoints for MLExplain.

Provides JSON endpoints for dataset listing, model training,
experiment retrieval, feature importance, confusion matrix,
prediction, and model comparison.
"""

import json
import logging

from flask import Blueprint, jsonify, request

from models.database import get_db
from models.schemas import Experiment, Dataset, ModelResult
from services.ml_engine import MLEngine
from services.datasets import load_dataset_by_name, get_available_datasets

logger = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__, url_prefix="/api")


# -- Health ------------------------------------------------------------------

@api_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "mlexplain"})


# -- Datasets ----------------------------------------------------------------

@api_bp.route("/datasets", methods=["GET"])
def list_datasets():
    """List all available built-in datasets."""
    datasets = get_available_datasets()
    return jsonify({"datasets": datasets})


@api_bp.route("/datasets/<name>", methods=["GET"])
def get_dataset(name):
    """Get dataset info and a preview of the first rows."""
    try:
        info = load_dataset_by_name(name)
        preview_rows = 5
        preview = {
            "feature_names": info["feature_names"],
            "target_names": info["target_names"],
            "num_samples": info["num_samples"],
            "num_features": info["num_features"],
            "num_classes": info["num_classes"],
            "sample_data": info["data"][:preview_rows].tolist(),
            "sample_targets": info["target"][:preview_rows].tolist(),
        }
        return jsonify(preview)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404


# -- Training ----------------------------------------------------------------

@api_bp.route("/train", methods=["POST"])
def train_model():
    """Train a model on a dataset.

    Expected JSON body:
    {
        "dataset": "iris",
        "algorithm": "random_forest",
        "name": "My Experiment",
        "test_ratio": 0.2,
        "random_state": 42,
        "hyperparameters": {"n_estimators": 100, "max_depth": 5}
    }
    """
    payload = request.get_json(silent=True) or {}

    dataset_name = payload.get("dataset", "iris")
    algorithm = payload.get("algorithm", "random_forest")
    experiment_name = payload.get("name", f"{algorithm} on {dataset_name}")
    test_ratio = float(payload.get("test_ratio", 0.2))
    random_state = int(payload.get("random_state", 42))
    hyperparameters = payload.get("hyperparameters", {})

    # Validate
    valid_algorithms = [
        "decision_tree", "random_forest", "svm", "knn", "logistic_regression",
    ]
    if algorithm not in valid_algorithms:
        return jsonify({"error": f"Unknown algorithm: {algorithm}",
                        "valid": valid_algorithms}), 400

    try:
        dataset_info = load_dataset_by_name(dataset_name)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if test_ratio <= 0 or test_ratio >= 1:
        return jsonify({"error": "test_ratio must be between 0 and 1"}), 400

    db = get_db()

    # Ensure dataset record exists
    ds_record = Dataset.query.filter_by(name=dataset_name).first()
    if ds_record is None:
        ds_record = Dataset(
            name=dataset_name,
            num_samples=dataset_info["num_samples"],
            num_features=dataset_info["num_features"],
            num_classes=dataset_info["num_classes"],
            feature_names=json.dumps(dataset_info["feature_names"]),
            target_names=json.dumps(dataset_info["target_names"]),
        )
        db.session.add(ds_record)
        db.session.flush()

    # Train
    engine = MLEngine()
    result = engine.train(
        data=dataset_info["data"],
        target=dataset_info["target"],
        feature_names=dataset_info["feature_names"],
        target_names=dataset_info["target_names"],
        algorithm=algorithm,
        hyperparameters=hyperparameters,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    # Persist experiment
    experiment = Experiment(
        name=experiment_name,
        dataset_id=ds_record.id,
        algorithm=algorithm,
        hyperparameters=json.dumps(hyperparameters),
        test_ratio=test_ratio,
        random_state=random_state,
    )
    db.session.add(experiment)
    db.session.flush()

    model_result = ModelResult(
        experiment_id=experiment.id,
        accuracy=result["accuracy"],
        precision=result["precision"],
        recall=result["recall"],
        f1_score=result["f1_score"],
        confusion_matrix=json.dumps(result["confusion_matrix"]),
        feature_importance=json.dumps(result["feature_importance"]),
        classification_report=json.dumps(result["classification_report"]),
        train_samples=result["train_samples"],
        test_samples=result["test_samples"],
        training_time_ms=result["training_time_ms"],
    )
    db.session.add(model_result)
    db.session.commit()

    logger.info("Trained %s on %s -- accuracy=%.4f",
                algorithm, dataset_name, result["accuracy"])

    return jsonify(experiment.to_dict()), 201


# -- Experiments -------------------------------------------------------------

@api_bp.route("/experiments", methods=["GET"])
def list_experiments():
    """List all experiments, newest first."""
    experiments = Experiment.query.order_by(Experiment.created_at.desc()).all()
    return jsonify({"experiments": [e.to_dict() for e in experiments]})


@api_bp.route("/experiments/<int:exp_id>", methods=["GET"])
def get_experiment(exp_id):
    """Get a single experiment by ID."""
    experiment = Experiment.query.get(exp_id)
    if experiment is None:
        return jsonify({"error": "Experiment not found"}), 404
    return jsonify(experiment.to_dict())


@api_bp.route("/experiments/<int:exp_id>", methods=["DELETE"])
def delete_experiment(exp_id):
    """Delete an experiment and its results."""
    db = get_db()
    experiment = Experiment.query.get(exp_id)
    if experiment is None:
        return jsonify({"error": "Experiment not found"}), 404
    if experiment.result:
        db.session.delete(experiment.result)
    db.session.delete(experiment)
    db.session.commit()
    return jsonify({"message": f"Experiment {exp_id} deleted"}), 200


# -- Explanation endpoints ---------------------------------------------------

@api_bp.route("/experiments/<int:exp_id>/importance", methods=["GET"])
def feature_importance(exp_id):
    """Get feature importance for an experiment."""
    experiment = Experiment.query.get(exp_id)
    if experiment is None or experiment.result is None:
        return jsonify({"error": "Experiment not found"}), 404

    importance_data = json.loads(experiment.result.feature_importance) \
        if experiment.result.feature_importance else []

    return jsonify({
        "experiment_id": exp_id,
        "algorithm": experiment.algorithm,
        "feature_importance": importance_data,
    })


@api_bp.route("/experiments/<int:exp_id>/confusion", methods=["GET"])
def confusion_matrix(exp_id):
    """Get confusion matrix for an experiment."""
    experiment = Experiment.query.get(exp_id)
    if experiment is None or experiment.result is None:
        return jsonify({"error": "Experiment not found"}), 404

    cm_data = json.loads(experiment.result.confusion_matrix) \
        if experiment.result.confusion_matrix else []

    ds = experiment.dataset
    target_names = json.loads(ds.target_names) if ds and ds.target_names else []

    return jsonify({
        "experiment_id": exp_id,
        "confusion_matrix": cm_data,
        "labels": target_names,
    })


@api_bp.route("/experiments/<int:exp_id>/metrics", methods=["GET"])
def detailed_metrics(exp_id):
    """Get all metrics for an experiment."""
    experiment = Experiment.query.get(exp_id)
    if experiment is None or experiment.result is None:
        return jsonify({"error": "Experiment not found"}), 404

    result = experiment.result
    report = json.loads(result.classification_report) \
        if result.classification_report else {}

    return jsonify({
        "experiment_id": exp_id,
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1_score": result.f1_score,
        "classification_report": report,
        "train_samples": result.train_samples,
        "test_samples": result.test_samples,
        "training_time_ms": result.training_time_ms,
    })


# -- Prediction --------------------------------------------------------------

@api_bp.route("/predict/<int:exp_id>", methods=["POST"])
def predict(exp_id):
    """Predict using a trained model.

    Expected JSON body:
    {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    """
    experiment = Experiment.query.get(exp_id)
    if experiment is None:
        return jsonify({"error": "Experiment not found"}), 404

    payload = request.get_json(silent=True) or {}
    features = payload.get("features")
    if not features or not isinstance(features, list):
        return jsonify({"error": "features must be a list of numbers"}), 400

    ds = experiment.dataset
    dataset_info = load_dataset_by_name(ds.name)

    # Re-train the model to get the fitted estimator
    engine = MLEngine()
    result = engine.train(
        data=dataset_info["data"],
        target=dataset_info["target"],
        feature_names=dataset_info["feature_names"],
        target_names=dataset_info["target_names"],
        algorithm=experiment.algorithm,
        hyperparameters=experiment.get_hyperparameters(),
        test_ratio=experiment.test_ratio,
        random_state=experiment.random_state,
    )

    prediction = engine.predict(features)
    target_names = dataset_info["target_names"]

    return jsonify({
        "experiment_id": exp_id,
        "features": features,
        "prediction": int(prediction["class"]),
        "predicted_label": target_names[int(prediction["class"])]
            if int(prediction["class"]) < len(target_names) else str(prediction["class"]),
        "probabilities": prediction.get("probabilities"),
    })


# -- Comparison --------------------------------------------------------------

@api_bp.route("/compare", methods=["POST"])
def compare_models():
    """Compare multiple models on the same dataset.

    Expected JSON body:
    {
        "dataset": "iris",
        "algorithms": ["decision_tree", "random_forest", "svm"],
        "test_ratio": 0.2,
        "random_state": 42
    }
    """
    payload = request.get_json(silent=True) or {}

    dataset_name = payload.get("dataset", "iris")
    algorithms = payload.get("algorithms", [
        "decision_tree", "random_forest", "logistic_regression",
    ])
    test_ratio = float(payload.get("test_ratio", 0.2))
    random_state = int(payload.get("random_state", 42))

    try:
        dataset_info = load_dataset_by_name(dataset_name)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    results = []
    engine = MLEngine()

    for algo in algorithms:
        try:
            result = engine.train(
                data=dataset_info["data"],
                target=dataset_info["target"],
                feature_names=dataset_info["feature_names"],
                target_names=dataset_info["target_names"],
                algorithm=algo,
                hyperparameters={},
                test_ratio=test_ratio,
                random_state=random_state,
            )
            results.append({
                "algorithm": algo,
                "accuracy": round(result["accuracy"], 4),
                "precision": round(result["precision"], 4),
                "recall": round(result["recall"], 4),
                "f1_score": round(result["f1_score"], 4),
                "training_time_ms": round(result["training_time_ms"], 2),
            })
        except Exception as exc:
            results.append({
                "algorithm": algo,
                "error": str(exc),
            })

    return jsonify({
        "dataset": dataset_name,
        "test_ratio": test_ratio,
        "comparisons": results,
    })
