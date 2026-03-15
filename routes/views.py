"""HTML-serving view routes for MLExplain."""

import json

from flask import Blueprint, render_template

from models.schemas import Experiment, Dataset
from services.datasets import get_available_datasets

views_bp = Blueprint("views", __name__)


@views_bp.route("/")
def index():
    """Dashboard -- overview of experiments and datasets."""
    experiments = Experiment.query.order_by(
        Experiment.created_at.desc()
    ).limit(10).all()
    datasets = get_available_datasets()
    total_experiments = Experiment.query.count()

    # Compute average accuracy across all experiments that have results
    all_experiments = Experiment.query.all()
    accuracies = [
        e.result.accuracy for e in all_experiments if e.result is not None
    ]
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0

    return render_template(
        "index.html",
        experiments=experiments,
        datasets=datasets,
        total_experiments=total_experiments,
        avg_accuracy=avg_accuracy,
    )


@views_bp.route("/train")
def train():
    """Train a new model page."""
    datasets = get_available_datasets()
    algorithms = [
        {"value": "decision_tree", "label": "Decision Tree",
         "params": ["max_depth", "min_samples_split"]},
        {"value": "random_forest", "label": "Random Forest",
         "params": ["n_estimators", "max_depth"]},
        {"value": "svm", "label": "SVM",
         "params": ["C", "kernel"]},
        {"value": "knn", "label": "K-Nearest Neighbours",
         "params": ["n_neighbors", "metric"]},
        {"value": "logistic_regression", "label": "Logistic Regression",
         "params": ["C", "solver", "max_iter"]},
    ]
    return render_template(
        "train.html", datasets=datasets, algorithms=algorithms,
    )


@views_bp.route("/explain/<int:exp_id>")
def explain(exp_id):
    """Model explanation page for a specific experiment."""
    experiment = Experiment.query.get_or_404(exp_id)
    ds = experiment.dataset

    feature_names = json.loads(ds.feature_names) if ds.feature_names else []
    target_names = json.loads(ds.target_names) if ds.target_names else []

    importance = []
    cm = []
    report = {}
    if experiment.result:
        if experiment.result.feature_importance:
            importance = json.loads(experiment.result.feature_importance)
        if experiment.result.confusion_matrix:
            cm = json.loads(experiment.result.confusion_matrix)
        if experiment.result.classification_report:
            report = json.loads(experiment.result.classification_report)

    return render_template(
        "explain.html",
        experiment=experiment,
        feature_names=feature_names,
        target_names=target_names,
        importance=importance,
        confusion_matrix=cm,
        classification_report=report,
    )


@views_bp.route("/compare")
def compare():
    """Model comparison page."""
    datasets = get_available_datasets()
    experiments = Experiment.query.order_by(
        Experiment.created_at.desc()
    ).all()
    return render_template(
        "compare.html", datasets=datasets, experiments=experiments,
    )


@views_bp.route("/about")
def about():
    """About page with project information."""
    return render_template("about.html")
