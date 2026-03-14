"""Core ML engine -- training, prediction, and explanation.

Supports Decision Tree, Random Forest, SVM, KNN, and Logistic
Regression from scikit-learn.  Computes accuracy, precision, recall,
F1-score, confusion matrix, feature importance (tree-based and
permutation), and classification report.
"""

import time
import logging
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MLEngine:
    """Manages model training, evaluation, and explanation."""

    ALGORITHM_MAP = {
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "svm": SVC,
        "knn": KNeighborsClassifier,
        "logistic_regression": LogisticRegression,
    }

    DEFAULT_HYPERPARAMS: dict[str, dict[str, Any]] = {
        "decision_tree": {"max_depth": 5, "min_samples_split": 2, "random_state": 42},
        "random_forest": {"n_estimators": 100, "max_depth": 5, "random_state": 42},
        "svm": {"C": 1.0, "kernel": "rbf", "probability": True, "random_state": 42},
        "knn": {"n_neighbors": 5, "metric": "euclidean"},
        "logistic_regression": {
            "C": 1.0, "solver": "lbfgs", "max_iter": 1000, "random_state": 42,
        },
    }

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names: list[str] = []
        self.target_names: list[str] = []
        self._X_test = None
        self._y_test = None

    # ----- public API -------------------------------------------------------

    def train(
        self,
        data: np.ndarray,
        target: np.ndarray,
        feature_names: list[str],
        target_names: list[str],
        algorithm: str,
        hyperparameters: dict[str, Any] | None = None,
        test_ratio: float = 0.2,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Train a model and return comprehensive results."""
        self.feature_names = list(feature_names)
        self.target_names = list(target_names)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            data, target, test_size=test_ratio, random_state=random_state,
            stratify=target,
        )

        # Scale features for SVM, KNN, Logistic Regression
        needs_scaling = algorithm in ("svm", "knn", "logistic_regression")
        if needs_scaling:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        # Build model
        params = dict(self.DEFAULT_HYPERPARAMS.get(algorithm, {}))
        if hyperparameters:
            # Filter to valid params for the estimator
            valid_keys = self.ALGORITHM_MAP[algorithm]().get_params().keys()
            for key, value in hyperparameters.items():
                if key in valid_keys:
                    params[key] = value

        model_class = self.ALGORITHM_MAP[algorithm]
        self.model = model_class(**params)

        # Train
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time_ms = (time.time() - start_time) * 1000

        # Predict
        y_pred = self.model.predict(X_test)

        # Store for later prediction
        self._X_test = X_test
        self._y_test = y_test

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        avg_method = "weighted"
        prec = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=self.target_names if len(self.target_names) > 0 else None,
            output_dict=True,
            zero_division=0,
        )
        # Convert numpy values in report
        report = self._sanitise_report(report)

        # Feature importance
        importance = self._compute_feature_importance(X_test, y_test)

        logger.info(
            "Trained %s: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f (%.1f ms)",
            algorithm, acc, prec, rec, f1, training_time_ms,
        )

        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "confusion_matrix": cm,
            "feature_importance": importance,
            "classification_report": report,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "training_time_ms": training_time_ms,
        }

    def predict(self, features: list[float]) -> dict[str, Any]:
        """Predict a single sample."""
        if self.model is None:
            raise RuntimeError("No model has been trained yet")

        X = np.array(features).reshape(1, -1)
        if self.scaler is not None:
            X = self.scaler.transform(X)

        prediction = int(self.model.predict(X)[0])
        result: dict[str, Any] = {"class": prediction}

        # Probabilities if available
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            result["probabilities"] = {
                self.target_names[i] if i < len(self.target_names) else str(i):
                round(float(p), 4)
                for i, p in enumerate(proba)
            }

        return result

    # ----- private helpers --------------------------------------------------

    def _compute_feature_importance(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Compute feature importance using tree-based or permutation method."""
        importances: np.ndarray | None = None

        # Try tree-based importance first
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        else:
            # Fall back to permutation importance
            try:
                perm_result = permutation_importance(
                    self.model, X_test, y_test,
                    n_repeats=10, random_state=42, n_jobs=-1,
                )
                importances = perm_result.importances_mean
            except Exception as exc:
                logger.warning("Could not compute permutation importance: %s", exc)
                return []

        if importances is None:
            return []

        # Build list sorted by importance (descending)
        feature_imp = []
        for i, imp in enumerate(importances):
            name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            feature_imp.append({
                "feature": name,
                "importance": round(float(imp), 6),
            })

        feature_imp.sort(key=lambda x: x["importance"], reverse=True)
        return feature_imp

    @staticmethod
    def _sanitise_report(report: dict) -> dict:
        """Ensure all values in the classification report are JSON-serialisable."""
        sanitised = {}
        for key, value in report.items():
            if isinstance(value, dict):
                sanitised[key] = {
                    k: round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v)
                    for k, v in value.items()
                }
            elif isinstance(value, (float, np.floating)):
                sanitised[key] = round(float(value), 4)
            elif isinstance(value, (int, np.integer)):
                sanitised[key] = int(value)
            else:
                sanitised[key] = value
        return sanitised
