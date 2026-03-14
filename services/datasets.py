"""Built-in dataset loading from scikit-learn.

Provides iris, wine, breast_cancer, and digits datasets with
metadata for use in the training pipeline.
"""

import json
import logging
import os
from typing import Any

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits

logger = logging.getLogger(__name__)

# Registry of available datasets
_DATASET_LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
}

_DATASET_DESCRIPTIONS = {
    "iris": (
        "Classic Fisher's Iris dataset with 150 samples of 3 species "
        "(setosa, versicolor, virginica) and 4 features (sepal/petal "
        "length and width)."
    ),
    "wine": (
        "Wine recognition dataset with 178 samples of 3 cultivar classes "
        "and 13 chemical analysis features."
    ),
    "breast_cancer": (
        "Wisconsin Breast Cancer dataset with 569 samples (malignant / "
        "benign) and 30 features computed from cell nuclei images."
    ),
    "digits": (
        "Optical handwritten digits dataset with 1,797 samples of digits "
        "0-9 and 64 pixel intensity features (8x8 images)."
    ),
}


def get_available_datasets() -> list[dict[str, Any]]:
    """Return metadata for all available built-in datasets."""
    datasets = []
    for name, loader_fn in _DATASET_LOADERS.items():
        bunch = loader_fn()
        datasets.append({
            "name": name,
            "description": _DATASET_DESCRIPTIONS.get(name, ""),
            "num_samples": int(bunch.data.shape[0]),
            "num_features": int(bunch.data.shape[1]),
            "num_classes": int(len(bunch.target_names)),
            "feature_names": [str(f) for f in bunch.feature_names]
                if hasattr(bunch, "feature_names") else
                [f"feature_{i}" for i in range(bunch.data.shape[1])],
            "target_names": [str(t) for t in bunch.target_names],
        })
    return datasets


def load_dataset_by_name(name: str) -> dict[str, Any]:
    """Load a dataset by name and return data, target, and metadata.

    Parameters
    ----------
    name : str
        One of 'iris', 'wine', 'breast_cancer', 'digits'.

    Returns
    -------
    dict with keys: data (ndarray), target (ndarray), feature_names,
    target_names, num_samples, num_features, num_classes.

    Raises
    ------
    ValueError
        If *name* is not a recognised dataset.
    """
    loader_fn = _DATASET_LOADERS.get(name)
    if loader_fn is None:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available: {list(_DATASET_LOADERS.keys())}"
        )

    bunch = loader_fn()
    feature_names = (
        [str(f) for f in bunch.feature_names]
        if hasattr(bunch, "feature_names")
        else [f"feature_{i}" for i in range(bunch.data.shape[1])]
    )
    target_names = [str(t) for t in bunch.target_names]

    return {
        "data": bunch.data,
        "target": bunch.target,
        "feature_names": feature_names,
        "target_names": target_names,
        "num_samples": int(bunch.data.shape[0]),
        "num_features": int(bunch.data.shape[1]),
        "num_classes": len(target_names),
    }


def seed_database(db) -> int:
    """Seed the database with dataset records if empty."""
    from models.schemas import Dataset

    if Dataset.query.count() > 0:
        return 0

    count = 0
    for name, loader_fn in _DATASET_LOADERS.items():
        bunch = loader_fn()
        feature_names = (
            [str(f) for f in bunch.feature_names]
            if hasattr(bunch, "feature_names")
            else [f"feature_{i}" for i in range(bunch.data.shape[1])]
        )
        target_names = [str(t) for t in bunch.target_names]

        record = Dataset(
            name=name,
            num_samples=int(bunch.data.shape[0]),
            num_features=int(bunch.data.shape[1]),
            num_classes=len(target_names),
            feature_names=json.dumps(feature_names),
            target_names=json.dumps(target_names),
        )
        db.session.add(record)
        count += 1

    db.session.commit()
    logger.info("Seeded %d datasets into the database", count)
    return count


def load_seed_data() -> dict:
    """Load supplementary seed data from seed_data/data.json."""
    seed_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "seed_data", "data.json",
    )
    if not os.path.exists(seed_path):
        return {}
    with open(seed_path, "r", encoding="utf-8") as fh:
        return json.load(fh)
