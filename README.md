# MLExplain

> Interactive Machine Learning Model Explainer -- train, evaluate, and
> understand scikit-learn models through feature importance, confusion
> matrices, and side-by-side comparison.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikitlearn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Tests](https://img.shields.io/badge/Tests-14%20passing-22c55e)
![Coverage](https://img.shields.io/badge/Coverage-92%25-22c55e)

---

## Overview

MLExplain is an educational and practical tool for exploring how
different machine-learning algorithms perform on classic datasets.
Train models with a click, inspect feature importance bar charts,
study confusion matrices, and compare algorithms side by side -- all
from a clean, browser-based interface.

### Key Features

| Feature | Description |
|---------|-------------|
| Built-in Datasets | Iris, Wine, Breast Cancer, Digits from scikit-learn |
| Five Algorithms | Decision Tree, Random Forest, SVM, KNN, Logistic Regression |
| Hyperparameters | Configurable max depth, n_estimators, C, k, solver, etc. |
| Train / Test Split | Adjustable ratio (default 80/20) with reproducible seed |
| Metrics Dashboard | Accuracy, Precision, Recall, F1-Score per experiment |
| Feature Importance | Tree-based & permutation importance bar charts |
| Confusion Matrix | Interactive heatmap with per-class counts |
| Model Comparison | Train multiple models, compare metrics side by side |
| Experiment History | Every run saved to SQLite with full metadata |
| Prediction API | POST features, receive prediction + confidence scores |
| REST API | Full CRUD for experiments, datasets, and predictions |

---

## Architecture

```
mlexplain/
+-- app.py                          # Flask entry point & factory
+-- config.py                       # App configuration
+-- requirements.txt                # Pinned dependencies
+-- models/
|   +-- __init__.py
|   +-- database.py                 # SQLAlchemy setup
|   +-- schemas.py                  # Experiment, Dataset, ModelResult
+-- routes/
|   +-- __init__.py
|   +-- api.py                      # REST API endpoints
|   +-- views.py                    # HTML page routes
+-- services/
|   +-- __init__.py
|   +-- ml_engine.py                # Training, prediction, explanation
|   +-- datasets.py                 # Built-in dataset loading
+-- templates/
|   +-- base.html                   # Layout with navigation
|   +-- index.html                  # Dashboard
|   +-- train.html                  # Train a model
|   +-- explain.html                # Model explanations
|   +-- compare.html                # Compare models
|   +-- about.html                  # About page
+-- static/
|   +-- css/style.css               # Scientific theme
|   +-- js/main.js                  # Chart.js visualisations
+-- tests/
|   +-- conftest.py
|   +-- test_api.py
|   +-- test_models.py
|   +-- test_services.py
+-- seed_data/data.json
```

---

## Quick Start

### Prerequisites

- Python 3.11 or later
- pip

### Installation

```bash
git clone https://github.com/your-org/mlexplain.git
cd mlexplain
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The application will be available at **http://localhost:8006**.

### Docker

```bash
docker compose up --build
```

---

## API Reference

### Datasets

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/datasets` | List available datasets |
| GET | `/api/datasets/<name>` | Get dataset info & preview |

### Training

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/train` | Train a model |
| GET | `/api/experiments` | List all experiments |
| GET | `/api/experiments/<id>` | Get experiment details |
| DELETE | `/api/experiments/<id>` | Delete an experiment |

### Explanation

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/experiments/<id>/importance` | Feature importance |
| GET | `/api/experiments/<id>/confusion` | Confusion matrix |
| GET | `/api/experiments/<id>/metrics` | Detailed metrics |

### Prediction

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict/<id>` | Predict with a trained model |

### Comparison

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/compare` | Compare multiple models |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8006` | Server port |
| `FLASK_DEBUG` | `0` | Enable debug mode |
| `DATABASE_URL` | `sqlite:///instance/mlexplain.db` | Database URI |
| `SECRET_KEY` | *(dev key)* | Flask secret key |

---

## Running Tests

```bash
pytest -v --cov=. --cov-report=term-missing
```

---

## Supported Algorithms

### Decision Tree
Interpretable tree-based classifier. Configurable `max_depth` and
`min_samples_split`. Provides direct feature importance via Gini
impurity.

### Random Forest
Ensemble of decision trees with bagging. Configurable `n_estimators`,
`max_depth`. Feature importance averaged across all trees.

### Support Vector Machine (SVM)
Kernel-based classifier (RBF, linear, poly). Configurable
regularisation parameter `C` and `kernel` type.

### K-Nearest Neighbours (KNN)
Instance-based learning. Configurable `n_neighbors` and distance
`metric` (euclidean, manhattan).

### Logistic Regression
Linear model for classification. Configurable `C`, `solver`
(lbfgs, liblinear, saga), and `max_iter`.

---

## Feature Importance Methods

- **Tree-based importance** -- uses `feature_importances_` attribute
  from Decision Tree and Random Forest (Gini impurity reduction).
- **Permutation importance** -- available for all models. Measures
  accuracy drop when each feature is randomly shuffled.

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, Flask 3.0 |
| ML Engine | scikit-learn 1.3, NumPy 1.26 |
| Database | SQLite via SQLAlchemy 2.0 |
| Frontend | Jinja2 templates, Chart.js 4 |
| Testing | pytest 7.4 with coverage |
| Deployment | Docker, Gunicorn |

---

## License

This project is licensed under the **MIT License** -- see [LICENSE](LICENSE).

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Acknowledgements

- [scikit-learn](https://scikit-learn.org/) for ML algorithms and datasets
- [Chart.js](https://www.chartjs.org/) for interactive visualisations
- [Flask](https://flask.palletsprojects.com/) for the web framework
