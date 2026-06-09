# 🚢 Titanic Survival Prediction

> End-to-end machine learning pipeline predicting passenger survival — from raw data ingestion through preprocessing, model training, and a live web application demo.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=flat&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## Overview

This project trains a **Random Forest classifier** on the classic [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic) to predict whether a passenger survived the disaster. It demonstrates a complete, production-style workflow:

```
Raw CSV → Feature Engineering → Preprocessing → Model Training → REST API
```

| Metric | Value |
|---|---|
| Dataset size | 891 passengers |
| Engineered features | 7 |
| Test set accuracy | ~82% |
| Model type | Random Forest Classifier |

**Key survival factors:** Sex, passenger class (Pclass), age, fare, and port of embarkation are the strongest predictors. The model captures non-linear interactions between these through decision-tree ensembles.

---

## Project Structure

```
Titanic-Survival-Prediction/
│
├── titanic.csv                    # Original 891-row Kaggle dataset
├── X_train.csv                    # Training features (80% split)
├── X_test.csv                     # Test features (20% split)
├── y_train.csv                    # Training labels
├── y_test.csv                     # Test labels
│
├── imputer_age.pkl                # Fitted median imputer for Age
├── imputer_embarked.pkl           # Fitted mode imputer for Embarked
├── label_encoder_sex.pkl          # Binary encoder for Sex
├── one_hot_encoder_embarked.pkl   # One-hot encoder for Embarked
├── scaler.pkl                     # Standard scaler for Age & Fare
├── titanic_model.pkl              # Serialised trained model
│
├── titanic_preprocessing.py       # Data preprocessing pipeline
├── titanic_model_training.py      # Model training & evaluation
├── titanic_model.py               # Model definition and utilities
│
└── titanic_app/
    ├── requirements.txt           # Web application dependencies
    └── src/
        ├── main.py                # FastAPI application entry point
        └── database/
            └── app.db             # SQLite store for prediction history
```

---

## ML Pipeline

### 1. Data Loading
`titanic.csv` is loaded and split into train/test sets (80/20 stratified). Splits are persisted as CSVs for reproducibility.

### 2. Preprocessing (`titanic_preprocessing.py`)

| Step | Feature | Method |
|---|---|---|
| Imputation | `Age` | Median imputation |
| Imputation | `Embarked` | Mode imputation |
| Encoding | `Sex` | Label encoding (binary) |
| Encoding | `Embarked` | One-hot encoding |
| Scaling | `Age`, `Fare` | Standard scaling |

All transformers are fitted on training data only and serialised to `.pkl` files to prevent data leakage.

### 3. Model Training (`titanic_model_training.py`)
A `RandomForestClassifier` is trained with cross-validated hyperparameter selection. The final estimator is serialised to `titanic_model.pkl`. Feature importances rank **Pclass**, **Sex**, and **Age** as the top predictors.

### 4. Serving (`titanic_app/`)
The FastAPI app loads the serialised model and preprocessors at startup. Incoming requests are preprocessed through the same pipeline and scored. Results are persisted to `app.db` for auditability.

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/zakir-maswani/Titanic-Survival-Prediction.git
cd Titanic-Survival-Prediction
```

### 2. Create a virtual environment (recommended)

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
# Core ML dependencies
pip install pandas scikit-learn numpy joblib

# Web application
cd titanic_app
pip install -r requirements.txt
cd ..
```

#### Core dependencies

| Package | Purpose |
|---|---|
| `pandas` | DataFrame I/O and feature engineering |
| `scikit-learn` | Preprocessing, RandomForest, model serialisation |
| `numpy` | Numerical arrays and transforms |
| `joblib` | Efficient `.pkl` serialisation / deserialisation |
| `fastapi` | Async REST framework for the web app |
| `uvicorn` | ASGI server |
| `sqlalchemy` | ORM layer on top of SQLite |

---

## Usage

### Step 1 — Run preprocessing

Generates split CSVs and fitted transformer `.pkl` files. Only needed once, or when source data changes.

```bash
python titanic_preprocessing.py
```

### Step 2 — Train the model

Fits the classifier on the training split, evaluates on the test split, and writes `titanic_model.pkl`.

```bash
python titanic_model_training.py
```

### Step 3 — Launch the web application

```bash
cd titanic_app/src
python main.py
```

The server starts at `http://127.0.0.1:8000`.

| Endpoint | URL |
|---|---|
| Application | http://127.0.0.1:8000 |
| Swagger UI | http://127.0.0.1:8000/docs |
| ReDoc | http://127.0.0.1:8000/redoc |

### Example API request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pclass": 1,
    "Sex": "female",
    "Age": 29,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 211.3375,
    "Embarked": "S"
  }'
```

---

## FAQ

<details>
<summary><strong>Why are there separate <code>.pkl</code> files for each preprocessor?</strong></summary>

Keeping each transformer serialised separately means you can hot-swap one step (e.g. change the age imputation strategy) without retraining the model or re-fitting every other transformer. It also makes unit-testing individual pipeline stages straightforward.

</details>

<details>
<summary><strong>What happens if I send a passenger with missing fields to the API?</strong></summary>

The FastAPI layer validates the request schema with Pydantic before inference. Missing optional fields (`Age`, `Embarked`) are handled by the same imputers fitted during training. Required fields (`Pclass`, `Sex`, `Fare`) will trigger a `422 Unprocessable Entity` error if absent.

</details>

<details>
<summary><strong>Can I swap in a different model (e.g. XGBoost) without changing the app?</strong></summary>

Yes — `titanic_model.pkl` is loaded generically via `joblib.load()`. As long as your replacement model exposes a `.predict()` and `.predict_proba()` interface (any scikit-learn-compatible estimator does), the web app requires no changes.

</details>

<details>
<summary><strong>How do I retrain with new data?</strong></summary>

Replace `titanic.csv`, then re-run `titanic_preprocessing.py` followed by `titanic_model_training.py`. This re-fits all transformers and the classifier from scratch, writing fresh `.pkl` artifacts. Restart the web app to pick up the new model.

</details>

<details>
<summary><strong>Where are past predictions stored?</strong></summary>

Each prediction is written to `titanic_app/src/database/app.db` via SQLAlchemy. Inspect it with any SQLite browser or query directly:

```bash
sqlite3 app.db "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 10;"
```

</details>

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
