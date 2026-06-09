# Titanic Survival Prediction

This repository contains a machine learning project to predict Titanic survival. It includes data preprocessing, model training, and a simple web application to demonstrate the model.

## Project Structure

- `titanic.csv`: The original dataset.
- `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`: Split datasets for training and testing.
- `imputer_age.pkl`, `imputer_embarked.pkl`, `label_encoder_sex.pkl`, `one_hot_encoder_embarked.pkl`, `scaler.pkl`: Preprocessing artifacts.
- `titanic_model.pkl`: The trained machine learning model.
- `titanic_model.py`: Python script containing the model definition or utility functions.
- `titanic_model_training.py`: Script for training the Titanic survival prediction model.
- `titanic_preprocessing.py`: Script for data preprocessing.
- `titanic_app/`: Directory containing a web application to interact with the model.
  - `titanic_app/requirements.txt`: Python dependencies for the web application.
  - `titanic_app/src/main.py`: Main entry point for the web application.
  - `titanic_app/src/database/app.db`: SQLite database for the application.

## Setup and Installation

To set up the project, clone the repository and install the necessary dependencies.

```bash
git clone https://github.com/zakir-maswani/Titanic-Survival-Prediction.git
cd Titanic-Survival-Prediction

# Install Python dependencies for the model training and preprocessing
pip install -r requirements.txt # Assuming a requirements.txt for the main project, if not, install individually

# Install Python dependencies for the web application
cd titanic_app
pip install -r requirements.txt
cd ..
```

## Usage

### Model Training and Preprocessing

To train the model and generate preprocessing artifacts, run:

```bash
python titanic_preprocessing.py
python titanic_model_training.py
```

### Running the Web Application

To run the web application, navigate to the `titanic_app` directory and execute the main script:

```bash
cd titanic_app/src
python main.py
```

Then, open your web browser and go to the address provided by the application (usually `http://127.0.0.1:8000`).
