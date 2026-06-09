from flask import Blueprint, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

prediction_bp = Blueprint('prediction', __name__)

# Load the trained model and preprocessing objects
model_path = os.path.join(os.path.dirname(__file__), '..', 'titanic_model.pkl')
imputer_age_path = os.path.join(os.path.dirname(__file__), '..', 'imputer_age.pkl')
imputer_embarked_path = os.path.join(os.path.dirname(__file__), '..', 'imputer_embarked.pkl')
label_encoder_sex_path = os.path.join(os.path.dirname(__file__), '..', 'label_encoder_sex.pkl')
one_hot_encoder_embarked_path = os.path.join(os.path.dirname(__file__), '..', 'one_hot_encoder_embarked.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), '..', 'scaler.pkl')

model = joblib.load(model_path)
imputer_age = joblib.load(imputer_age_path)
imputer_embarked = joblib.load(imputer_embarked_path)
label_encoder_sex = joblib.load(label_encoder_sex_path)
one_hot_encoder_embarked = joblib.load(one_hot_encoder_embarked_path)
scaler = joblib.load(scaler_path)

@prediction_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract features from the request
        pclass = data.get('pclass')
        sex = data.get('sex')
        age = data.get('age')
        sibsp = data.get('sibsp')
        parch = data.get('parch')
        fare = data.get('fare')
        embarked = data.get('embarked')
        alone = data.get('alone', True)
        
        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'pclass': [pclass],
            'sex': [sex],
            'age': [age],
            'sibsp': [sibsp],
            'parch': [parch],
            'fare': [fare],
            'embarked': [embarked],
            'alone': [alone]
        })
        
        # Handle missing values
        if pd.isna(input_data['age'].iloc[0]):
            input_data['age'] = imputer_age.transform(input_data[['age']])
        
        if pd.isna(input_data['embarked'].iloc[0]):
            input_data['embarked'] = imputer_embarked.transform(input_data[['embarked']]).ravel()
        
        # Encode categorical features
        input_data['sex'] = label_encoder_sex.transform(input_data['sex'])
        
        # One-hot encode 'embarked'
        embarked_encoded = one_hot_encoder_embarked.transform(input_data[['embarked']])
        embarked_df = pd.DataFrame(embarked_encoded, columns=one_hot_encoder_embarked.get_feature_names_out(['embarked']))
        input_data = pd.concat([input_data, embarked_df], axis=1)
        input_data.drop('embarked', axis=1, inplace=True)
        
        # Scale features
        input_data[['fare', 'age']] = scaler.transform(input_data[['fare', 'age']])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'survival_probability': float(prediction_proba[1]),
            'death_probability': float(prediction_proba[0])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

