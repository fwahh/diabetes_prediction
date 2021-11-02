import pickle
import os
import pandas as pd
import json
import xgboost as xgb

from flask import Flask, request, jsonify

# retrieve the 3 models:

dirname = 'models'
lr_path = os.path.join(dirname, 'logistic_regression.bin')
with open(lr_path, 'rb') as f_in:
    lr_model = pickle.load(f_in)

rfc_path = os.path.join(dirname, 'random_forest_classifier.bin')
with open(rfc_path, 'rb') as f_in:
    rfc_model = pickle.load(f_in)

xgbc_path = os.path.join(dirname, 'xgboost_classifier.bin')
with open(xgbc_path, 'rb') as f_in:
    xgbc_model = pickle.load(f_in)

dv_path = os.path.join(dirname, 'dv.bin')
with open(dv_path, 'rb') as f_in:
    dv = pickle.load(f_in)

test = {'age': 47,
        'gender': "Male",
        'polyuria': "No",
        'polydipsia': "No",
        'sudden weight loss': "No",
        'weakness': "No",
        'polyphagia': "No",
        'genital thrush': "No",
        'visual blurring': "No",
        'itching': "Yes", 
        'irritability': "No",
        'delayed healing': "No",
        'partial paresis': "No",
        'muscle stiffness': "No",
        'alopecia': "Yes",
        'obesity': "No"
        }

X = dv.transform([test])
y_prob_lr = lr_model.predict_proba(X)[0, 1]
y_prob_rfc = rfc_model.predict_proba(X)[0, 1]

features = dv.get_feature_names_out()
dtest = xgb.DMatrix(data=X, feature_names=features)
y_prob_xgbc = xgbc_model.predict(dtest)

if len(y_prob_xgbc) == 1:
    y_prob_xgbc = y_prob_xgbc[0]

print(f'Probability of having diabetes: \n' 
      f'Logistic Regression Model:  {y_prob_lr} \n'
      f'Random Forest Classifier:   {y_prob_rfc} \n' 
      f'XGBoost Classifier:         {y_prob_xgbc} \n')

# app = Flask('churn')

# @app.route('/predict', methods=['POST'])
# def predict():
#     customer = request.get_json()
#     X = dv.transform([customer])
#     results = {'churn_proba': model.predict_proba(X)[0, 1]}
#     return jsonify(results)

# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=9696)