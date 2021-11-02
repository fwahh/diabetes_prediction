#!/usr/bin/env python
# coding: utf-8


import os
import pickle

import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Create models directory for storing models if it does not exist
if not os.path.exists('models'):
    os.makedirs('models')
    print("models directory created to store models\n")

data_path = os.path.join('data', 'diabetes.csv')
df = pd.read_csv(data_path)
df.columns = [var.lower().replace(' ', '_') for var in df.columns]

# We'll only be splitting into train and test set instead of train, validation and test set as cross validation will be used on the train set itself.
df_train, df_test = train_test_split(df, test_size=0.2, random_state=86431)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = (df_train['class'] == 'Positive').astype(int)
y_test = (df_test['class'] == 'Positive').astype(int)
del df_train['class']
del df_test['class']

# dict vectorizing values
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(df_train.to_dict(orient='records'))
X_test = dv.transform(df_test.to_dict(orient='records'))



# Hyperparameters used below are already tuned. Refer to notebook for details on hyperparameter tuning.

# Logistic Regression
lr = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
lr.fit(X_train, y_train)
y_pred_proba = lr.predict_proba(X_test)[:, 1]
y_pred = lr.predict(X_test)

print ("Logistic Regression")
print(f'Test ROC-AUC score: {roc_auc_score(y_test, y_pred_proba)}')
print(f'Test Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Test Classification Report: \n {classification_report(y_test, y_pred)}')

# Save the model
output_lr = os.path.join('models', 'logistic_regression.bin')

with open(output_lr, 'wb') as f_out:
    pickle.dump(lr, f_out)
print(f'Logistic Regression model saved to: {output_lr}\n\n')


# Random Forest Classifier
rfc_tuned_params = {'n_estimators': 140,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_depth': 70,
                    'bootstrap': True}
rfc = RandomForestClassifier(**rfc_tuned_params, random_state=86431)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
y_pred_proba = rfc.predict_proba(X_test)[:, 1]

print ("Random Forest Classifier")
print(f'Test ROC-AUC score: {roc_auc_score(y_test, y_pred_proba)}')
print(f'Test Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Test Classification Report: \n {classification_report(y_test, y_pred)}')

# Save the model
output_rfc = os.path.join('models', 'random_forest_classifier.bin')

with open(output_rfc, 'wb') as f_out:
    pickle.dump(rfc, f_out)
print(f'Random Forest Classifier model saved to: {output_rfc}\n\n')

# XGBoost Classifier
features = dv.get_feature_names_out()
dtrain = xgb.DMatrix(data=X_train, label=y_train, feature_names=features)
xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'nthread': 8,
    
    'seed': 86431,
    'verbosity': 1,
    'eval_metric': 'auc'
}
xgbC = xgb.train(xgb_params, dtrain, num_boost_round=100)


dtest = xgb.DMatrix(data=X_test, feature_names=features)
y_pred_proba = xgbC.predict(dtest)
y_pred = np.where(y_pred_proba >= 0.5, 1, 0)
roc_auc_score(y_test, y_pred_proba)

print ("XGBoost Classifier")
print(f'Test ROC-AUC score: {roc_auc_score(y_test, y_pred_proba)}')
print(f'Test Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Test Classification Report: \n {classification_report(y_test, y_pred)}')


# Save the model
output_xgbc = os.path.join('models', 'xgboost_classifier.bin')

with open(output_xgbc, 'wb') as f_out:
    pickle.dump(xgbC, f_out)
print(f'XGBoost Classifier model saved to: {output_xgbc}\n\n')

# Save dict vectorizer
output_dv = os.path.join('models', 'dv.bin')

with open(output_dv, 'wb') as f_out:
    pickle.dump(dv, f_out)
print(f'DictVectorizer saved to: {output_dv}')