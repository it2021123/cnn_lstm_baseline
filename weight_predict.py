#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 16:10:39 2024

@author: poulimenos
"""
# Εισαγωγή βιβλιοθηκών
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import joblib

#Data links:
"""
    https://www.kaggle.com/datasets/israrullahkhan/ageheightweightgenderlikeness-dataset
    https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset
"""
# Φόρτωση των δεδομένων
df = pd.read_csv('weight_predict.csv')

# Διαχείριση κατηγορικών δεδομένων (Sex με encoding)
df['Sex'] = df['Sex'].map({'M': 0, 'F': 1})

# Ορισμός χαρακτηριστικών και στόχου
X = df[['Sex', 'Age', 'Height']]  # Χαρακτηριστικά
y = df['Weight']  # Στόχος (εξαρτημένη μεταβλητή)
X = X.replace(',', '.', regex=True).astype(float)
y = y.replace(',', '.', regex=True).astype(float)

# Διαχωρισμός των δεδομένων σε σύνολα εκπαίδευσης και δοκιμής
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Ορισμός υπερπαραμέτρων προς βελτιστοποίηση για κάθε μοντέλο
param_grids = {
    'Linear Regression': {},  # Δεν έχει υπερπαραμέτρους για grid search
    'Decision Tree': {
        'max_depth': [3, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4,8]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'SVR': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf']
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10]
    },
    'k-NN': {
        'n_neighbors': [3, 5, 10,15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
}

# Αρχικοποίηση των μοντέλων
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'SVR': SVR(),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'k-NN': KNeighborsRegressor()
}

# Αποθήκευση των αποτελεσμάτων
results = {}

# Εκπαίδευση και Grid Search για κάθε μοντέλο
for name, model in models.items():
    print(f"Training and tuning: {name}")
    
    # Δημιουργία του GridSearchCV
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        scoring='neg_mean_absolute_error',  # Εναλλακτικά, μπορείς να βάλεις R² ή MSE
        cv=5,  # Cross-validation με 5-folds
        verbose=1,
        n_jobs=-1  # Χρήση όλων των επεξεργαστών
    )
    
    # Εκπαίδευση του GridSearchCV
    grid.fit(X_train, y_train)
    
    # Καλύτερο μοντέλο και υπερπαράμετροι
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    
    # Προβλέψεις στο test set
    y_pred = best_model.predict(X_test)
    
    # Υπολογισμός μετρικών
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Αποθήκευση των αποτελεσμάτων
    results[name] = {
        'Best Params': best_params,
        'MAE': mae,
        'MSE': mse,
        'R²': r2
    }

# Εκτύπωση των αποτελεσμάτων
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    print(f"  Best Params: {metrics['Best Params']}")
    print(f"  Mean Absolute Error (MAE): {metrics['MAE']:.2f}")
    print(f"  Mean Squared Error (MSE): {metrics['MSE']:.2f}")
    print(f"  R-squared Score (R²): {metrics['R²']:.2f}")
    print("-" * 30)

import matplotlib.pyplot as plt

models = list(results.keys())
mae_scores = [metrics['MAE'] for metrics in results.values()]

plt.bar(models, mae_scores, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Comparison of Model Performance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


best_model = SVR( kernel='linear', gamma='scale',  C=10) # Λήψη του αντικειμένου από το λεξικό

# Επανεκπαίδευση του καλύτερου μοντέλου
best_model.fit(X_train, y_train)

# Αποθήκευση
model_filename = f"best_model_SVR.joblib"
joblib.dump(best_model, model_filename)
print(f"Το καλύτερο μοντέλο (SVR) αποθηκεύτηκε ως '{model_filename}'.")


# Επαναφόρτωση για έλεγχο
loaded_model = joblib.load(model_filename)
print("Το αποθηκευμένο μοντέλο φορτώθηκε με επιτυχία!")