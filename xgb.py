import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

xgb_model = xgb.XGBClassifier()

param_space = {
    'learning_rate': (0.01, 1.0, 'log-uniform'),
    'max_depth': (1, 50),
    'min_child_weight': (0, 10),
    'n_estimators': (50, 1000),
    'subsample': (0.1, 1.0, 'uniform'),
    'gamma': (0, 10),
    'colsample_bytree': (0.1, 1.0, 'uniform'),
    'reg_alpha': (0, 10),
    'reg_lambda': (0, 10),
}

opt = BayesSearchCV(
    xgb_model,
    param_space,
    n_iter=50,#964,9824
    cv=5,
    n_jobs=-1,
    verbose=1,
)

opt.fit(X_train, y_train)

import json

# Save to file
with open("best_params.json", "w") as f:
    json.dump(opt.best_params_, f)


# Load from file
with open("best_params.json", "r") as f:
    loaded_params = json.load(f)


# Initialize model with loaded parameters
loaded_model = xgb.XGBClassifier(**loaded_params)

# Fit the model
loaded_model.fit(X_train, y_train)
##################


from sklearn.metrics import accuracy_score

# Make predictions on the test set
y_pred = loaded_model.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")
