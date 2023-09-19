from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ramda import pipe
from functools import partial
import numpy as np
import wandb
from wandb.keras import WandbCallback
from wandb.xgboost import WandbCallback
#wandb login
from config import min_time, max_time, minbetodd, maxbetodd, insufficient, threshold
from common import removeDotFromColumnNames, dropMinutes, sortByDate, dropNotDraw, oddsFilter, addSumStats, dropInsufficient, dif_threshold
from XGBwithBayesSearch import xgb_model

df = pd.read_csv('/home/kk/PycharmProjects/oddmaker/data/exp23_withLeagues_LIMITED_minutes_4-35_odd_1.1-5_insfufficient_10_dif_1_onlyDraws.csv')
df.dropna(inplace=True)
print(df.shape)


pipeline = pipe(
    removeDotFromColumnNames,
    dropMinutes,
    sortByDate,
    dropNotDraw,
    oddsFilter,
    addSumStats,
    dropInsufficient,
    dif_threshold
)

# Apply pipeline
df = pipeline(df)
df.drop(['datetimestamp'], axis=1, inplace=True)

print(df.shape)


import xgboost as xgb
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(df.drop('zzz_play', axis=1), df['zzz_play'], test_size=0.2,random_state=42)
#
# xgb_model = xgb.XGBClassifier()
#
# param_space = {
#     'learning_rate': (0.01, 1.0, 'log-uniform'),
#     'max_depth': (1, 50),
#     'min_child_weight': (0, 10),
#     'n_estimators': (50, 1000),
#     'subsample': (0.1, 1.0, 'uniform'),
#     'gamma': (0, 10),
#     'colsample_bytree': (0.1, 1.0, 'uniform'),
#     'reg_alpha': (0, 10),
#     'reg_lambda': (0, 10),
# }
#
# opt = BayesSearchCV(
#     xgb_model,
#     param_space,
#     n_iter=2,#964,9824
#     cv=5,
#     n_jobs=-1,
#     verbose=1,
# )
#
# opt.fit(X_train, y_train)
#
# import json
#
# # Save to file
# with open("best_params.json", "w") as f:
#     json.dump(opt.best_params_, f)
#
#
# Load from file
xgb_model(X_train, X_test, y_train, y_test)

with open("best_params.json", "r") as f:
    loaded_params = json.load(f)


# Initialize model with loaded parameters
loaded_model = xgb.XGBClassifier(**loaded_params)

# Fit the model
loaded_model.fit(X_train, y_train)



from sklearn.metrics import accuracy_score

# Make predictions on the test set
y_pred = loaded_model.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")
