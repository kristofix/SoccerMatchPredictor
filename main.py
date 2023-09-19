from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ramda import pipe
import json
import matplotlib
matplotlib.use('TkAgg')  # Or use another backend that you have, like 'Qt5Agg'

import wandb
from wandb.keras import WandbCallback
from wandb.xgboost import WandbCallback
#wandb login
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
df = df.iloc[:len(df)//8]               #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print(df.shape)


import xgboost as xgb
X_train, X_test, y_train, y_test = train_test_split(df.drop('zzz_play', axis=1), df['zzz_play'], test_size=0.2,random_state=42)

xgb_model(X_train, X_test, y_train, y_test)

#Load params from xgb training
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

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
#nn_model(X_train, X_test, y_train, y_test)

# Calculate the confusion matrix for holdout
cm1 = confusion_matrix(y_test, y_pred)
yieldd_holdout = int((cm1[1][1] + cm1[2][2] - cm1[0][1] - cm1[0][2] - cm1[1][2] - cm1[2][1])) / int(
    (cm1[1][1] + cm1[2][2] + cm1[0][1] + cm1[0][2] + cm1[1][2] + cm1[2][1]))  #
total_bets_holdout = cm1[1][1] + cm1[2][2] + cm1[0][1] + cm1[0][2] + cm1[1][2] + cm1[2][1]  #
income_holdout = cm1[1][1] + cm1[2][2] - cm1[0][1] - cm1[0][2] - cm1[1][2] - cm1[2][1]  #
# Results on holdout
print('Test income: ', income_holdout)
print('Test total placed bet: ', total_bets_holdout)
print('Test yield: ', yieldd_holdout)

# new wandb funct
from wandb import Image

#fig, ax = plt.subplots(figsize=(10, 6))
#lgb.plot_importance(loaded_model, ax=ax)
#wandb.log({'feature_importances': Image(fig)})
#plt.show()

# new wandb funct
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
plt.xlabel('Predicted')
plt.ylabel('True')
#wandb.log({'confusion_matrix': Image(fig)})
plt.show()

