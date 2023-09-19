from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ramda import pipe
import json
import matplotlib
matplotlib.use('TkAgg')

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


# Make predictions on the test set
y_pred = loaded_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
precision = precision_score(y_test, y_pred, average='weighted')
print("precision:", precision)
recall = recall_score(y_test, y_pred, average='weighted')
print("Recall:", recall)
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score:", f1)



# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
#nn_model(X_train, X_test, y_train, y_test)

# Calculate the confusion matrix for holdout
cm = confusion_matrix(y_test, y_pred)
yield1 = int((cm[1][1] + cm[2][2] - cm[0][1] - cm[0][2] - cm[1][2] - cm[2][1])) / int(
    (cm[1][1] + cm[2][2] + cm[0][1] + cm[0][2] + cm[1][2] + cm[2][1]))  #
total_bets = cm[1][1] + cm[2][2] + cm[0][1] + cm[0][2] + cm[1][2] + cm[2][1]  #
income = cm[1][1] + cm[2][2] - cm[0][1] - cm[0][2] - cm[1][2] - cm[2][1]  #
# Results on holdout
print('Test income: ', income)
print('Test total placed bet: ', total_bets)
print('Test yield: ', yield1)

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

# to do:
# add result to dict
# add result to set