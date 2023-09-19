from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ramda import pipe
import numpy as np
import json
import matplotlib
import xgboost as xgb

matplotlib.use('TkAgg')

import wandb
from wandb.keras import WandbCallback
from wandb.xgboost import WandbCallback
#wandb login
from common import removeDotFromColumnNames, dropMinutes, sortByDate, dropNotDraw, oddsFilter, addSumStats, dropInsufficient, dif_threshold
from XGBwithBayesSearch import xgb_model
from neuralNetwork import nn_model

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
df = df.iloc[:len(df)//100]      #uncomment for quick test run
print(df.shape)
df.to_csv('your_file_name.csv', index=False)
df.to_csv('your_file_name1.csv', index=False)

#Normalize data because data do not follow GaussianDistribution - checked in other file
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
print(df.head(5))
df.to_csv('your_file_name2.csv', index=False)

X_train, X_test, y_train, y_test = train_test_split(df.drop('zzz_play', axis=1), df['zzz_play'], test_size=0.2,random_state=42)


#SELECT YOUR MODEL
#XGB - 1
#neural network - 2
#se selector to choose model
model_selector = 2
if model_selector == 1:
    xgb_model(X_train, X_test, y_train, y_test)
    #Load params from xgb training
    with open("best_params.json", "r") as f:
       loaded_params = json.load(f)
    #Initialize model with loaded parameters
    loaded_model = xgb.XGBClassifier(**loaded_params)
    loaded_model.fit(X_train, y_train)
    y_pred = loaded_model.predict(X_test)
elif model_selector == 2:
    nn_model(X_train, X_test, y_train, y_test)
    from tensorflow.keras.models import load_model
    loaded_model = load_model('my_model.h5')
    loaded_model.fit(X_train, y_train)
    y_pred = np.argmax(loaded_model.predict(X_test), axis=-1)  # because result is probabilities


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

yield1 = int((cm[1][1] + cm[2][2] - cm[0][1] - cm[0][2] - cm[1][2] - cm[2][1])) / int((cm[1][1] + cm[2][2] + cm[0][1] + cm[0][2] + cm[1][2] + cm[2][1]))
total_bets = cm[1][1] + cm[2][2] + cm[0][1] + cm[0][2] + cm[1][2] + cm[2][1]
income = cm[1][1] + cm[2][2] - cm[0][1] - cm[0][2] - cm[1][2] - cm[2][1]

print('Test income: ', income, '<-----------------------')
print('Test total placed bet: ', total_bets)
print('Test yield: ', yield1)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
plt.xlabel('Predicted')
plt.ylabel('True')
#wandb.log({'confusion_matrix': Image(fig)})
plt.show()

# to do:
# add result to dict
# add result to set