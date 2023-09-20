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
import time
import wandb
from XGBwithBayesSearch import xgb_model
from neuralNetwork import nn_model
from config import min_time, max_time, minbetodd, maxbetodd, insufficient, threshold, n_iter
from common import removeDotFromColumnNames, dropMinutes, sortByDate, dropNotDraw, oddsFilter,addSumStats, dif_threshold, dropInsufficient

wandb.init(
    project="09-23 xgb and nn",
    #notes="SMOTE, 3 layers only, no regularization",
    notes="xgb drop sumstats,frameshomeodd",
    tags=["xgb","nn"]
)

start_time = time.time() # to compare running time with and without normalization

matplotlib.use('TkAgg')

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

df = pipeline(df)

df.drop(['datetimestamp', 'sumAstats', 'sumBstats', 'frameshomeodd'], axis=1, inplace=True) #mainly because of high correlation

print(df.shape)
#df = df.iloc[:len(df)//1000]      #uncomment for quick test run <-----------------------------------------------
print(df.shape)
df.to_csv('your_file_name1.csv', index=False)

#Normalize data because data do not follow GaussianDistribution - checked in other file
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

print(df.head(5))
df.to_csv('your_file_name2.csv', index=False)

X_train, X_test, y_train, y_test = train_test_split(df.drop('zzz_play', axis=1), df['zzz_play'], test_size=0.2,random_state=42)


# SELECT YOUR MODEL <-----------------------------------------------------------
# neural network - 1
# XGB - 2
# Use selector to choose model

model_selector = 2

if model_selector == 2:
    xgb_model(X_train, X_test, y_train, y_test)
    #Load params from xgb training
    with open("best_params.json", "r") as f:
       loaded_params = json.load(f)
    loaded_model = xgb.XGBClassifier(**loaded_params)
    y_pred = loaded_model.predict(X_test)
elif model_selector == 1:
    nn_model(X_train, X_test, y_train, y_test)
    from tensorflow.keras.models import load_model
    loaded_model = load_model('my_model.h5')
    y_pred = np.argmax(loaded_model.predict(X_test), axis=-1)  # because result is probabilities


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
precision = precision_score(y_test, y_pred, average='weighted')
print("precision:", precision)
recall = recall_score(y_test, y_pred, average='weighted')
print("Recall:", recall)
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score:", f1)

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
# wandb.log({'confusion_matrix': Image(fig)})
plt.show(block=False)

# to do:
# add result to dict
# add result to set

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Function took {elapsed_time} seconds to run.") #78,43 with normalization and 63 without for small dataset - strange - need to check in future.

plt.show()

wandb.log({
    'Total bets placed': total_bets,
    'income': income,
    'yield': yield1,
    'Minute': min_time + 1,
    'Min odd:': minbetodd,
    'Max odd:': maxbetodd,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'odds_interval': (minbetodd, maxbetodd),
    'n_iter': n_iter,
    'dif': dif_threshold,
    'insufficient': insufficient,
})

wandb.finish()