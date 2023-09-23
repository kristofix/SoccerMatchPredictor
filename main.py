from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
import time
import wandb
from xgb_with_bayes_search import xgb_model
from neural_network import nn_model
from config import min_time, minbetodd, maxbetodd, insufficient, n_iter
from common_function import dif_threshold
from data_preparation import data_preparation
from statistics_checks.outlier_detection import z_score_outlier

wandb.init(
    project="09-23 xgb and nn",
    notes="xgb drop sumstats",
    tags=["xgb","nn"]
)

start_time = time.time()

df = data_preparation()

# df = df.head(1000) #uncomment for quick test run <-----------------------------------------------

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

# remove outliers
df = z_score_outlier(df)

X_train, X_test, y_train, y_test = train_test_split(df.drop('zzz_play', axis=1), df['zzz_play'], test_size=0.2,random_state=42)

# SELECT YOUR MODEL <-----------------------------------------------------------
# neural network - 1
# XGB - 2
# Use selector to choose model

model_selector = 1

if model_selector == 1:
    nn_model(X_train, X_test, y_train, y_test)
    from tensorflow.keras.models import load_model
    loaded_model = load_model('nn_model.keras')
    y_pred = np.argmax(loaded_model.predict(X_test), axis=-1)  # because result is probabilities
elif model_selector == 2:
    xgb_model(X_train, X_test, y_train, y_test)
    # Initialize an empty XGBoost model
    loaded_model = xgb.XGBClassifier()
    loaded_model.load_model("best_xgb_model.model")
    y_pred = loaded_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy:", accuracy)
precision = precision_score(y_test, y_pred, average='weighted')
print("Test set precision:", precision)
recall = recall_score(y_test, y_pred, average='weighted')
print("Test set recall:", recall)
f1 = f1_score(y_test, y_pred, average='weighted')
print("Test set f1 Score:", f1)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

yield1 = int((cm[1][1] + cm[2][2] - cm[0][1] - cm[0][2] - cm[1][2] - cm[2][1])) / int((cm[1][1] + cm[2][2] + cm[0][1] + cm[0][2] + cm[1][2] + cm[2][1]))
total_bets = cm[1][1] + cm[2][2] + cm[0][1] + cm[0][2] + cm[1][2] + cm[2][1]
income = cm[1][1] + cm[2][2] - cm[0][1] - cm[0][2] - cm[1][2] - cm[2][1]

print('Test income: ', income)
print('Test total placed bet: ', total_bets)
print('Test yield: ', yield1)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show(block=False)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Function took {elapsed_time} seconds to run.")

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