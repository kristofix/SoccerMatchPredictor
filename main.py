from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time
import wandb
from config import min_time, minbetodd, maxbetodd, insufficient, n_iter
from common_function import dif_threshold, calculate_metrics, select_and_train_model
from data_preparation import data_preparation
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

wandb.init(
    project="09-23 xgb and nn",
    notes="xgb drop sumstats",
    tags=["xgb","nn"]
)

start_time = time.time()

df = data_preparation()
df = df.head(1000) #uncomment for quick test run <------------------------------------------------------------------------------------------------------

# scaler = MinMaxScaler()
# df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

X_train, X_test, y_train, y_test = train_test_split(df.drop('zzz_play', axis=1), df['zzz_play'], test_size=0.2,random_state=42)

#Select model
model_selector = "xgb"  # "nn" or "xgb"
y_pred = select_and_train_model(model_selector, X_train, X_test, y_train, y_test)

accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
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

# save test with predictions to csv
combined_df = pd.concat([X_test, y_test], axis=1)
combined_df['predict'] = y_pred
combined_df.to_csv('test_and_predictions.csv', index=False)
