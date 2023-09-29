from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from config import min_time, minbetodd, maxbetodd, insufficient, n_iter
from common_function import dif_threshold, calculate_metrics, select_and_train_model, sport_metrics
from data_preparation import data_preparation
import pandas as pd
from analyze_predictions.plot_income_with_odd import plot_income_flow

#Select model
model_selector = "nn"  #  "catboost", "lgbm", "xgb", "nn" or "nn_custom_loss" [nn - neural_network]

wandb.init(
    project="version 9.2 9 2023",
    notes=model_selector,
    tags=["xgb","nn","lgbm"]
)

# Run pipeline with normalize, cleaning, sorting and filtering dataset
df = data_preparation()

# df = df.head(1500) #uncomment for quick test run <------------------------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(df.drop('zzz_play', axis=1), df['zzz_play'], test_size=0.2,random_state=42)
y_pred = select_and_train_model(model_selector, X_train, X_test, y_train, y_test)

# Metrics, confusion matrix and printing results
accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
yield1, total_bets, income = sport_metrics(cm)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
plt.xlabel('Predicted')
plt.ylabel('True')

# Log results to wandb
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

# save X_test and y_test with predictions to csv
combined_df = pd.concat([X_test, y_test], axis=1)
combined_df['predict'] = y_pred
combined_df.to_csv('test_and_predictions.csv', index=False)


plot_income_flow()

