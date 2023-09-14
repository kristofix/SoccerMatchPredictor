def model_lgbm(df):

  wandb.init(
      project="LightGBM Bayes",
      notes="LightGBM 40prc train 60hold",
      tags=["LGBM", "Bayes"]
  )

  # Split the dataset into training, validation, and testing sets
  X_train, X_test, y_train, y_test = train_test_split(df.drop('zzz_play', axis=1), dfTrain['zzz_play'], test_size=0.2,random_state=42)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,random_state=42)

  # Define the search space for the hyperparameters
  space = {
      'learning_rate': hp.loguniform('learning_rate', -5, 0),
      'n_estimators': hp.choice('n_estimators', range(400, 750)),
      'max_depth': hp.choice('max_depth', range(3, 5)),
      'min_child_samples': hp.choice('min_child_samples', range(1, 11)),
      'min_child_weight': hp.choice('min_child_weight', range(1, 11)),
      'subsample': hp.uniform('subsample', 0.5, 1),
      'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
      'reg_lambda': hp.uniform('reg_lambda', 0, 1),
      'reg_alpha': hp.uniform('reg_alpha', 0, 1),
  }

  n_iter = 30 ###########################################################################################################################
  # Train and evaluate the model with the best hyperparameters
  # Map the best hyperparameters back to their corresponding values
  best_params = {
      'learning_rate': best['learning_rate'],
      'n_estimators': int(400 + best['n_estimators']),
      'max_depth': int(3 + best['max_depth']),
      'min_child_samples': int(1 + best['min_child_samples']),
      'min_child_weight': int(1 + best['min_child_weight']),
      'subsample': best['subsample'],
      'colsample_bytree': best['colsample_bytree'],
      'reg_lambda': best['reg_lambda'],
      'reg_alpha': best['reg_alpha']
  }




  # Train and evaluate the model with the best hyperparameters
  model = lgb.LGBMClassifier(**best_params)
  def log_metrics_custom(eval_result, config=None):
      wandb.log({
          'train_loss': eval_result[0][2],
          'valid_loss': eval_result[1][2]
      })
      return False


  model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=False)
  #model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=False, callbacks=[WandbCallback(log_model=True)])

  # Make predictions on the test data
  y_pred = model.predict(X_test)
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

  yieldd = int((cm[1][1]+cm[2][2]-cm[0][1]-cm[0][2]-cm[1][2]-cm[2][1]))/int((cm[1][1]+cm[2][2]+cm[0][1]+cm[0][2]+cm[1][2]+cm[2][1]))#
  total_bets= cm[1][1]+cm[2][2]+cm[0][1]+cm[0][2]+cm[1][2]+cm[2][1]#
  income = cm[1][1]+cm[2][2]-cm[0][1]-cm[0][2]-cm[1][2]-cm[2][1]#

  print("best_params", best_params)
  print("Confusion Matrix:\n", cm)
  print('Minute: ', min+1)
  print('Min played odd:', minbetodd)
  print('Max played odd:', maxbetodd)
  print('Income: ', income)
  print('Total bets placed : ', total_bets)
  print('Yield: ', yieldd)

  #new wandb funct
  from wandb import Image
  fig, ax = plt.subplots(figsize=(10, 6))
  lgb.plot_importance(model, ax=ax)
  wandb.log({'feature_importances': Image(fig)})
  plt.show()

  #new wandb funct
  fig, ax = plt.subplots()
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
  plt.xlabel('Predicted')
  plt.ylabel('True')
  wandb.log({'confusion_matrix': Image(fig)})
  plt.show()



  wandb.log({
      'Total bets placed_holdout': total_bets_holdout,
      'income_holdout': income_holdout,
      'yield_holdout': yieldd_holdout,
      'Minute': min+1,
      'Min odd:': minbetodd,
      'Max odd:': maxbetodd,
      'Total bets placed': total_bets,
      'final_accuracy': accuracy,
      'precision': precision,
      'recall': recall,
      'f1_score': f1,
      'income': income,
      'yield': yieldd,
      'odds_interval': (minbetodd, maxbetodd),
      'n_iter': n_iter,
      #'best_params': best_params,
      #'params': params,
      'dif' : dif_threshold,
      'insufficient' : insufficient,
      'efficienty' : efficienty,
      'min_date_dfTrain' : min_date_dfTrain,
      'max_date_dfTrain' : max_date_dfTrain,
      'min_date_holdout' : min_date_holdout,
      'max_date_holdout' : max_date_holdout,
      'played_from_qualified' : played_from_qualified
  })

  # Save the LightGBM model
  from datetime import datetime
  now = datetime.now()
  current_time = now.strftime("%d%b%Y%H%M")

  booster = model.booster_
  booster.save_model(f'best_model_LGBM_Hyperopt_date_{current_time}_min_16_dif_1_insf_20_odd(1_9-2_4).txt')

  import os

  # Save the model to WandB
  local_directory = '/content/model/'
  os.makedirs(local_directory, exist_ok=True)
  wandb.save(os.path.join(local_directory, 'lgbm_model.txt'))
  wandb.finish()


