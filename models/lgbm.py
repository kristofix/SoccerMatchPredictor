import lightgbm as lgb
import matplotlib.pyplot as plt
import json

def lgbm_model(X_train, y_train):

    # Manually set hyperparameters - todo: change to bayessearch
    params = {
        'learning_rate': 0.1,
        'max_depth': 10,
        'min_child_weight': 1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0,
        'reg_lambda': 1,
    }

    lgbm_model = lgb.LGBMClassifier(**params)
    lgbm_model.fit(X_train, y_train)

    lgbm_model.booster_.save_model("best_lgbm_model.txt")

    # Plot feature importance
    lgb.plot_importance(lgbm_model)
    plt.show(block=False)

    # Get and print feature importances
    importances = lgbm_model.feature_importances_
    feature_names = X_train.columns

    for feature_name, importance in zip(feature_names, importances):
        print(f"Feature {feature_name} : Importance {importance}")

