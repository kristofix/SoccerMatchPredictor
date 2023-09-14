import xgboost as xgb
from skopt import BayesSearchCV
from config import n_iter, cv
def xgb_model(X_train, X_test, y_train, y_test):
    xgb_model = xgb.XGBClassifier()

    param_space = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'max_depth': (1, 50),
        'min_child_weight': (0, 10),
        'n_estimators': (50, 1000),
        'subsample': (0.1, 1.0, 'uniform'),
        'gamma': (0, 10),
        'colsample_bytree': (0.1, 1.0, 'uniform'),
        'reg_alpha': (0, 10),
        'reg_lambda': (0, 10),
    }

    opt = BayesSearchCV(
        xgb_model,
        param_space,
        n_iter=n_iter,
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    opt.fit(X_train, y_train)

    import json

    # Save to file
    with open("best_params.json", "w") as f:
        json.dump(opt.best_params_, f)
    #
    #
    # # Load from file
    # with open("best_params.json", "r") as f:
    #     loaded_params = json.load(f)
