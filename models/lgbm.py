from skopt import BayesSearchCV
import lightgbm as lgb
import matplotlib.pyplot as plt
from config import n_iter, cv

def lgbm_model(X_train, X_test, y_train, y_test):

    search_space = {
        'learning_rate': (0.01, 0.2, 'uniform'),
        'num_leaves': (20, 60),
        'max_depth': (5, 15, 'uniform'),
        'min_child_weight': (1, 10, 'uniform'),
        'n_estimators': (50, 200, 'uniform'),
        'subsample': (0.5, 1.0, 'uniform'),
        'colsample_bytree': (0.5, 1.0, 'uniform'),
        'reg_alpha': (0, 1.0, 'uniform'),
        'reg_lambda': (0, 1.0, 'uniform'),
    }

    lgbm_model = lgb.LGBMClassifier()

    bayes_search = BayesSearchCV(
        estimator=lgbm_model,
        search_spaces=search_space,
        n_iter=n_iter,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    bayes_search.fit(X_train, y_train)

    # Get the best parameters and estimator
    best_params = bayes_search.best_params_
    best_estimator = bayes_search.best_estimator_

    print(f"Best Parameters: {best_params}")

    best_estimator.booster_.save_model("best_lgbm_model.txt")

    # Plot feature importance
    lgb.plot_importance(best_estimator)
    plt.show(block=False)
    importances = best_estimator.feature_importances_
    feature_names = X_train.columns

    for feature_name, importance in zip(feature_names, importances):
        print(f"Feature {feature_name} : Importance {importance}")
