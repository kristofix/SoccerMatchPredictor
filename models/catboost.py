from skopt import BayesSearchCV
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from config import n_iter, cv

def catboost_model(X_train, X_test, y_train, y_test):

    search_space = {
        'learning_rate': (0.01, 0.2, 'uniform'),
        'depth': (4, 10),
        'l2_leaf_reg': (1, 10, 'uniform'),
        'iterations': (50, 200),
        'border_count': (32, 255),
        'subsample': (0.5, 1.0, 'uniform'),
        'random_strength': (0, 1.0, 'uniform'),
        'bootstrap_type': ['Bernoulli']
    }

    catboost_model = CatBoostClassifier(verbose=0)

    bayes_search = BayesSearchCV(
        estimator=catboost_model,
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

    best_estimator.save_model("best_catboost_model.cbm")

    # Plot feature importance
    feature_importances = best_estimator.get_feature_importance()
    feature_names = X_train.columns

    plt.barh(range(len(feature_importances)), feature_importances)
    plt.yticks(range(len(feature_importances)), [feature_names[i] for i in range(len(feature_importances))])
    plt.xlabel('Feature Importances')
    plt.show(block=False)

    for feature_name, importance in zip(feature_names, feature_importances):
        print(f"Feature {feature_name} : Importance {importance}")
