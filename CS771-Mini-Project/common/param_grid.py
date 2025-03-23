param_grids = {
    "xgb" : {
        'n_estimators': [50, 200, 500],
        'max_depth': [3, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0],
        'gamma': [0, 0.2],
        'min_child_weight': [1, 3],
        'eval_metric': ['logloss', 'auc']
    },
    "lr" : {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': [0.1, 1.0, 10.0],
        'solver': ['liblinear', 'saga', 'lbfgs'],
        'fit_intercept': [True]
    },
    "rf" : {
        'n_estimators': [100, 500],
        'max_depth': [None, 10, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    },
    "mlp" : {
        'activation': ['tanh', 'relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.01],
        'learning_rate': ['adaptive']
    },
    "svc" : {
        'C': [0.1, 10],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 4],
        'gamma': ['scale', 'auto']
    },
    "mnb" : {
        'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
        'fit_prior': [True, False]
    },
}