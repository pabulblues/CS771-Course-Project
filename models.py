from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from param_grid import param_grids
import numpy as np

def predict_xgboost(x_train, y_train, x_val,grid_search : bool = False, eval_metric='logloss') :
    '''
    inputs : training data and validation data in dataframe
    outputs : validation predictions and truth values
    '''
    if grid_search :
        xgb_model = XGBClassifier(eval_metric=eval_metric)
        best_params = grid_search_(xgb_model, x_train, y_train, model_name='xgb')
        
        xgb_model = XGBClassifier(**best_params)
        xgb_model.fit(x_train, y_train)
        
        y_pred = xgb_model.predict(x_val)

        return y_pred
    else :
        xgb_model = XGBClassifier(eval_metric=eval_metric)
        xgb_model.fit(x_train, y_train)
        
        y_pred = xgb_model.predict(x_val)

        return y_pred


def predict_logistic_regression(x_train, y_train, x_val,grid_search : bool = False, max_iter=10000, random_state=42) :
    '''
    inputs : training data and validation data in dataframe
    outputs : validation predictions and truth values
    '''
    if grid_search :
        lr_model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        best_params = grid_search_(lr_model, x_train, y_train, model_name='lr')
        
        lr_model = LogisticRegression(**best_params, max_iter=max_iter, random_state=random_state)
        lr_model.fit(x_train, y_train)
        
        y_pred = lr_model.predict(x_val)

        return y_pred
    else :
        lr_model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        lr_model.fit(x_train, y_train)
        
        y_pred = lr_model.predict(x_val)

        return y_pred

def predict_random_forest(x_train, y_train, x_val, grid_search : bool = False, n_estimators=100, random_state=42, min_samples_leaf=1, max_features='sqrt') :
    '''
    inputs : training data and validation data in dataframe
    outputs : validation predictions and truth values
    '''
    
    if grid_search :
        rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, min_samples_leaf=min_samples_leaf, max_features=max_features)
        best_params = grid_search_(rf_model, x_train, y_train, model_name='rf')
        
        rf_model = RandomForestClassifier(**best_params, random_state=random_state)
        rf_model.fit(x_train, y_train)
        
        y_pred = rf_model.predict(x_val)

        return y_pred
    else :
        rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, min_samples_leaf=min_samples_leaf, max_features=max_features)
        rf_model.fit(x_train, y_train)
        
        y_pred = rf_model.predict(x_val)

        return y_pred

def predict_mlp(x_train, y_train, x_val, grid_search : bool = False, max_iter=10000, random_state=42, hidden_layer_sizes=(16,)) :
    '''
    inputs : training data and validation data in dataframe
    outputs : validation predictions and truth values
    '''
    if grid_search :
        mlp_model = MLPClassifier(max_iter=max_iter, random_state=random_state, hidden_layer_sizes=hidden_layer_sizes)
        best_params = grid_search_(mlp_model, x_train, y_train, model_name='mlp')
        
        mlp_model = MLPClassifier(**best_params, max_iter=max_iter, random_state=random_state)
        mlp_model.fit(x_train, y_train)
        
        y_pred = mlp_model.predict(x_val)
        
        return y_pred
    
    else :
        mlp_model = MLPClassifier(max_iter=max_iter, random_state=random_state, hidden_layer_sizes=hidden_layer_sizes)
        mlp_model.fit(x_train, y_train)
        
        y_pred = mlp_model.predict(x_val)
        
        def count_parameters_sklearn_mlp(model):
            total_params = 0
            # Summing number of parameters in each layer (weights + biases)
            for coef in model.coefs_:
                total_params += np.prod(coef.shape)  # Number of weights
            for intercept in model.intercepts_:
                total_params += intercept.shape[0]  # Number of biases
            return total_params

        print("Number of parameters in the MLP model: ", count_parameters_sklearn_mlp(mlp_model))
        return y_pred

def predict_svc(x_train, y_train, x_val, grid_search : bool = False, random_state=42, kernel='rbf', c=1.0, gamma='scale', max_iter=10000) :
    '''
    inputs : training data and validation data in dataframe
    outputs : validation predictions and truth values
    '''
    if grid_search :
        svm_model = SVC(random_state=random_state, kernel=kernel, C=c, gamma=gamma, max_iter=max_iter)
        best_params = grid_search_(svm_model, x_train, y_train, model_name='svc')
        
        svm_model = SVC(**best_params, random_state=random_state)
        svm_model.fit(x_train, y_train)
        
        y_pred = svm_model.predict(x_val)

        return y_pred
    else :
        svm_model = SVC(random_state=random_state, kernel=kernel, C=c, gamma=gamma, max_iter=max_iter)
        svm_model.fit(x_train, y_train)
        
        y_pred = svm_model.predict(x_val)

        return y_pred

def predict_mnb(x_train, y_train, x_val, grid_search : bool = False, alpha=1.0, fit_prior=True) :
    '''
    inputs : training data and validation data in dataframe
    outputs : validation predictions and truth values
    '''
    if grid_search :
        mnb_model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
        best_params = grid_search_(mnb_model, x_train, y_train, model_name='mnb')
        
        mnb_model = MultinomialNB(**best_params)
        mnb_model.fit(x_train, y_train)
        y_pred = mnb_model.predict(x_val)
        
        return y_pred
    
    else :
        mnb_model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
        mnb_model.fit(x_train, y_train)
        
        y_pred = mnb_model.predict(x_val)

        return y_pred

def grid_search_(model, X_train, y_train, param_grid = None,model_name : str = None, cv = 5, scoring = 'accuracy', n_jobs = 4) :
    '''
    inputs : model, associated parameter grid, cross validation, scoring metric, seed
    outputs : best parameters and the associated score (accuracy by default)
    '''
    if param_grid is None :
        if model_name is None :
            raise ValueError("Either param_grid or model_name should be provided")
        elif model_name not in param_grids :
            raise ValueError("Model name not found in param_grids")
        param_grid = param_grids[model_name]
        
    grid_search = GridSearchCV(model, param_grid = param_grid, cv = cv, scoring = scoring, n_jobs= n_jobs)
    grid_search.fit(X_train, y_train)
    # write code so that it breaks if the param_grid does not match the associated model parameters

    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
    
    return grid_search.best_params_
    