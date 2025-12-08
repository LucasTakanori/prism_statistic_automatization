from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet

from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures

def get_model(model_name, random_state=42, k_best=50, poly_degree=1, **kwargs):
    """
    Factory to create model instances.
    Each model is wrapped in a pipeline with Imputer, Scaler, and optional Feature Selection.
    
    Args:
        model_name (str): 'random_forest', 'ridge', 'mlp', 'svr', 'elasticnet', 'xgboost'.
        k_best (int): Number of features to select. Default 50.
        poly_degree (int): Degree of polynomial features. 1 = no interactions.
    """
    
    # Base preprocessing
    # Impute missing features with mean
    # Scale features
    steps = [
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]
    
    # Add Polynomial Features if requested
    if poly_degree > 1:
        steps.append(('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)))
        
    # Feature Selection
    steps.append(('selector', SelectKBest(score_func=f_regression, k=k_best)))
    
    model_name = model_name.lower()
    
    if model_name == 'random_forest':
        # Default RF parameters, can be overridden by kwargs
        rf_params = {
            'n_estimators': 100,
            'random_state': random_state,
            'n_jobs': -1
        }
        rf_params.update(kwargs) # Override with user args
        
        regressor = RandomForestRegressor(**rf_params)
    elif model_name == 'xgboost':
        regressor = XGBRegressor(random_state=random_state, n_jobs=-1, **kwargs)
    elif model_name == 'ridge':
        regressor = Ridge(random_state=random_state, **kwargs)
    elif model_name == 'elasticnet':
        regressor = ElasticNet(random_state=random_state, **kwargs)
    elif model_name == 'svr':
        # SVR doesn't take random_state usually (unless linear/poly with dual=False sometimes?)
        # Base SVR (RBF) is deterministic.
        # kwargs can contain C, epsilon, kernel, etc.
        regressor = SVR(**kwargs)
    elif model_name == 'mlp':
        regressor = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=random_state,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
        
    steps.append(('regressor', regressor))
    pipeline = Pipeline(steps=steps)
    return pipeline
