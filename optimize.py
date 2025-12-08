import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from ml_pipeline.data_loader import load_data
from ml_pipeline.models import get_model
import argparse
import joblib
import os

def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def optimize(target, output_dir="optimization_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    # We use all available training data for CV
    print(f"Loading data for optimization (Target: {target})...")
    # We load standard train/val split but we will merge them back or just use Train for CV?
    # Standard practice: Optimize on Train, Verify on Val.
    train_df, val_df = load_data()
    
    def prepare(df):
        df_clean = df.dropna(subset=[target])
        y = df_clean[target]
        # Metadata cols to drop
        ignore = ['subject', 'source_file', 'sex', 'category', 'date', 'session', 
                  'audio_type', 'item_id', 'target_transcription', 'task_id', 'duration',
                  'ALSFRS_R', 'FVC', 'ALSFRS-R_TotalScore', 'FVC% ']
        X = df_clean.drop(columns=[c for c in ignore if c in df_clean.columns], errors='ignore')
        X = X.select_dtypes(include=[np.number])
        return X, y
        
    X_train, y_train = prepare(train_df)
    
    print(f"Optimization Set: {X_train.shape} samples")
    
    # 2. Define Param Grids
    # We need to construct the pipeline via get_model inside the scorer? 
    # Or better, we use GridSearchCV on the Pipeline returned by get_model.
    # But get_model takes arguments.
    # We can create a wrapper or just manually instantiate pipelines for each model type.
    
    models_to_tune = {
        'svr': {
            'model_name': 'svr',
            'param_grid': {
                'selector__k': [20, 50, 100],
                'regressor__C': [1, 10, 100],
                'regressor__gamma': ['scale', 0.1, 0.01],
                'regressor__epsilon': [0.1, 0.01]
            }
        },
        'xgboost': {
            'model_name': 'xgboost',
            'param_grid': {
                'selector__k': [20, 50],
                'regressor__n_estimators': [100, 200],
                'regressor__max_depth': [3, 5],
                'regressor__learning_rate': [0.01, 0.1],
                'regressor__subsample': [0.8, 1.0]
            }
        },
        'ridge_poly': {
            'model_name': 'ridge',
            'param_grid': {
                'poly_degree': [2], # Explicitly test interaction
                'selector__k': [20, 50],
                'regressor__alpha': [10, 100, 1000] 
            }
        },
        'elasticnet': {
            'model_name': 'elasticnet',
            'param_grid': {
                'selector__k': [20, 50],
                'regressor__alpha': [1, 10],
                'regressor__l1_ratio': [0.1, 0.5, 0.9]
            }
        },
        'random_forest': {
            'model_name': 'random_forest',
            'param_grid': {
                'selector__k': [50],
                'regressor__n_estimators': [100],
                'regressor__max_depth': [3, 5],
                'regressor__min_samples_leaf': [5, 10]
            }
        }
    }
    
    results = []
    
    # CV Strategy
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, config in models_to_tune.items():
        print(f"\n--- Tuning {name} ---")
        
        # We need to handle 'poly_degree' if it's in param_grid.
        # But GridSearchCV takes an estimator and params.
        # 'poly_degree' determines the STRUCTURE of the pipeline (adding a step), not just a param.
        # So we can't easily grid search it unless the step always exists and we set 'degree'.
        # However, our get_model adds the step conditionally.
        
        # Workaround: For 'ridge_poly', we instantiate with poly_degree=2.
        # And remove poly_degree from param_grid passed to GridSearch.
        
        model_kwargs = {}
        grid_params = config['param_grid'].copy()
        
        if 'poly_degree' in grid_params:
            # We assume a single value for poly_degree for now, or we iterate manually.
            # GridSearch can't handle changing pipeline steps effectively without "SelectStep" trick.
            # Let's support simple integer in list.
            degrees = grid_params.pop('poly_degree')
            # For simplicity, if multiple degrees, we'd loop.
            # Let's accept the first one for now as per our config [2].
            model_kwargs['poly_degree'] = degrees[0]
            
        base_pipeline = get_model(config['model_name'], **model_kwargs)
        
        # Grid Search
        grid = GridSearchCV(
            base_pipeline,
            grid_params,
            cv=cv,
            scoring='r2', 
            n_jobs=-1,
            verbose=1
        )
        
        grid.fit(X_train, y_train)
        
        print(f"Best CV R2: {grid.best_score_:.4f}")
        print(f"Best Params: {grid.best_params_}")
        
        results.append({
            'model': name,
            'best_r2': grid.best_score_,
            'best_params': grid.best_params_,
            'best_estimator': grid.best_estimator_
        })
        
    # Sort by score
    results.sort(key=lambda x: x['best_r2'], reverse=True)
    
    # Save best overall
    best_result = results[0]
    print(f"\n\n=== WINNER: {best_result['model']} (R2: {best_result['best_r2']:.4f}) ===")
    print(f"Params: {best_result['best_params']}")
    
    # Save results to text file
    with open(os.path.join(output_dir, f"optimization_report_{target}.txt"), "w") as f:
        for res in results:
            f.write(f"Model: {res['model']}\n")
            f.write(f"CV R2: {res['best_r2']:.4f}\n")
            f.write(f"Params: {res['best_params']}\n")
            f.write("-" * 30 + "\n")
            
    print(f"Report saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True)
    args = parser.parse_args()
    
    optimize(args.target)
