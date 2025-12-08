import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from ml_pipeline.data_loader import load_data
from ml_pipeline.models import get_model
import joblib
import os
import json
from datetime import datetime

def train_model(
    target_name,
    model_name="random_forest",
    output_dir="results",
    **kwargs
):
    """
    Main training routine.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"{target_name}_{model_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Starting training for Target: {target_name}, Model: {model_name}")
    
    # 1. Load Data
    train_df, val_df = load_data()
    
    # 2. Prepare X and y
    # We need to identify feature columns. 
    # Usually: drop metadata columns + targets.
    # Metadata candidates: subject, sex, category, date, session, audio_type, item_id, target_transcription, source_file...
    # Safe bet: Select numeric types and drop known targets/ids.
    
    ignore_cols = [
        'subject', 'ID', 'sex', 'category', 'date', 'session', 
        'audio_type', 'item_id', 'target_transcription', 'source_file',
        'ALSFRS_R', 'FVC'
    ]
    
    # Also ignore columns that might be other targets or strings
    # We will select only numeric features for now
    
    def get_features_targets(df):
        # Drop rows where target is NaN
        df_clean = df.dropna(subset=[target_name])
        
        y = df_clean[target_name]
        
        # Select features
        X = df_clean.drop(columns=[c for c in ignore_cols if c in df_clean.columns], errors='ignore')
        X = X.select_dtypes(include=[np.number])
        
        return X, y

    X_train, y_train = get_features_targets(train_df)
    X_val, y_val = get_features_targets(val_df)
    
    # 2.5 Drop columns that are completely NaN in Train (Imputer can't handle them)
    # We should apply the same drop to Val
    valid_cols = X_train.columns[X_train.notna().any()].tolist()
    # Also drop columns that have 0 variance? Maybe later.
    
    X_train = X_train[valid_cols]
    X_val = X_val[valid_cols]
    
    print(f"Training features: {X_train.shape[1]}")
    
    if X_train.empty:
        raise ValueError("No valid features found after cleaning.")
    
    # 3. Initialize Model
    model = get_model(model_name, **kwargs)
    
    # 4. Train
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Metric Calculation Helper
    def get_metrics(y_true, y_pred, prefix=""):
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        return {f"{prefix}rmse": rmse, f"{prefix}r2": r2}
    
    metrics = {}
    metrics.update(get_metrics(y_train, y_pred_train, "train_"))
    metrics.update(get_metrics(y_val, y_pred_val, "val_"))
    
    # 6. Save Metrics
    print("Metrics:", json.dumps(metrics, indent=2))
    
    joblib.dump(model, os.path.join(run_dir, "model.pkl"))
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    # 7. Save Predictions
    # Save both Train and Val
    # Keep only metadata and predictions
    
    def save_preds(X, y_true, y_pred, name):
        # Re-retrieve metadata using index
        meta = val_df.copy() if name == 'val' else train_df.copy()
        meta = meta.loc[X.index, ['subject', 'sex']].copy()
        meta['true'] = y_true
        meta['pred'] = y_pred
        meta.to_csv(os.path.join(run_dir, f"predictions_{name}.csv"), index=False)
        
    save_preds(X_train, y_train, y_pred_train, 'train')
    save_preds(X_val, y_val, y_pred_val, 'val')
        
    print(f"Results saved to {run_dir}")
    return metrics, run_dir

if __name__ == "__main__":
    # Test run
    # train_model("ALSFRS_R")
    pass
