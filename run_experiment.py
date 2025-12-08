import argparse
import sys
from ml_pipeline.train import train_model

def main():
    parser = argparse.ArgumentParser(description="Run ALS Prediction Experiment")
    parser.add_argument("--target", type=str, required=True, choices=['ALSFRS_R', 'FVC'], help="Target metric to predict")
    parser.add_argument("--model", type=str, default="random_forest", choices=['random_forest', 'ridge', 'mlp', 'svr', 'elasticnet', 'xgboost'], help="Model architecture")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    # Feature selection arg
    parser.add_argument("--k_best", type=int, default=50, help="Number of features to select")
    parser.add_argument("--poly_degree", type=int, default=1, help="Polynomial degree. 1=None, 2=Interactions")
    
    # Parse known args, leave the rest for model kwargs
    args, unknown = parser.parse_known_args()
    
    # Convert unknown args (e.g. --max_depth 5) to kwargs
    kwargs = {}
    i = 0
    while i < len(unknown):
        key = unknown[i]
        if key.startswith("--"):
            key = key[2:]
            if i + 1 < len(unknown) and not unknown[i+1].startswith("--"):
                val = unknown[i+1]
                # Try to convert to int/float
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                kwargs[key] = val
                i += 2
            else:
                print(f"Warning: Argument {key} has no value, ignoring.")
                i += 1
        else:
            i += 1
            
    # Add k_best/poly_degree to kwargs
    kwargs['k_best'] = args.k_best
    kwargs['poly_degree'] = args.poly_degree
    
    print(f"Parsed Model Kwargs: {kwargs}")
    
    try:
        metrics, run_dir = train_model(
            target_name=args.target,
            model_name=args.model,
            output_dir=args.output_dir,
            **kwargs
        )
        
        print("\n=== Experiment Completed ===")
        print(f"Target: {args.target}")
        print(f"Model: {args.model}")
        print(f"Test RMSE: {metrics['val_rmse']:.4f}")
        print(f"Test R2: {metrics['val_r2']:.4f}")
        print(f"Saved to: {run_dir}")
        
    except Exception as e:
        print(f"Experiment Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
