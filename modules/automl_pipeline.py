
from modules.ml_pipeline import MLModelPipeline
import joblib
import os

def run_automl(df, features, target, save_dir="outputs/saved_models"):
    pipeline = MLModelPipeline(df, features, target)
    results = []
    results.append(pipeline.run_linear_regression())
    results.append(pipeline.run_ridge_regression())
    results.append(pipeline.run_lasso_regression())
    results.append(pipeline.run_elasticnet_regression())
    results.append(pipeline.run_decision_tree_regression())
    results.append(pipeline.run_random_forest_regression())
    results.append(pipeline.run_gradient_boosting_regression())
    results.append(pipeline.run_polynomial_regression(degree=2))
    results.append(pipeline.run_exponential_regression())
    results.append(pipeline.run_logarithmic_regression())
    results.append(pipeline.run_power_regression())

    best_result = max(results, key=lambda x: x['r2'])

    print(f"\n Best Model: {best_result['model']}")
    print(f" Best R2 Score: {best_result['r2']}")

    os.makedirs(save_dir, exist_ok=True)

    best_model = best_result['model']
    if hasattr(best_model, "predict"):
        filename = os.path.join(save_dir, f"best_model.pkl")
        joblib.dump(best_model, filename)
        print(f" Best model saved at: {filename}")
    else:
        print(" Best model is not a sklearn model object, skipping save.")

    return best_result, results

