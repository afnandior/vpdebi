from sklearn.model_selection import GridSearchCV
import joblib
import os

def tune_hyperparameters(model, param_grid, X_train, y_train, save_path=None, cv=5, scoring="r2"):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best Params: {grid_search.best_params_}")
    print(f"Best {scoring}: {grid_search.best_score_}")

    best_model = grid_search.best_estimator_

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(best_model, save_path)
        print(f"Model saved to {save_path}")

    return best_model, grid_search
