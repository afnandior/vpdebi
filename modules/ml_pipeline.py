import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit

class MLModelPipeline:
    def __init__(self, df, features, target, test_size=0.2, random_state=42):
        self.df = df
        self.features = features
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

        # divide data
        self.X = self.df[self.features]
        self.y = self.df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

    def train_model(self, model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f" Model: {model.__class__.__name__}")
        print("MSE:", mse)
        print("R2:", r2)
        print("--------------")

        return {
            "model": model,
            "mse": mse,
            "r2": r2,
            "predictions": y_pred
        }

    #  Linear Models
    def run_linear_regression(self):
        print(" Running Linear Regression...")
        model = LinearRegression()
        return self.train_model(model)

    def run_ridge_regression(self, alpha=1.0):
        print(" Running Ridge Regression...")
        model = Ridge(alpha=alpha)
        return self.train_model(model)

    def run_lasso_regression(self, alpha=0.1):
        print(" Running Lasso Regression...")
        model = Lasso(alpha=alpha)
        return self.train_model(model)

    def run_elasticnet_regression(self, alpha=0.1, l1_ratio=0.5):
        print(" Running ElasticNet Regression...")
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        return self.train_model(model)

    #  Tree-based Models
    def run_decision_tree_regression(self):
        print(" Running Decision Tree Regression...")
        model = DecisionTreeRegressor()
        return self.train_model(model)

    def run_random_forest_regression(self, n_estimators=100):
        print(" Running Random Forest Regression...")
        model = RandomForestRegressor(n_estimators=n_estimators)
        return self.train_model(model)

    def run_gradient_boosting_regression(self, n_estimators=100):
        print(" Running Gradient Boosting Regression...")
        model = GradientBoostingRegressor(n_estimators=n_estimators)
        return self.train_model(model)

    #  Polynomial Regression
    def run_polynomial_regression(self, degree=2):
        print(f" Running Polynomial Regression (degree={degree})...")
        poly = PolynomialFeatures(degree=degree)
        X_poly_train = poly.fit_transform(self.X_train)
        X_poly_test = poly.transform(self.X_test)

        model = LinearRegression()
        model.fit(X_poly_train, self.y_train)
        y_pred = model.predict(X_poly_test)

        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print("MSE:", mse)
        print("R2:", r2)
        print("--------------")

        return {
            "model": model,
            "mse": mse,
            "r2": r2,
            "predictions": y_pred
        }

    #  Exponential Regression
    def run_exponential_regression(self):
        print(" Running Exponential Regression...")
        X_train_flat = self.X_train.values.flatten()
        y_train = self.y_train.values

        def exp_func(x, a, b):
            return a * np.exp(b * x)

        popt, _ = curve_fit(exp_func, X_train_flat, y_train, p0=(1, 0.1))

        X_test_flat = self.X_test.values.flatten()
        y_pred = exp_func(X_test_flat, *popt)

        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print("MSE:", mse)
        print("R2:", r2)
        print("--------------")

        return {
            "model": f"Exponential Regression a={popt[0]}, b={popt[1]}",
            "mse": mse,
            "r2": r2,
            "predictions": y_pred
        }

    #  Logarithmic Regression
    def run_logarithmic_regression(self):
        print(" Running Logarithmic Regression...")
        X_train_flat = self.X_train.values.flatten()
        y_train = self.y_train.values

        def log_func(x, a, b):
            return a + b * np.log(x)

        popt, _ = curve_fit(log_func, X_train_flat, y_train, p0=(1, 1))

        X_test_flat = self.X_test.values.flatten()
        y_pred = log_func(X_test_flat, *popt)

        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print("MSE:", mse)
        print("R2:", r2)
        print("--------------")

        return {
            "model": f"Logarithmic Regression a={popt[0]}, b={popt[1]}",
            "mse": mse,
            "r2": r2,
            "predictions": y_pred
        }

    #  Power Regression
    def run_power_regression(self):
        print(" Running Power Regression...")
        X_train_flat = self.X_train.values.flatten()
        y_train = self.y_train.values

        def power_func(x, a, b):
            return a * np.power(x, b)

        popt, _ = curve_fit(power_func, X_train_flat, y_train, p0=(1, 1))

        X_test_flat = self.X_test.values.flatten()
        y_pred = power_func(X_test_flat, *popt)

        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print("MSE:", mse)
        print("R2:", r2)
        print("--------------")

        return {
            "model": f"Power Regression a={popt[0]}, b={popt[1]}",
            "mse": mse,
            "r2": r2,
            "predictions": y_pred
        }
