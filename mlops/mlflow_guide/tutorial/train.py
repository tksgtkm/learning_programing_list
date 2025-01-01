import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from generate_apple_dataset import generate_apple_sales_data_with_promo_adjustment

mlflow.set_tracking_uri("http://127.0.0.1:8080")

apple_experiment = mlflow.set_experiment("Apple_Models")

run_name = "apples_rf_test"

artifact_path = "rf_apples"

data = generate_apple_sales_data_with_promo_adjustment(base_demand=1_000, n_rows=1_000)

X = data.drop(columns=["date", "demand"])
y = data["demand"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    "n_estimators": 100,
    "max_depth": 6,
    "min_samples_split": 10,
    "min_samples_leaf": 4,
    "bootstrap": True,
    "oob_score": False,
    "random_state": 888
}

rf = RandomForestRegressor(**params)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)

mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)

metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        sk_model=rf,
        input_example=X_val,
        artifact_path=artifact_path
    )