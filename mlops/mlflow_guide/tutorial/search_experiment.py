from mlflow import MlflowClient

client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

apples_experiment = client.search_experiments(
    filter_string="tags.`project_name` = 'grocery-forecasting'"
)

print(vars(apples_experiment[0]))