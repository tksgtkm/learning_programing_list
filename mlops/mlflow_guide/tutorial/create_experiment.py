from mlflow import MlflowClient

client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

experiment_description = (
    "これは食料品需要プロジェクトです"
    "この実験はりんごのモデルを含みます"
)

experiment_tags = {
    "project_name": "grocery-forecasting",
    "store_project": "produce",
    "team": "stores-ml",
    "project_quater": "Q3-2024",
    "mlflow.note.content": experiment_description
}

produce_apples_experiment = client.create_experiment(
    name="Apple_Models", tags=experiment_tags
)
