"""
MLflowの基本的な使い方

・パラメータ、メトリック、モデルを記録する方法
・MLflow Fluent API の基礎
・ログ中にモデルを登録する方法
・MLflow UIでモデルに移動する方法
・推論のためにログに記録されたモデルをロードする方法

事前にトラッキングサーバーを起動する
ポート番号は使用されていないポートであれば任意の番号でいい
$ mlflow server --host 127.0.0.1 --port 8080
"""

import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# トラッキングサーバーURIを設定する
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# 新しいMLflowの実験プロジェクトを作成する
mlflow.set_experiment("MLflow Quickstart")

X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888
}

lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# モデルとメタデータをMLflowに記録する

# MLflow runを開始する
with mlflow.start_run():
    # ハイパーパラメータのログ
    mlflow.log_params(params)

    # loss metricのログ
    mlflow.log_metric("accuracy", accuracy)

    # 実行した内容についての覚え書き
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # モデルの署名を推測する
    signature = infer_signature(X_train, lr.predict(X_train))

    # モデルのログ
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart"
    )

# モデルをPython関数(pyfunc)として読み込み、推論に使用する
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

# 読み込まれたモデルを用いて新しく予測に使うことができる
predictions = loaded_model.predict(X_test)

iris_feature_names = datasets.load_iris().feature_names

result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

print(result[:4])