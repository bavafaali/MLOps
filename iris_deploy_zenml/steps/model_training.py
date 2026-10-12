from typing import Tuple, Annotated

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC


import mlflow
from zenml import step
from zenml.client import Client

client = Client()
experiment_tracker = client.active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def svc_trainer(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    gamma: float = 0.001
    ) -> Tuple[
    Annotated[ClassifierMixin, "trained_model"],
    Annotated[float, "training_acc"]
    ]:
    
    """Train a sklearn SVC classifier and log to MLflow."""
    
    mlflow.sklearn.autolog()
    
    model = SVC(gamma=gamma)
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    test_acc = model.score(X_test.to_numpy(), y_test.to_numpy())
    
    mlflow.log_metric("accuracy", test_acc)
    print(f"Train accuracy: {test_acc}")

    return model, test_acc


# if __name__ == "__main__":
#     df = get_data()
#     X_train, X_test, y_train, y_test = data_split(df)
#     _, acc = svc_trainer(X_train, X_test, y_train, y_test)
#     print(f"acc type:{type(acc)}")