from zenml import pipeline

from steps.model_training import svc_trainer
from steps.data_ingestion import get_data
from steps.data_cleaning import data_split  


@pipeline(enable_cache = False)
def train_pipeline(gamma: float=0.003):
    df = get_data()
    X_train, X_test, y_train, y_test = data_split(df)
    svc_trainer(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    train_pipeline()

    # X_train = run.steps["data_split"].outputs["X_train"][0].load()
    # print(f"X_train:{X_train}")
    