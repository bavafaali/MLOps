import json

import pandas as pd
import numpy as np


from zenml import step, pipeline
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer

from steps.model_training import svc_trainer
from steps.data_ingestion import get_data
from steps.data_cleaning import data_split  
from utils import data_for_inference

@step
def deployment_trigger (
    current_accuracy: float,
    threshold_accuracy: float = 0.8
) -> bool:
    
    """Implements a simple model deployment trigger that looks at the
    input model accuracy and decides if it is good enough to deploy"""

    return current_accuracy > threshold_accuracy



@step
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print(f"existing_service:{existing_services}")
    print(f"type of existing_service: {type(existing_services)}")
    return existing_services[0]


@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> list:
    """Run an inference request against a prediction service."""

    service.start(timeout=10) # should be a NOP if already started

    data_dict = json.loads(data)
    data_dict.pop("columns")
    data_dict.pop("index")

    columns_for_df = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
        ]

    df = pd.DataFrame(data_dict["data"], columns=columns_for_df)
    data_array = df.to_numpy(dtype=float)

    prediction = service.predict(data_array)
    print(f"prediction:{prediction}")
    
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()

    return prediction


@pipeline(enable_cache=False)
def continuous_deployment_pipeline(
    min_accuracy: float = 0.9,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = get_data()
    X_train, X_test, y_train, y_test = data_split(df)
    model, acc = svc_trainer(X_train, X_test, y_train, y_test)
    
    deployment_decision = deployment_trigger(
        current_accuracy = acc, 
        threshold_accuracy = min_accuracy
        )
    
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )


@pipeline(enable_cache=False)
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    
    batch_data = data_for_inference()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    predictor(service=model_deployment_service, data=batch_data)
