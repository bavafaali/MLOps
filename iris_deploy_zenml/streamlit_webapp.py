import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipeline.deployment_pipeline import prediction_service_loader
from run_deployment import main


def main():
    st.title("Classification of Iris dataset")

    whole_pipeline_image = Image.open("assets/pipeline.png")

    st.image(whole_pipeline_image, caption="Pipeline")
    st.markdown(
        """ 
    First the data is ingested and cleaned, then the model is trained, and evaluated. If data source changes or any hyperparameter values changes, deployment will be triggered, and the model is (re)trained; if the model meets minimum accuracy requirement, it will be deployed.
    """
    )

    st.markdown(
        """ 
    #### App and dataset
    This app is designed to predict the type of flowers from Iris dataset based on four features using Support Vector Classifier (about [Iris dataset](https://archive.ics.uci.edu/dataset/53/iris)).
    """
    )
        

    sepal_length = st.number_input("sepal length (cm)")
    sepal_width = st.number_input("sepal width (cm)")
    petal_length = st.number_input("petal length (cm)")
    petal_width = st.number_input("petal width (cm)")

    if st.button("Predict"):
        service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            run_main()

        df = pd.DataFrame(
            {
                "sepal length (cm)": [sepal_length],
                "sepal width (cm)": [sepal_width],
                "petal length (cm)": [petal_length],
                "petal width (cm)": [petal_width]
            }
        )
    
        pred = service.predict(df.to_numpy(dtype=float))
        st.success(f"The prediction is class:{pred}")
    
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown(
    " Most of the icons are from [flaticon](https://www.flaticon.com)."
    
    )
if __name__ == "__main__":
    main()
