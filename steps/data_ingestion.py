import logging
from typing import Annotated

import pandas as pd
from sklearn.datasets import load_iris

from zenml import step


class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def get_data(self) -> pd.DataFrame:
        
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        df['species'] = df['target'].map(dict(enumerate(iris.target_names
                                                        )))
        return df

@step
def get_data() -> Annotated[pd.DataFrame, "df"]:
    """
    Args:
        None
    Returns:
        df: pd.DataFrame
    """
    try:
        ingest_data = IngestData()
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(e)
        raise e
