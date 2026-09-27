import logging

from steps.data_ingestion import IngestData


def data_for_inference():
    
    try:
        ingest_data = IngestData()
        df = ingest_data.get_data()
        df = df.sample(n=100)
        df.drop(["target", "species"], axis=1, inplace=True)
        result = df.to_json(orient="split")
        return result
    
    except Exception as e:
    
        logging.error(e)
        raise e