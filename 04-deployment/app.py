from fastapi import FastAPI
import mlflow
from mlflow.client import MlflowClient

mlflow_client = MlflowClient(tracking_uri="http://34.100.166.236:5000")
model_name = "NYC_taxi"
latest_version_info = mlflow_client.get_latest_versions(name=model_name, stages=["Production"])
model_uri = latest_version_info[0].source

model = mlflow.pyfunc.load_model(model_uri=model_uri)

def preprocess(features):
    Features = {}
    Features['PU_DO'] = f"{str(features['PULocationID'])}_{str(features['DOLocationID'])}"
    Features['trip_distance'] = features['trip_distance']
    return Features

def predict(features):
    features = preprocess(features)
    y_pred = model.predict([features])
    return y_pred[0]    

app = FastAPI()

@app.post('/predict')
def predict_endpoint(features: dict):
    return predict(features)



