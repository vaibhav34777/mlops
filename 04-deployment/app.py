from fastapi import FastAPI
import pickle

with open('models/lin_reg.bin','rb') as f_in:
    dv, lr = pickle.load(f_in)

def preprocess(features):
    features['PU_DO'] = f"{str(features['PULocationID'])}_{str(features['DOLocationID'])}"
    features['trip_distance'] = features['trip_distance']
    return features

def predict(features):
    features = preprocess(features)
    X = dv.transform(features)
    y_pred = lr.predict(X)
    return y_pred[0]    

app = FastAPI()

@app.post('/predict')
def predict_endpoint(features: dict):
    return predict(features)



