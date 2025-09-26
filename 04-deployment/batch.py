import pandas as pd
import mlflow
import uuid
import sys
import argparse

categorical = ['PULocationID','DOLocationID']
new_features = ['PU_DO']    
numerical = ['trip_distance']
target = 'duration'
model_name = "NYC_taxi"

def get_df(parquet_file_path):
    df = pd.read_parquet(parquet_file_path)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df["duration"].dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <=60)]
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']   # creating a new feature
    return df

def load_model():
    client = mlflow.tracking.MlflowClient(tracking_uri="http://localhost:5000")
    latest_version_info = client.get_latest_versions(name=model_name, stages=["Production"])
    model_uri = latest_version_info[0].source
    version = latest_version_info[0].version
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    return model, version

def create_uuid(n):
    return [str(uuid.uuid4()) for _ in range(n)]


def batch_predict(input_file, output_file):
    print("loading model...")
    model,version = load_model()
    print("Reading data...")
    df = get_df(input_file)
    dicts = df[new_features + numerical].to_dict(orient='records')
    df_results = pd.DataFrame()
    df_results['ride_id'] = create_uuid(len(df))
    df_results['tpep_pickup_datetime'] = df['tpep_pickup_datetime']
    df_results['PULocationID'] = df['PULocationID']
    df_results['DOLocationID'] = df['DOLocationID']
    df_results['actual_duration'] = df['duration']
    print("Predicting...")
    df_results['predicted_duration'] = model.predict(dicts)
    df_results['diff'] = df_results['actual_duration'] - df_results['predicted_duration']
    df_results['model_version'] = model_name + str(version)
    df.to_parquet(
        output_file,
        engine='pyarrow',
        index=False
    )
    print(f"predictions saved to {output_file}")

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--taxi_type', default='yellow', type=str, help='yellow or green')
    parser.add_argument('--year', default=2023, type=int, help='year of the data')
    parser.add_argument('--month', default=1, type=int, help='month of the data')
    args = parser.parse_args()
    taxi_type = args.taxi_type
    year = args.year 
    month = args.month  
    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"gs://mlflow_vaibhav/predictions/{taxi_type}/{year:04d}-{month:02d}.parquet"
    batch_predict(input_file, output_file)

if __name__ == "__main__":
    run()
