import pandas as pd
import pickle
import os, tempfile
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb
import mlflow

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment("duration-prediction")


categorical = ['PULocationID','DOLocationID']
numerical = ['trip_distance']
target = 'duration'
new_feature = ['PU_DO']
def read_df(year,month):
    df = pd.read_parquet(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df["duration"].dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <=60)]
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']   # creating a new feature
    return df

def create_Xy(df,dv=None):
    dicts = df[categorical + numerical + new_feature].to_dict(orient='records')
    if dv is None:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    y = df[target].values
    return X, y, dv

def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)
        params = {
            "learning_rate": 0.6856143053942376,
            "max_depth": 7,
            "min_child_weight": 4.434221562748838,
            "objective": "reg:squarederror",
            "reg_alpha": 0.16562147689397697,
            "reg_lambda": 0.06268756235943868,
            "seed": 42
        }
        mlflow.log_params(params)
        mlflow.set_tag("model", "xgboost")
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=20
        )
        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric('rmse', rmse)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "xgboost-model")
            dv_path = os.path.join(temp_dir, "preprocessor.b")
            mlflow.xgboost.save_model(booster,model_path)
            with open(dv_path, "wb") as f_out:
                pickle.dump(dv, f_out)
            mlflow.log_artifact(model_path, artifact_path="model")
            mlflow.log_artifact(dv_path, artifact_path="preprocessor")
        run_id = run.info.run_id
        return rmse, run_id

def main(args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', default=2021, type=int)
    parser.add_argument('--month', default=1, type=int)
    if args is None:
        args = []
    args = parser.parse_args(args)
    train_df = read_df(args.year, args.month)
    train_df = train_df.sample(n=10000, random_state=42).copy()  # using a subset of data for training
    month = args.month+1 if args.month<12 else 1
    year = args.year if month<12 else args.year+1
    val_df = read_df(year, month)
    val_df = val_df.sample(n=10000, random_state=42).copy()  # using a subset of data for validation
    X_train, y_train, dv = create_Xy(train_df)
    X_val, y_val, _ = create_Xy(val_df, dv)
    rmse, run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f'RMSE: {rmse}')
    print(f'Run ID: {run_id}')

if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if len(args)==0:
        args = ['--year','2021','--month','1']
    main(args)