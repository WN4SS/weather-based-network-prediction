import os
import argparse

from trainer import model_pipeline
import torch.multiprocessing as mp

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import pandas as pd
import numpy as np

from tqdm import tqdm
from copy import deepcopy

DEVICE = "cpu"
DTYPE = np.float32

os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["WANDB_SILENT"] = "false"

SAVE_PATH = "data_processed/"
TRAIN_DATA_FILENAME = "Buffalo_80_from_06_20_2022_01_00_00_to_04_25_2024_19_00_00_no_outliers"
VAL_DATA_FILENAME   = "Buffalo_80_from_04_25_2024_20_00_00_to_05_31_2024_23_00_00_no_outliers"
TEST_DATA_FILENAME  = "Buffalo_80_from_04_25_2024_20_00_00_to_05_31_2024_23_00_00"

PROJECT_NAME = "wn4ss-cbrs-weather-project-12"
EXPERIMENT_NUMBER = 4

EXPERIMENTS_BATCH_SIZE = 5
N_EPOCHS    = 100
LR          = 1e-2

N_SPLITS = 5

BATCH_SIZE_TRIALS  = [
    64
    # 0
    # 1024
]

N_NODES_LAYER_TRIALS = [
    # [512, 512, 1]
    # [256, 512, 512, 256, 1]
    [128, 128, 1]
]

LAYER_ACTIVATIONS_TRIALS = [
    ["lrelu", "lrelu", "relu"]
    # ["tanh", "tanh", "tanh"]
    # ["lrelu", "lrelu", "lrelu", "lrelu", "relu"]
]

weather_features = [
    'temperature (degC)',
    'relative_humidity (0-1)',
    'absolute_humidity (g/m^3)',
    'wind_speed (m/s)',
    'surface_pressure (Pa)',
    'is_it_snowing',
    'snowfall (mm of water equivalent)',
    'is_it_raining',
    'rainfall (mm)',
    'total_precipitation (mm of water equivalent)'
]

timestamp_features = [
    "month",
    "day",
    "hour"
]

signal_features = [
    "serving_enb_id"
]

target_columns = [
    "mean_rsrp"
]

assert len(N_NODES_LAYER_TRIALS) == len(LAYER_ACTIVATIONS_TRIALS)

def scale_encode_data(features: pd.DataFrame,
                      targets: pd.DataFrame,
                      features_scaler: MinMaxScaler = None,
                      features_encoder: OneHotEncoder = None,
                      targets_scaler: MinMaxScaler = None,
                      feature_range: tuple = (0, 1)):
    features_numerical_columns = features.select_dtypes(include="number").columns
    features_categorical_columns = features.select_dtypes(include="category").columns

    if features_scaler is None:
        features_scaler = MinMaxScaler(feature_range)
        features_scaler.fit(features[features_numerical_columns])

    if features_encoder is None:
        features_encoder = OneHotEncoder()
        features_encoder.fit(features[features_categorical_columns])
    
    if targets_scaler is None:
        targets_scaler = MinMaxScaler(feature_range)
        targets_scaler.fit(targets)

    features_scaled_encoded = pd.concat([
        pd.DataFrame(features_scaler.transform(features[features_numerical_columns]),
                     columns=features_numerical_columns,
                     index=features.index),
        pd.DataFrame(features_encoder.transform(features[features_categorical_columns]).toarray(),
                     columns=features_encoder.get_feature_names_out(),
                     index=features.index)
        ],
        axis=1)
    
    targets_scaled = pd.DataFrame(targets_scaler.transform(targets),
                                  columns=targets.columns,
                                  index=targets.index)

    X, y = features_scaled_encoded, targets_scaled

    return X, y, features_scaler, features_encoder, targets_scaler

def get_df(file_name: str):
    df = pd.read_parquet(SAVE_PATH + file_name + ".parquet")

    df["serial"] = df["serial"].astype(str)
    df = df[df["serial"] != "21140007371"]
    df = df.reset_index(drop=True)
    df["serial"] = df["serial"].astype("category")
    
    # df["serving_enb_id"] = df["serving_enb_id"].astype("category")
    df["month"] = df.datetime.dt.month.astype("category")
    df["day"] = df.datetime.dt.day.astype("category")
    df["hour"] = df.datetime.dt.hour.astype("category")
    df["is_it_raining"] = (df["rainfall (mm)"] > 0).astype("category")
    df["is_it_snowing"] = (df["snowfall (mm of water equivalent)"] > 0).astype("category")
    
    # For training only worst performing sensor
    # df = df[df["serial"] == "TLR001052420"]

    return df

def get_feature_columns(version: int):
    feature_columns = weather_features + timestamp_features

    if version == 7:
        feature_columns += ["sensor_distance"]
    elif version == 8:
        feature_columns += signal_features + ["serial"]

    return feature_columns

def zero_mean(df: pd.DataFrame,
              sensor_means: dict = None) -> tuple[pd.DataFrame, dict]:
    if sensor_means:
        for serial in df["serial"].unique():
            serial_mask = (df["serial"] == serial)
            df.loc[serial_mask, "mean_rsrp"] -= sensor_means[serial]
    else:
        sensor_means = dict()
        for serial in df["serial"].unique():
            serial_mask = (df["serial"] == serial)
            sensor_means[serial] = df.loc[serial_mask, "mean_rsrp"].mean()
            df.loc[serial_mask, "mean_rsrp"] -= sensor_means[serial]
    return df, sensor_means

def create_worker(semaphore: mp.Semaphore, # type: ignore
                  queue: mp.Queue,
                  project_name: str,
                  run_name: str,
                  config: dict,
                  shared_data: dict,
                  targets_scaler: MinMaxScaler) -> None:
    with semaphore:
        try:
            # Start progress bar
            queue.put((run_name, 0))
            model_pipeline(project_name,
                           run_name,
                           config,
                           shared_data,
                           targets_scaler,
                           queue)
        finally:
        # Signal completion
            queue.put((run_name, None))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that processes command-line arguments.")
    parser.add_argument("-v", "--version", type=int, default=5, help="An optional argument")
    
    args = parser.parse_args()
    print(f"Starting version {args.version}")

    mp.set_start_method('spawn')
    with mp.Manager() as manager:
        train_df = get_df(TRAIN_DATA_FILENAME)
        train_df, sensor_means = zero_mean(train_df)
        print(sensor_means)

        val_df = get_df(VAL_DATA_FILENAME)
        val_df, _ = zero_mean(val_df, sensor_means)

        test_df = get_df(TEST_DATA_FILENAME)
        test_df, _ = zero_mean(test_df, sensor_means)

        feature_columns = get_feature_columns(args.version)
        features_numerical_columns = train_df[feature_columns].select_dtypes(include="number").columns
        features_categorical_columns = train_df[feature_columns].select_dtypes(include="category").columns

        X_train, y_train, features_scaler, features_encoder, targets_scaler = scale_encode_data(train_df[feature_columns],
                                                                                                train_df[target_columns],
                                                                                                feature_range=(0.25, 0.75))
        X_val, y_val, _, _, _ = scale_encode_data(val_df[feature_columns],
                                                  val_df[target_columns],
                                                  features_scaler,
                                                  features_encoder,
                                                  targets_scaler)
        X_test, y_test, _, _, _ = scale_encode_data(test_df[feature_columns],
                                                    test_df[target_columns],
                                                    features_scaler,
                                                    features_encoder,
                                                    targets_scaler)

        # Created shared dict for sub-processes
        shared_data = manager.dict()
        shared_data['X_train'] = X_train.to_numpy(copy=False,
                                                  dtype=DTYPE)
        shared_data['y_train'] = y_train.to_numpy(copy=False,
                                                  dtype=DTYPE)
        shared_data['X_val'] = X_val.to_numpy(copy=False,
                                              dtype=DTYPE)
        shared_data['y_val'] = y_val.to_numpy(copy=False,
                                              dtype=DTYPE)
        shared_data['X_test'] = X_test.to_numpy(copy=False,
                                                dtype=DTYPE)
        shared_data['y_test'] = y_test.to_numpy(copy=False,
                                                dtype=DTYPE)
        shared_data["test_datetime"] = test_df["datetime"]
        shared_data["test_serial"] = test_df["serial"]
        shared_data["val_datetime"] = val_df["datetime"]
        shared_data["val_serial"] = val_df["serial"]
        shared_data["sensor_means"] = sensor_means

        del train_df, val_df, test_df
        
        experiments = []
        semaphore = mp.Semaphore(EXPERIMENTS_BATCH_SIZE) # for limiting concurrency
        pbar_dict = {} # progress bar is displayed only from main process
        queue = mp.Queue() # for sending pbar updates from sub to main process
        
        for split_number in range(1, N_SPLITS+1):
            model_number = 0
            for (N_NODES_LAYER, LAYER_ACTIVATIONS) in zip(N_NODES_LAYER_TRIALS, LAYER_ACTIVATIONS_TRIALS):
                for BATCH_SIZE in BATCH_SIZE_TRIALS:
                    model_number += 1

                    run_name = f"regression_experiment{EXPERIMENT_NUMBER}_version{args.version}_model{model_number}_split{split_number}"
                    experiment = mp.Process(target=create_worker,
                                            args=(
                                                semaphore,
                                                queue,
                                                PROJECT_NAME,
                                                run_name, 
                                                {
                                                    "EXPERIMENT": EXPERIMENT_NUMBER,
                                                    "VERSION": args.version,
                                                    "LR": LR,
                                                    "N_EPOCHS": N_EPOCHS,
                                                    "BATCH_SIZE": BATCH_SIZE,
                                                    "N_NODES_LAYER" : N_NODES_LAYER,
                                                    "LAYER_ACTIVATIONS" : LAYER_ACTIVATIONS,
                                                    "NUMERICAL_FEATURES": features_numerical_columns,
                                                    "CATEGORICAL_FEATURES": features_categorical_columns,
                                                    "TARGETS": target_columns,
                                                    "TRAIN_DATASET": TRAIN_DATA_FILENAME,
                                                    "VAL_DATASET": VAL_DATA_FILENAME,
                                                    "TEST_DATASET": TEST_DATA_FILENAME
                                                },
                                                shared_data,
                                                deepcopy(targets_scaler))
                                            )
                    experiments.append(experiment)

        completed = 0
        pbar_dict["total"] = tqdm(total=len(experiments),
                                  desc="total")

        [experiment.start() for experiment in experiments[:EXPERIMENTS_BATCH_SIZE]]

        while completed < len(experiments):
            run_name, progress = queue.get()
            if progress is None:
                completed += 1
                pbar_dict["total"].update(1)
                if run_name in pbar_dict:
                    pbar_dict[run_name].close()
                if len(experiments) > completed + EXPERIMENTS_BATCH_SIZE - 1:
                    experiments[completed + EXPERIMENTS_BATCH_SIZE - 1].start()
            elif progress == 0:
                pbar_dict[run_name] = tqdm(total=N_EPOCHS,
                                           desc=run_name,
                                           leave=False)
            elif progress == 1:
                if run_name in pbar_dict:
                    pbar_dict[run_name].update(1)

        for experiment in experiments:
            experiment.join()
            experiment.terminate()