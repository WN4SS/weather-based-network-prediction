import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
import lightgbm as lgb

SAVE_PATH  = "data_processed/"
OUTPUT_DIR = "fold_predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_FILE   = "Buffalo_80_from_04_25_2024_20_00_00_to_05_31_2024_23_00_00"
ELMIRA_FILE = "Elmira_80_from_03_07_2024_09_00_00_to_11_10_2024_23_00_00"

FOLDS = {
    1: {
        "train": "Buffalo_80_from_06_20_2022_01_00_00_to_04_25_2024_19_00_00_no_outliers",
        "val":   "Buffalo_80_fold1_val_from_06_20_2022_01_00_00_to_10_29_2022_17_00_00_no_outliers"
    },
    2: {
        "train": "Buffalo_80_fold2_train_from_06_20_2022_01_00_00_to_10_29_2022_17_00_00_no_outliers",
        "val":   "Buffalo_80_fold2_val_from_10_29_2022_17_00_00_to_03_28_2023_00_00_00_no_outliers"
    },
    3: {
        "train": "Buffalo_80_fold3_train_from_06_20_2022_01_00_00_to_03_28_2023_00_00_00_no_outliers",
        "val":   "Buffalo_80_fold3_val_from_03_28_2023_00_00_00_to_07_17_2023_04_00_00_no_outliers"
    },
    4: {
        "train": "Buffalo_80_fold4_train_from_06_20_2022_01_00_00_to_07_17_2023_04_00_00_no_outliers",
        "val":   "Buffalo_80_fold4_val_from_07_17_2023_04_00_00_to_11_30_2023_11_00_00_no_outliers"
    },
    5: {
        "train": "Buffalo_80_fold5_train_from_06_20_2022_01_00_00_to_11_30_2023_11_00_00_no_outliers",
        "val":   "Buffalo_80_fold5_val_from_11_30_2023_11_00_00_to_04_25_2024_19_00_00_no_outliers"
    }
}

weather_features = [
    'temperature (degC)', 'relative_humidity (0-1)', 'absolute_humidity (g/m^3)',
    'wind_speed (m/s)', 'surface_pressure (Pa)', 'is_it_snowing',
    'snowfall (mm of water equivalent)', 'is_it_raining',
    'rainfall (mm)', 'total_precipitation (mm of water equivalent)'
]
timestamp_features = ["month", "day", "hour"]
target_columns     = ["mean_rsrp"]
feature_columns    = weather_features + timestamp_features + ["sensor_distance"]

_kpi_raw = pd.read_csv(
    os.path.expanduser("~/CBRSdata/kpi/KPI_Data_Buffalo.csv"),
    usecols=["serial", "sensor_distance"]
)
_kpi_raw["serial"] = _kpi_raw["serial"].astype(str)
SENSOR_DISTANCE = _kpi_raw.groupby("serial")["sensor_distance"].mean().to_dict()


def get_buffalo_df(file_name):
    df = pd.read_parquet(SAVE_PATH + file_name + ".parquet")
    df["serial"] = df["serial"].astype(str)
    df = df[df["serial"] != "21140007371"]
    df = df.reset_index(drop=True)
    df["serial"] = df["serial"].astype("category")
    df["month"]          = df.datetime.dt.month.astype("category")
    df["day"]            = df.datetime.dt.day.astype("category")
    df["hour"]           = df.datetime.dt.hour.astype("category")
    df["is_it_raining"]  = (df["rainfall (mm)"] > 0).astype("category")
    df["is_it_snowing"]  = (df["snowfall (mm of water equivalent)"] > 0).astype("category")
    df["sensor_distance"] = df["serial"].map(SENSOR_DISTANCE)
    return df


def get_elmira_df():
    df = pd.read_parquet(SAVE_PATH + ELMIRA_FILE + ".parquet")
    df["serial"] = df["serial"].astype(str)
    df = df.reset_index(drop=True)
    df["serial"] = df["serial"].astype("category")
    df["month"]          = df.datetime.dt.month.astype("category")
    df["day"]            = df.datetime.dt.day.astype("category")
    df["hour"]           = df.datetime.dt.hour.astype("category")
    df["is_it_raining"]  = (df["rainfall (mm)"] > 0).astype("category")
    df["is_it_snowing"]  = (df["snowfall (mm of water equivalent)"] > 0).astype("category")
    return df


def zero_mean(df, sensor_means=None):
    if sensor_means:
        for serial in df["serial"].unique():
            mask = (df["serial"] == serial)
            df.loc[mask, "mean_rsrp"] -= sensor_means[serial]
    else:
        sensor_means = {}
        for serial in df["serial"].unique():
            mask = (df["serial"] == serial)
            sensor_means[serial] = df.loc[mask, "mean_rsrp"].mean()
            df.loc[mask, "mean_rsrp"] -= sensor_means[serial]
    return df, sensor_means


def scale_encode(features, targets, fs=None, fe=None, ts=None,
                 feature_range=(0.25, 0.75)):
    num_cols = features.select_dtypes(include="number").columns
    cat_cols = features.select_dtypes(include="category").columns
    if fs is None:
        fs = MinMaxScaler(feature_range); fs.fit(features[num_cols])
    if fe is None:
        fe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        fe.fit(features[cat_cols])
    if ts is None:
        ts = MinMaxScaler(feature_range); ts.fit(targets)
    X = np.hstack([fs.transform(features[num_cols]),
                   fe.transform(features[cat_cols])])
    y = ts.transform(targets).ravel()
    return X, y, fs, fe, ts


def make_output_df(datetimes, serials, y_true_centered,
                   y_pred_centered, sensor_means, ts):
    y_true = ts.inverse_transform(
        y_true_centered.reshape(-1,1)).ravel()
    y_pred = ts.inverse_transform(
        y_pred_centered.reshape(-1,1)).ravel()
    y_true_real = np.zeros_like(y_true)
    y_pred_real = np.zeros_like(y_pred)
    for serial, mean_val in sensor_means.items():
        mask = (serials == serial)
        y_true_real[mask] = y_true[mask] + mean_val
        y_pred_real[mask] = y_pred[mask] + mean_val
    return pd.DataFrame({
        "datetime":           datetimes,
        "serial":             serials,
        "true_rsrp_dBm":      y_true_real,
        "pred_rsrp_dBm":      y_pred_real,
        "absolute_error_dB":  np.abs(y_true_real - y_pred_real)
    })


MODELS = {
    "RandomForest": lambda: RandomForestRegressor(
        n_estimators=100, n_jobs=-1, random_state=42),
    "XGBoost": lambda: XGBRegressor(
        n_estimators=1000, learning_rate=0.01, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        n_jobs=-1, random_state=42, eval_metric="mae"),
    "LinearSVR": lambda: LinearSVR(
        C=1.0, epsilon=0.01, max_iter=5000, random_state=42),
    "LightGBM": lambda: lgb.LGBMRegressor(
        objective="regression_l1", metric="mae",
        learning_rate=0.01, num_leaves=127,
        min_child_samples=20, feature_fraction=0.8,
        subsample=0.8, subsample_freq=5,
        n_estimators=1000, n_jobs=-1, random_state=42, verbose=-1)
}


if __name__ == "__main__":

    print("Loading Elmira data...")
    elmira_df = get_elmira_df()
    _, elmira_means = zero_mean(elmira_df.copy())

    for fold_num, fold in FOLDS.items():
        print(f"\n{'='*50}")
        print(f"FOLD {fold_num}")
        print(f"{'='*50}")

        train_df = get_buffalo_df(fold["train"])
        train_df, buf_means = zero_mean(train_df)

        test_df = get_buffalo_df(TEST_FILE)
        test_df, _ = zero_mean(test_df, buf_means)

        elmira_c = elmira_df.copy()
        elmira_c, _ = zero_mean(elmira_c, elmira_means)

        X_train, y_train, fs, fe, ts = scale_encode(
            train_df[feature_columns], train_df[target_columns])
        X_test, y_test, _, _, _ = scale_encode(
            test_df[feature_columns], test_df[target_columns], fs, fe, ts)
        X_elm, y_elm, _, _, _ = scale_encode(
            elmira_c[feature_columns], elmira_c[target_columns], fs, fe, ts)

        cm = np.nanmean(X_train, axis=0)
        X_train = np.where(np.isnan(X_train), cm, X_train)
        X_test  = np.where(np.isnan(X_test),  cm, X_test)
        X_elm   = np.where(np.isnan(X_elm),   cm, X_elm)

        test_dt  = test_df["datetime"].values
        test_ser = test_df["serial"].values
        elm_dt   = elmira_df["datetime"].values
        elm_ser  = elmira_df["serial"].values

        for model_name, model_fn in MODELS.items():
            print(f"  {model_name}...")
            model = model_fn()
            model.fit(X_train, y_train)

            # Buffalo
            out_buf = make_output_df(
                test_dt, test_ser, y_test,
                model.predict(X_test), buf_means, ts)
            mae_buf = out_buf["absolute_error_dB"].mean()
            fname = f"{OUTPUT_DIR}/{model_name}_Buffalo_fold{fold_num}.csv"
            out_buf.to_csv(fname, index=False)

            # Elmira
            out_elm = make_output_df(
                elm_dt, elm_ser, y_elm,
                model.predict(X_elm), elmira_means, ts)
            mae_elm = out_elm["absolute_error_dB"].mean()
            fname = f"{OUTPUT_DIR}/{model_name}_Elmira_fold{fold_num}.csv"
            out_elm.to_csv(fname, index=False)

            print(f"    Buffalo MAE: {mae_buf:.4f} | Elmira MAE: {mae_elm:.4f}")

    print("\nDone. Files saved to fold_predictions/")
    print(f"Total files: {len(os.listdir(OUTPUT_DIR))}")