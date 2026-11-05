import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
import lightgbm as lgb
import pickle

# ── file paths ─────────────────────────────────────────────────────────────────
SAVE_PATH            = "data_processed/"
ELMIRA_DATA_FILENAME = "Elmira_80_from_03_07_2024_09_00_00_to_11_10_2024_23_00_00"

# Buffalo training files — needed to fit scalers
TRAIN_DATA_FILENAME  = "Buffalo_80_from_06_20_2022_01_00_00_to_04_25_2024_19_00_00_no_outliers"

# ── features (same as all other scripts) ──────────────────────────────────────
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
timestamp_features = ["month", "day", "hour"]
target_columns     = ["mean_rsrp"]
feature_columns    = weather_features + timestamp_features + ["sensor_distance"]

# ── load sensor_distance from raw KPI files ────────────────────────────────────
_kpi_buffalo = pd.read_csv(
    os.path.expanduser("~/CBRSdata/kpi/KPI_Data_Buffalo.csv"),
    usecols=["serial", "sensor_distance"]
)
_kpi_buffalo["serial"] = _kpi_buffalo["serial"].astype(str)
BUFFALO_SENSOR_DISTANCE = _kpi_buffalo.groupby("serial")["sensor_distance"].mean().to_dict()


# ── load and prepare Buffalo training data ─────────────────────────────────────
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
    df["sensor_distance"] = df["serial"].map(BUFFALO_SENSOR_DISTANCE)
    return df


# ── load and prepare Elmira data ───────────────────────────────────────────────
def get_elmira_df(file_name):
    df = pd.read_parquet(SAVE_PATH + file_name + ".parquet")
    df["serial"] = df["serial"].astype(str)
    df = df.reset_index(drop=True)
    df["serial"] = df["serial"].astype("category")
    df["month"]          = df.datetime.dt.month.astype("category")
    df["day"]            = df.datetime.dt.day.astype("category")
    df["hour"]           = df.datetime.dt.hour.astype("category")
    df["is_it_raining"]  = (df["rainfall (mm)"] > 0).astype("category")
    df["is_it_snowing"]  = (df["snowfall (mm of water equivalent)"] > 0).astype("category")
    # sensor_distance already in Elmira parquet
    return df


# ── mean center RSRP per sensor ────────────────────────────────────────────────
def zero_mean(df, sensor_means=None):
    if sensor_means:
        for serial in df["serial"].unique():
            serial_mask = (df["serial"] == serial)
            df.loc[serial_mask, "mean_rsrp"] -= sensor_means[serial]
    else:
        sensor_means = {}
        for serial in df["serial"].unique():
            serial_mask = (df["serial"] == serial)
            sensor_means[serial] = df.loc[serial_mask, "mean_rsrp"].mean()
            df.loc[serial_mask, "mean_rsrp"] -= sensor_means[serial]
    return df, sensor_means


# ── scale and encode ───────────────────────────────────────────────────────────
def scale_encode_data(features, targets,
                      features_scaler=None,
                      features_encoder=None,
                      targets_scaler=None,
                      feature_range=(0.25, 0.75)):
    num_cols = features.select_dtypes(include="number").columns
    cat_cols = features.select_dtypes(include="category").columns

    if features_scaler is None:
        features_scaler = MinMaxScaler(feature_range)
        features_scaler.fit(features[num_cols])
    if features_encoder is None:
        features_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        features_encoder.fit(features[cat_cols])
    if targets_scaler is None:
        targets_scaler = MinMaxScaler(feature_range)
        targets_scaler.fit(targets)

    X = np.hstack([
        features_scaler.transform(features[num_cols]),
        features_encoder.transform(features[cat_cols])
    ])
    y = targets_scaler.transform(targets).ravel()

    return X, y, features_scaler, features_encoder, targets_scaler


# ── evaluate one model on Elmira ──────────────────────────────────────────────
def evaluate(model_name, y_true, y_pred, test_serials, test_df):
    overall_mae = mean_absolute_error(y_true, y_pred)
    print(f"\n{model_name} — Overall MAE: {overall_mae:.4f} dB")
    print("MAE per sensor:")
    for serial in sorted(test_df["serial"].unique()):
        mask = (test_serials == serial)
        sensor_mae = mean_absolute_error(y_true[mask], y_pred[mask])
        print(f"  {serial}: {sensor_mae:.4f} dB")
    return overall_mae


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("Loading Buffalo training data (to fit scalers)...")
    train_df = get_buffalo_df(TRAIN_DATA_FILENAME)
    train_df, _ = zero_mean(train_df)

    print("Loading Elmira test data...")
    elmira_df = get_elmira_df(ELMIRA_DATA_FILENAME)
    elmira_df, elmira_sensor_means = zero_mean(elmira_df)

    print("Fitting scalers on Buffalo training data...")
    X_train, y_train, feat_scaler, feat_encoder, tgt_scaler = scale_encode_data(
        train_df[feature_columns], train_df[target_columns])

    print("Encoding Elmira data with Buffalo scalers...")
    X_elmira, y_elmira, _, _, _ = scale_encode_data(
        elmira_df[feature_columns], elmira_df[target_columns],
        feat_scaler, feat_encoder, tgt_scaler)

    # fill any NaN from sensor_distance mapping
    col_means = np.nanmean(X_train, axis=0)
    X_elmira = np.where(np.isnan(X_elmira), col_means, X_elmira)

    print(f"\nElmira test set size: {X_elmira.shape}")

    y_true = tgt_scaler.inverse_transform(y_elmira.reshape(-1, 1)).ravel()
    test_serials = elmira_df["serial"].values
    results = {}

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("\nTraining Random Forest on Buffalo, evaluating on Elmira...")
    rf = RandomForestRegressor(
        n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = tgt_scaler.inverse_transform(
        rf.predict(X_elmira).reshape(-1, 1)).ravel()
    results["Random Forest"] = evaluate("Random Forest", y_true, y_pred,
                                         test_serials, elmira_df)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("\nTraining XGBoost on Buffalo, evaluating on Elmira...")

    # need val set for early stopping
    from sklearn.model_selection import train_test_split
    X_tr, X_v, y_tr, y_v = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42)

    xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric="mae"
    )
    xgb.fit(X_tr, y_tr,
            eval_set=[(X_v, y_v)],
            verbose=100)
    y_pred = tgt_scaler.inverse_transform(
        xgb.predict(X_elmira).reshape(-1, 1)).ravel()
    results["XGBoost"] = evaluate("XGBoost", y_true, y_pred,
                                   test_serials, elmira_df)

    # ── LinearSVR ─────────────────────────────────────────────────────────────
    print("\nTraining LinearSVR on Buffalo, evaluating on Elmira...")
    X_train_clean = np.where(np.isnan(X_train), col_means, X_train)
    svr = LinearSVR(C=1.0, epsilon=0.01, max_iter=5000, random_state=42)
    svr.fit(X_train_clean, y_train)
    y_pred = tgt_scaler.inverse_transform(
        svr.predict(X_elmira).reshape(-1, 1)).ravel()
    results["LinearSVR"] = evaluate("LinearSVR", y_true, y_pred,
                                     test_serials, elmira_df)

    # ── LightGBM ──────────────────────────────────────────────────────────────
    print("\nTraining LightGBM on Buffalo, evaluating on Elmira...")
    X_v_lgb = X_train[-10000:]
    y_v_lgb = y_train[-10000:]
    X_tr_lgb = X_train[:-10000]
    y_tr_lgb = y_train[:-10000]

    dtrain = lgb.Dataset(X_tr_lgb, label=y_tr_lgb, free_raw_data=False)
    dval   = lgb.Dataset(X_v_lgb,  label=y_v_lgb,
                         reference=dtrain, free_raw_data=False)

    params = {
        "objective":        "regression_l1",
        "metric":           "mae",
        "learning_rate":    0.01,
        "num_leaves":       127,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "verbose":          -1,
        "n_jobs":           -1,
        "seed":             42
    }
    lgb_model = lgb.train(
        params, dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    y_pred = tgt_scaler.inverse_transform(
        lgb_model.predict(X_elmira).reshape(-1, 1)).ravel()
    results["LightGBM"] = evaluate("LightGBM", y_true, y_pred,
                                    test_serials, elmira_df)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(f"{'Model':<20} {'Buffalo MAE':>12} {'Elmira MAE':>12}")
    print("-"*50)
    print(f"{'Neural Network':<20} {'1.83 dB':>12} {'2.13 dB':>12}")
    print(f"{'XGBoost':<20} {'2.08 dB':>12} {results['XGBoost']:.4f} dB")
    print(f"{'LightGBM':<20} {'2.16 dB':>12} {results['LightGBM']:.4f} dB")
    print(f"{'LinearSVR':<20} {'2.47 dB':>12} {results['LinearSVR']:.4f} dB")
    print(f"{'Random Forest':<20} {'2.50 dB':>12} {results['Random Forest']:.4f} dB")