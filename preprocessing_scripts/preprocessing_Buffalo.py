import pandas as pd
import argparse
import ast

WITHIN_DISTRIBUTION_IDS = {
    "21140007371": [1.0, 2.0],
    "TLR001021183": [2.0, 1.0, 3.0],
    "TLR001023697": [2.0],
    "TLR001023730": [1.0, 2.0],
    "TLR001024549": [1.0, 2.0, 4.0],
    "TLR001024748": [2.0, 1.0],
    "TLR001048674": [1.0, 2.0, 3.0],
    "TLR001052247": [2.0, 1.0],
    "TLR001052281": [1.0, 2.0],
    "TLR001052293": [2.0, 1.0],
    "TLR001052420": [1.0, 2.0],
    "TLR001052985": [1.0, 2.0, 3.0],
    "TLR001061822": [2.0, 1.0, 3.0],
    "TLR001062295": [3.0, 2.0, 1.0],
}

def get_df(filepath: str) -> pd.DataFrame:
    if "csv" in filepath:
        df = pd.read_csv(filepath,
                         na_values="UnKnown")
    elif "parquet" in filepath:
        df = pd.read_parquet(filepath)
    return df

def set_datetime_column(df: pd.DataFrame,
                        datetime_column: str) -> pd.DataFrame:
    df["datetime"] = pd.to_datetime(df[datetime_column])
    df.sort_values(by=["datetime"],
                   inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.drop(columns=[datetime_column], inplace=True)
    return df

def set_consistent_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(columns={"mean_AvgRsrp": "mean_rsrp",
                       "mean_AvgSinr": "mean_sinr",
                       "mean_AvgRsrq": "mean_rsrq",
                       "mean_AvgRssi": "mean_rssi"},
              inplace=True)
    return df

# def drop_na_values(df: pd.DataFrame,
#                    threshold: float) -> pd.DataFrame:    
#     df_na_columns = df.isna().sum()[df.isna().sum() > 0]
#     df_na = pd.DataFrame({"% missing values": (df_na_columns/df.shape[0]),
#                           "# missing values": df_na_columns})
#     df_high_na_columns = list(df_na[df_na["% missing values"] > threshold].index)
#     df.drop(columns=df_high_na_columns + ["distance"],
#             inplace=True)
#     return df

def drop_na_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(axis="columns", inplace=True)
    return df

def fill_serving_enb_id(df: pd.DataFrame) -> pd.DataFrame:
    for serial, data in df.groupby("serial"):
        df.loc[(df["serial"] == serial) & data["serving_enb_id"].isna(), "serving_enb_id"] = data["serving_enb_id"].mode().iloc[0]
    return df

def remove_outliers(df: pd.DataFrame,
                    selected_serving_enb_ids: dict,
                    lower_bound: float = 0.01,
                    upper_bound: float = 0.99) -> pd.DataFrame:
    for serial in df["serial"].unique():
        serial_df = df[df["serial"] == serial]
        for serving_enb_id in serial_df["serving_enb_id"].unique():
            if serving_enb_id not in selected_serving_enb_ids[serial]:
                df.drop(serial_df[serial_df["serving_enb_id"] == serving_enb_id].index,
                        inplace=True)
    
    for serial in df["serial"].unique():
        serial_mask = (df["serial"] == serial)
        
        serial_data = df.loc[serial_mask, "mean_rsrp"]

        less_than_lower_bound = (serial_data < serial_data.quantile(lower_bound))
        more_than_upper_bound = (serial_data > serial_data.quantile(upper_bound))

        outlier_mask = (less_than_lower_bound | more_than_upper_bound)
        inlier_mask = ~(outlier_mask & serial_mask)
        
        df = df[inlier_mask]
    return df

def reset_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df_categorical_columns = df.select_dtypes(include="object").columns
    for column in df_categorical_columns:
        df[column] = df[column].astype("category")
    return df

def slice_dataframe(df: pd.DataFrame,
                    first_timestamp: str,
                    last_timestamp: str) -> pd.DataFrame:
    if first_timestamp:
        df = df[df["datetime"] >= first_timestamp]
        print("Slicing forward from first_timestamp")
    if last_timestamp:
        df = df[df["datetime"] <= last_timestamp]
        print("Slicing backward from last_timestamp")
    return df

def get_min_time(df: pd.DataFrame) -> str:
    return df.datetime.min().strftime('%m_%d_%Y_%H_%M_%S')

def get_max_time(df: pd.DataFrame) -> str:
    return df.datetime.max().strftime('%m_%d_%Y_%H_%M_%S')

def split_lat_long(df: pd.DataFrame) -> pd.DataFrame:
    df["coordinates (lat,lon)"] = df["coordinates (lat,lon)"].apply(ast.literal_eval)
    df[["latitude", "longitude"]] = pd.DataFrame(df["coordinates (lat,lon)"].tolist(), index=df.index)
    df.drop(columns="coordinates (lat,lon)", inplace=True)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that processes command-line arguments.")
    parser.add_argument("--kpi_filepath",
                        type=str,
                        required=True,
                        help="Filepath of KPI dataset")
    parser.add_argument("--weather_filepath",
                        type=str,
                        required=True,
                        help="Filepath of Weather dataset")
    parser.add_argument("--remove_outliers",
                        type=str,
                        required=False,
                        help="Whether to remove outliers from KPI data (Not removed by default)")
    parser.add_argument("--first_timestamp",
                        type=str,
                        required=False,
                        help="First timestamp to slice data (optional)")
    parser.add_argument("--last_timestamp",
                        type=str,
                        required=False,
                        help="Last time to slice data (optional)")
    parser.add_argument("--savename",
                        type=str,
                        required=True,
                        help="Savename of output dataset")
    args = parser.parse_args()

    SAVENAME = args.savename

    # KPI Data
    # ========
    kpi_df = get_df(args.kpi_filepath)
    kpi_df = set_datetime_column(kpi_df, datetime_column="coll_time_round")
    kpi_df = set_consistent_feature_names(kpi_df)
    # kpi_df = drop_na_values(kpi_df, threshold=0.5)
    kpi_df = fill_serving_enb_id(kpi_df)
    if args.remove_outliers == "True":
        print("Removing outliers")
        kpi_df = remove_outliers(kpi_df, WITHIN_DISTRIBUTION_IDS)
    kpi_df = drop_na_columns(kpi_df)
    kpi_df = reset_dtypes(kpi_df)
    kpi_df = slice_dataframe(kpi_df, args.first_timestamp, args.last_timestamp)

    # kpi_df.to_parquet(f"data_processed/kpi/{args.savename}.parquet")
    # print(f"Saved - kpi_data_processed_from_{get_min_time(kpi_df)}_to_{get_max_time(kpi_df)}")

    # Weather Data
    # ============
    weather_df = get_df(args.weather_filepath)
    weather_df = set_datetime_column(weather_df, datetime_column="datetime (UTC)")
    weather_df = drop_na_columns(weather_df)
    weather_df = split_lat_long(weather_df) 
    weather_df = slice_dataframe(weather_df, args.first_timestamp, args.last_timestamp)

    # weather_df.to_parquet(f"data_processed/weather/{args.savename}.parquet")
    # print(f"Saved - weather_data_processed_from_{get_min_time(weather_df)}_to_{get_max_time(weather_df)}")

    # Combine
    # =======
    print(f"This data extends from {get_min_time(kpi_df)} to {get_max_time(kpi_df)}")
    df = pd.merge(weather_df,
                  kpi_df,
                  on='datetime')
    df.sort_values(by=["datetime", "serial"], inplace=True)
    df.reset_index(inplace=True, drop=True)

    SAVENAME += f"_from_{get_min_time(df)}_to_{get_max_time(df)}"
    if args.remove_outliers == "True":
        SAVENAME += "_no_outliers"

    df.to_parquet(f"data_processed/{SAVENAME}.parquet")
    print(f"Saved - weather_kpi_data_processed_from_{get_min_time(df)}_to_{get_max_time(df)}")