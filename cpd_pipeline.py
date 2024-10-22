import logging
import pandas as pd
import os
from os import path
from datetime import datetime
import argparse
from utils import (
    loader,
    load_data,
    preprocess_df_distribution,
    forecast_3_month_ahead,
    forecast_3_month_ahead_NA,
    adjust_ratio,
    combine_forecast,
    cpd_forecast,
    save_data
)

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

CURRENT_DIR = path.dirname(path.abspath(__file__))
SQL_DIR = path.join(CURRENT_DIR, "sql")

def main(test_window, forecast_window):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # initialize redshift loader
    rs = loader()
    logger.info("Start Monaco Distribution Pipeline")

    # Load time series of monaco distribution data
    df_distribution = load_data(
        sql_path=path.join(SQL_DIR, 'monaco_distribution.sql'), 
        loader=rs)
    
    # Load cpd per (channel, segment) data
    df_cpd = load_data(
        sql_path=path.join(SQL_DIR, 'cpd_segment_channel.sql'), 
        loader=rs)

    # Preprocess df_distribution
    logger.info("Preprocess Data")
    df_distribution_pre = preprocess_df_distribution(df_distribution)

    # Fit SARIMA model with the opimal order and seasonal orders for each segment
    # and forecast 3 months ahead for each segment
    logger.info("Fit Model and Forecast 3 Months Ahead")
    model_results, forecast_df, mae_dict = forecast_3_month_ahead(df_distribution_pre, test_window, forecast_window) 
    adj_forecast_df = adjust_ratio(forecast_df)

    # Fit SARIMAX model with the opimal order and seasonal orders for 'NA'
    model_results_NA, forecast_df_NA, mae_dict_NA = forecast_3_month_ahead_NA(df_distribution_pre, test_window, forecast_window)
    final_forecast = combine_forecast(forecast_df_NA, adj_forecast_df)

    # Save MAE from training/prediction data locally
    logger.info("Saving MAE from training/prediction data locally.")
    mae_dict.update(mae_dict_NA)
    #mae_dict_df = pd.DataFrame([mae_dict], columns=mae_dict.keys())
    local_output_path = "./outputs/{TIMESTAMP}/mae_dict.csv".format(TIMESTAMP = timestamp)
    save_data(pd.DataFrame(mae_dict), local_output_path)

    # Merge with cpd data to produce cpd per channel
    logger.info("Combine with CPD data to produce forecast of CPD per channel")
    cpd_forecast_df = cpd_forecast(final_forecast, df_cpd)
    
    earliest_trip_end_month = cpd_forecast_df['trip_end_month'].min()
    df_final = cpd_forecast_df.loc[cpd_forecast_df.trip_end_month == earliest_trip_end_month].reset_index(drop=True)

    logger.info("Saving forecast data locally.")
    local_output_path = "./outputs/{TIMESTAMP}/cpd_forecast_{TEST}_{FORECAST}.csv".format(TIMESTAMP = timestamp, 
                                                                                            TEST = test_window, 
                                                                                            FORECAST = forecast_window)
    save_data(df_final, local_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CPD Pipeline")
    parser.add_argument(
        "--test_window",
        type=int,
        default=3,
        help="Number of months behind from the current month to backtest the model"
    )
    parser.add_argument(
        "--forecast_window",
        type=int,
        default=3,
        help="Number of months ahead to forecast the model"
    )
    args = parser.parse_args()

    main(args.test_window, args.forecast_window)