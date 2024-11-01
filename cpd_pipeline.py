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
    forecast_n_month_ahead,
    forecast_n_month_ahead_NA,
    adjust_ratio,
    combine_forecast,
    update_paid_days,
    pdps_forecast_agg_weight,
    weighted_avg_distribution,
    seasonal_ratio,
    forecast_distribution_channel,
    cpd_forecast,
    save_data,
)

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

CURRENT_DIR = path.dirname(path.abspath(__file__))
SQL_DIR = path.join(CURRENT_DIR, "sql")


def main(end_month, test_window, forecast_window):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # initialize redshift loader
    rs = loader()
    logger.info("Start Monaco Distribution Pipeline")

    # Load time series of monaco distribution data
    df_distribution = load_data(
        sql_path=path.join(SQL_DIR, "monaco_distribution.sql"), loader=rs
    )

    # Load time series of monaco distribution by channel
    df_distribution_channel = load_data(
        sql_path=path.join(SQL_DIR, "monaco_distribution_channel.sql"), loader=rs
    )

    # Load cpd per (channel, segment) data
    df_cpd = load_data(
        sql_path=path.join(SQL_DIR, "cpd_segment_channel.sql"), loader=rs
    )

    # Load projected pdps data (NEED TO OPTIMIZE THIS LATER)
    # pdps_forecast = load_data(
    # sql_path=path.join(SQL_DIR, "paid_days_forecast.sql"), loader=rs
    # )

    # Preprocess df_distribution
    logger.info("Preprocess Data")
    df_distribution_pre, df_distribution_channel_pre = preprocess_df_distribution(
        df_distribution, df_distribution_channel, end_month
    )

    # Fit SARIMA model with the opimal order and seasonal orders for each segment
    # and forecast 3 months ahead for each segment
    logger.info("Fit Model and Forecast 12 Months Ahead")
    model_results, forecast_df, mae_dict = forecast_n_month_ahead(
        df_distribution_pre, test_window, forecast_window
    )
    adj_forecast_df = adjust_ratio(forecast_df)

    # Fit SARIMAX model with the opimal order and seasonal orders for 'NA'
    model_results_NA, forecast_df_NA, mae_dict_NA = forecast_n_month_ahead_NA(
        df_distribution_pre, test_window, forecast_window
    )
    final_forecast = combine_forecast(forecast_df_NA, adj_forecast_df)

    # Save MAE from training/prediction data locally
    logger.info("Saving MAE from training/prediction data locally.")
    mae_dict.update(mae_dict_NA)
    # mae_dict_df = pd.DataFrame([mae_dict], columns=mae_dict.keys())
    local_output_path = "./outputs/{TIMESTAMP}/mae_dict.csv".format(TIMESTAMP=timestamp)
    save_data(pd.DataFrame(mae_dict), local_output_path)

    # Compute projected paid days (NEED TO OPTIMIZE THIS LATER)
    # pdps_forecast.loc[
    # (pdps_forecast.projected_paid_days.isnull()),
    # "projected_paid_days",
    # ] = 0

    # for channel in pdps_forecast.channels.unique():
    # signup_month_start = pd.to_datetime("2024-11-01")
    # update_paid_days(signup_month_start, channel, pdps_forecast)
    pdps_forecast = pd.read_csv(os.path.join(CURRENT_DIR, "pdps_forecast.csv"))
    pdps_forecast.signup_month = pd.to_datetime(pdps_forecast.signup_month)
    pdps_forecast_agg = pdps_forecast_agg_weight(pdps_forecast)
    final_forecast_w = weighted_avg_distribution(final_forecast, pdps_forecast_agg)

    # Seasonal Ratio
    # |trip_end_month (forecast_month) | monaco_bin | ratio (ratio compared to the previous 3 month average)
    final_forecast_ratio = seasonal_ratio(final_forecast_w, df_distribution_pre)

    # Forecast distribution channel
    final_forecast_channel = forecast_distribution_channel(
        final_forecast_ratio, df_distribution_channel_pre
    )

    # Save final forecast channel data locally
    logger.info("Saving final forecast channel data locally.")
    local_output_path_channel = (
        "./outputs/{TIMESTAMP}/final_forecast_channel_{TEST}_{FORECAST}.csv".format(
            TIMESTAMP=timestamp, TEST=test_window, FORECAST=forecast_window
        )
    )
    save_data(final_forecast_channel, local_output_path_channel)

    # Merge with cpd data to produce cpd per channel
    logger.info("Combine with CPD data to produce forecast of CPD per channel")
    cpd_forecast_df = cpd_forecast(final_forecast_channel, df_cpd)

    earliest_trip_end_month = cpd_forecast_df["trip_end_month"].min()
    df_final = cpd_forecast_df.loc[
        cpd_forecast_df.trip_end_month == earliest_trip_end_month
    ].reset_index(drop=True)

    logger.info("Saving forecast data locally.")
    local_output_path = (
        "./outputs/{TIMESTAMP}/cpd_forecast_{TEST}_{FORECAST}.csv".format(
            TIMESTAMP=timestamp, TEST=test_window, FORECAST=forecast_window
        )
    )
    save_data(df_final, local_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CPD Pipeline")
    parser.add_argument(
        "--end_month",
        type=str,
        default="2024-10-01",
        help="End month of the observation period",
    )
    parser.add_argument(
        "--test_window",
        type=int,
        default=3,
        help="Number of months behind from the current month to backtest the model",
    )
    parser.add_argument(
        "--forecast_window",
        type=int,
        default=12,
        help="Number of months ahead to forecast the model",
    )
    args = parser.parse_args()

    main(args.end_month, args.test_window, args.forecast_window)
