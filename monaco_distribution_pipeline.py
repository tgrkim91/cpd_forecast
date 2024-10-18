import logging
import os
from os import path
from datetime import datetime
from utils import (
    loader,
    load_data,
    preprocess_df_distribution,
    forecast_3_month_ahead
)

CURRENT_DIR = path.dirname(path.abspath(__file__))
SQL_DIR = path.join(CURRENT_DIR, "sql")

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # initialize redshift loader
    rs = loader()
    logger.info("Start Monaco Distribution Pipeline")

    # Load time series of monaco distribution data
    df_distribution = load_data(
        sql_path=path.join(SQL_DIR, 'monaco_distribution.sql'), 
        loader=rs)

    # Preprocess df_distribution
    logger.info("Preprocess Data")
    df_distribution_pre = preprocess_df_distribution(df_distribution)

    # Fit SARIMA model with the opimal order and seasonal orders for each segment
    # and forecast 3 months ahead for each segment
    logger.info("Fit Model and Forecast 3 Months Ahead")
    model_results, forecast_df = forecast_3_month_ahead(df_distribution_pre, 3)

    return forecast_df

if __name__ == "__main__":
    df = main()
    print(df)