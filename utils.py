import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean
import os
from os import path
import itertools
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from python_ml_common.config import RedshiftConfig, load_envvars
from python_ml_common.loader.redshift import RedshiftLoader
from pathlib import Path

def loader():
    # Load env vars of TURO_REDSHIFT_USER and TURO_REDSHIFT_PASSWORD
    load_envvars()
    db_config = RedshiftConfig()
    db_config.username = os.getenv("TURO_REDSHIFT_USER")
    db_config.password = os.getenv("TURO_REDSHIFT_PASSWORD")

    # Initialize RedshiftLoader
    rs = RedshiftLoader(db_config)
    
    return rs

def load_data(sql_path, loader):
    # Load data into pd.DataFrame from sql_path
    with open(sql_path, 'r') as f:
        sql = f.read()
        df = loader.load(sql)
    
    return df

def preprocess_df_distribution(df):
    # Preprocess df_distribution
    df['trip_end_month'] = pd.to_datetime(df['trip_end_month'])
    df['paid_days'] = df['paid_days'].astype('float64')
    df['total_paid_days_known'] = df.groupby('trip_end_month')['paid_days'].transform(lambda x: x.loc[df['monaco_bin'] != 'NA'].sum())
    df['distribution_full'] = df['distribution_full'].astype('float64')
    df['distribution'] = df['paid_days'] / df['total_paid_days_known']

    # Start using data from 2021-09-01
    df_subset = df[df['trip_end_month'] >= '2021-09-01'].reset_index(drop=True)
    
    return df_subset

def log_ratio_geometric_transform(df):
    # Apply log ratio geometric transform
    df['log_ratio_geo_distribution'] = np.log(df['distribution'] / df.groupby('trip_end_month')['distribution'].transform(gmean))
    return df

# Function to find the optimal order and seasonal order
# Determine based on the lowest average mae from the rolling validation window
def find_optimal_sarima_order(segment_df, p=range(0, 3), d=range(0, 3), q=range(0, 3), 
                                  sp=range(0, 2), sd=range(0, 2), sq=range(0, 2), s=12):
    best_avg_mae = np.inf
    validation_window = 3
    initial_train_window = len(segment_df) - 4 * validation_window
    best_order = None
    best_seasonal_order = None
    for param in [(x[0], x[1], x[2]) for x in list(itertools.product(p, d, q))]:
        for seasonal_param in [(x[0], x[1], x[2], s) for x in list(itertools.product(sp, sd, sq))]:
            errors = []

            for i in range(4):
                train_end = initial_train_window + i * validation_window
                train_data = segment_df[:train_end]
                val_data = segment_df[train_end:train_end + validation_window]

            try:
                model = sm.tsa.statespace.SARIMAX(train_data,
                                                order=param,
                                                seasonal_order=seasonal_param)
                                                    #enforce_stationarity=False,
                                                    #enforce_invertibility=False)
                results = model.fit(disp=False)
                # Compute mean absolute error
                # forecast = results.get_prediction(start=segment_df.index[0], end=segment_df.index[-1])
                # mse = ((forecast.predicted_mean - segment_df['log_ratio_geo_distribution']) ** 2).mean()
                forecast = results.forecast(steps=validation_window)
                mae = mean_absolute_error(val_data, forecast)
                errors.append(mae)
            except:
                continue
        
            if len(errors) > 0:
                avg_error = np.mean(errors)
                if avg_error < best_avg_mae:
                    best_avg_mae = avg_error
                    best_order = param
                    best_seasonal_order = seasonal_param
            
    return best_order, best_seasonal_order

# Function to fit SARIMA model for each segment
def fit_sarima(df, segment, test_window):
    segment_df = df[df['monaco_bin'] == segment].set_index('trip_end_month')
    # Find optimal order and seasonal order
    optimal_order, optimal_seasonal_order = find_optimal_sarima_order(segment_df['distribution'])

    # Fit SARIMA model with optimal parameters
    train_window = len(segment_df) - test_window
    model = sm.tsa.statespace.SARIMAX(segment_df['distribution'][:train_window],
                                    order=optimal_order,
                                    seasonal_order=optimal_seasonal_order)
                                      #enforce_stationarity=False,
                                      #enforce_invertibility=False)
    results = model.fit(disp=False)
    return results

# Function to forecast 3 months ahead for each segment
def forecast_3_month_ahead(df, test_window):
    # Get unique segments
    segments = df['monaco_bin'].unique()
    sarima_results = {}
    forecast_df = pd.DataFrame()

    # Fit SARIMA model for each segment and save the model and forecast results
    for segment in segments:
        results = fit_sarima(df, segment, test_window) # full data is used for training
        sarima_results[segment] = results
        forecast = results.get_forecast(steps=3).predicted_mean.reset_index()
        forecast.columns = ['trip_end_month', 'distribution_forecast']
        forecast['monaco_bin'] = segment
        forecast_df = pd.concat([forecast_df, forecast], ignore_index=True)

    return (sarima_results, forecast_df)




