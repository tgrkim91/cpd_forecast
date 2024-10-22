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
def find_optimal_sarima_order(segment_df, test_window, p=range(0, 3), d=range(0, 3), q=range(0, 3), 
                                  sp=range(0, 2), sd=range(0, 2), sq=range(0, 2), s=12):
    best_avg_mae = np.inf
    validation_window = 3
    initial_train_window = len(segment_df) - test_window - 4 * validation_window
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

def find_optimal_sarimax_order(segment_df, exog, test_window, p=range(0, 3), d=range(0, 3), q=range(0, 3), 
                                sp=range(0, 2), sd=range(0, 2), sq=range(0, 2), s=12):
    best_avg_mae = np.inf
    validation_window = 3
    initial_train_window = len(segment_df) - test_window - 4 * validation_window
    best_order = None
    best_seasonal_order = None
    for param in [(x[0], x[1], x[2]) for x in list(itertools.product(p, d, q))]:
        for seasonal_param in [(x[0], x[1], x[2], s) for x in list(itertools.product(sp, sd, sq))]:
            errors = []

            for i in range(4):
                train_end = initial_train_window + i * validation_window
                train_data = segment_df[:train_end]
                val_data = segment_df[train_end:train_end + validation_window]
                exog_train = exog[:train_end]
                exog_val = exog[train_end:train_end + validation_window]

                try:
                    model = sm.tsa.statespace.SARIMAX(train_data,
                                                    order=param,
                                                    seasonal_order=seasonal_param,
                                                    exog=exog_train)
                    results = model.fit(disp=False)
                    forecast = results.forecast(steps=validation_window, exog=exog_val)
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
    optimal_order, optimal_seasonal_order = find_optimal_sarima_order(segment_df['distribution'], test_window)

    # Fit SARIMA model with optimal parameters
    train_window = len(segment_df) - test_window
    model = sm.tsa.statespace.SARIMAX(segment_df['distribution'][:train_window],
                                    order=optimal_order,
                                    seasonal_order=optimal_seasonal_order)
                                      #enforce_stationarity=False,
                                      #enforce_invertibility=False)
    results = model.fit(disp=False)
    return results

def fit_sarimax(df, segment, test_window):
    segment_df = df[df['monaco_bin'] == segment].set_index('trip_end_month')
    # Create an indicator variable for dates before '2023-04-01'
    segment_df['pre_change'] = (segment_df.index < '2023-04-01').astype(int)
    
    # Find optimal order and seasonal order
    optimal_order, optimal_seasonal_order = find_optimal_sarimax_order(segment_df['distribution_full'], segment_df['pre_change'], test_window)

    # Fit SARIMAX model with optimal parameters and external variable
    train_window = len(segment_df) - test_window
    model = sm.tsa.statespace.SARIMAX(segment_df['distribution_full'][:train_window],
                                      order=optimal_order,
                                      seasonal_order=optimal_seasonal_order,
                                      exog=segment_df['pre_change'][:train_window])
                                      #enforce_stationarity=False,
                                      #enforce_invertibility=False)
    results = model.fit(disp=False)
    return results

# Function to forecast 3 months ahead for each segment
def forecast_3_month_ahead(df, test_window, forecast_window):
    # Get unique segments
    segments = df['monaco_bin'].unique()
    sarima_results = {}
    forecast_df = pd.DataFrame()
    mae_dict = {}

    # Fit SARIMA model for each segment and save the model and forecast results
    for segment in segments:
        if segment != 'NA':
            results = fit_sarima(df, segment, test_window) # full data is used for training
            sarima_results[segment] = results
            forecast = results.get_forecast(steps=forecast_window).predicted_mean.reset_index()
            forecast.columns = ['trip_end_month', 'distribution_forecast']
            forecast['monaco_bin'] = segment
            forecast_df = pd.concat([forecast_df, forecast], ignore_index=True)

            # Calculate MAE for training and prediction datasets
            segment_df = df[df['monaco_bin'] == segment].set_index('trip_end_month')
            train_mae = mean_absolute_error(results.fittedvalues, segment_df['distribution'][:len(results.fittedvalues)])
            if test_window == forecast_window:
                pred_mae = mean_absolute_error(forecast['distribution_forecast'], segment_df['distribution'][-test_window:])
            elif test_window > forecast_window:
                pred_mae = mean_absolute_error(forecast['distribution_forecast'], segment_df['distribution'][-test_window:-(test_window-forecast_window)])
            mae_dict[segment] = {'train_mae': train_mae, 'pred_mae': pred_mae}

    return (sarima_results, forecast_df, mae_dict)

def adjust_ratio(forecast_df):
    # Adjust forecasted distribution to sum up to 1
    forecast_df['distribution_forecast_adj'] = forecast_df.groupby('trip_end_month')['distribution_forecast'].transform(lambda x: x / x.sum())
    return forecast_df

# Function to forecast 3 months ahead for 'NA
def forecast_3_month_ahead_NA(df, test_window, forecast_window):
    # Get unique segments
    sarimax_results = {}
    segment_df = df[df['monaco_bin'] == 'NA'].set_index('trip_end_month')
    segment_df['pre_change'] = (segment_df.index < '2023-04-01').astype(int)
    mae_dict = {}

    # Fit SARIMA model for each segment and save the model and forecast results
    results = fit_sarimax(df, 'NA', test_window) # full data is used for training
    sarimax_results['NA'] = results
    # Note that for any forecast after '2023-04-01', the pre_change variable is 0
    forecast = results.get_forecast(steps=forecast_window, exog=np.zeros(forecast_window)).predicted_mean.reset_index() 
    forecast.columns = ['trip_end_month', 'distribution_NA_forecast']
    forecast['monaco_bin'] = 'NA'

    # Calculate MAE for training and prediction datasets
    train_mae = mean_absolute_error(results.fittedvalues, df[df['monaco_bin'] == 'NA']['distribution_full'][:len(results.fittedvalues)])
    if test_window == forecast_window:
        pred_mae = mean_absolute_error(forecast['distribution_NA_forecast'], segment_df['distribution_full'][-test_window:])
    elif test_window > forecast_window:
        pred_mae = mean_absolute_error(forecast['distribution_NA_forecast'], segment_df['distribution_full'][-test_window:-(test_window-forecast_window)])
    mae_dict['NA'] = {'train_mae': train_mae, 'pred_mae': pred_mae}

    return (sarimax_results, forecast, mae_dict)

# Function to use forecast_NA and forecast_known_segments to calculate the final forecast
def combine_forecast(forecast_df_NA, adj_forecast_df):
    # Combine forecast_NA and forecast_known_segments
    forecast_all = adj_forecast_df.merge(
        forecast_df_NA[['trip_end_month', 'distribution_NA_forecast']], 
        how='left', 
        on=['trip_end_month']
    )
    
    forecast_all['distribution_forecast_final'] = (1-forecast_all['distribution_NA_forecast']) * forecast_all['distribution_forecast_adj']
    forecast_na = forecast_df_NA.rename(columns={'distribution_NA_forecast': 'distribution_forecast_final'})
    forecast_all = pd.concat([forecast_all, forecast_na], ignore_index=True)

    return forecast_all

def cpd_forecast(df_forecast, df_cpd):
    # Merge with cpd data to produce cpd per channel
    df_forecast = df_forecast.merge(
        df_cpd[['analytics_month', 'channels', 'monaco_bin', 'total_cost_per_trip_day']],
        left_on=['trip_end_month', 'monaco_bin'],
        right_on=['analytics_month', 'monaco_bin'],
        how='left'
    )

    df_forecast['cost_per_day'] = df_forecast['distribution_forecast_final'] * df_forecast['total_cost_per_trip_day']
    df_forecast = df_forecast.groupby(['trip_end_month', 'channels'], as_index=False)['cost_per_day'].sum()

    return df_forecast

def save_data(df, file_path):
    file_path = Path(file_path)
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)
    
    df.to_csv(file_path, index=False)


def cpd_accuracy_channel_plot(df):
    channels = df.channels.unique() # payback channels included in df
    num_plots = len(df.channels.unique())

    # list of metrics, colors, labels to utilize in each channel plot
    metrics = ['cpd_forecast_v2', 'cpd_forecast_v1', 'w_cpd_actual']
    colors = ['green', 'lightgreen', 'blue']
    names = ['New CPD Forecast', 'Old CPD Forecast', 'Actual CPD']

    _, axes = plt.subplots(num_plots, 1, figsize = (13,80))
    plt.subplots_adjust(hspace=0.5)

    for ax, channel in zip(axes, channels):
        for metric, color, name in zip(metrics, colors, names):
            ax.plot('forecast_month', metric, '--' if metric!='w_cpd_actual' else '-', label=name, color=color, marker='.', data=df.loc[df.channels==channel])
        
        ax1 = ax.twinx()
        ax1.bar(x = 'forecast_month', height = 'data_volume_y', label = 'Num Trips (Actual)', 
                width=20, color='blue', alpha=0.2,
                data = df.loc[df.channels==channel])
        ax.set_title(channel, fontweight='bold')
        ax.set_ylabel('CPD ($)')
        ax.set_xlabel('signup month')
        ax1.set_ylabel('# trips observed')
        
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax1.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0.)

    plt.show()
