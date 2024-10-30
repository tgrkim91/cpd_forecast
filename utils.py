from typing import Tuple
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
from joblib import Parallel, delayed

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

def preprocess_df_distribution(df, df_channel):
    # Preprocess df_distribution
    df['trip_end_month'] = pd.to_datetime(df['trip_end_month'])
    df['paid_days'] = df['paid_days'].astype('float64')
    df['total_paid_days_known'] = df.groupby('trip_end_month')['paid_days'].transform(lambda x: x.loc[df['monaco_bin'] != 'NA'].sum())
    df['distribution_full'] = df['distribution_full'].astype('float64')
    df['distribution'] = df['paid_days'] / df['total_paid_days_known']

    # Start using data from 2021-09-01
    # TODO: Change the end date to be dynamically populated
    df_subset = df[(df['trip_end_month'] >= '2021-09-01') & (df['trip_end_month'] <= '2024-10-01')].reset_index(drop=True)
    
    # Preprocess df_distribution_channel
    df_channel['trip_end_month'] = pd.to_datetime(df_channel['trip_end_month'])
    df_channel['paid_days'] = df_channel['paid_days'].astype('float64')
    df_channel['distribution_full'] = df_channel['distribution_full'].astype('float64')

    # Start using data from 2021-09-01
    df_channel_subset = df_channel[(df_channel['trip_end_month'] >= '2021-09-01') & (df_channel['trip_end_month'] <= '2024-10-01')].reset_index(drop=True)

    return df_subset, df_channel_subset

def log_ratio_geometric_transform(df):
    # Apply log ratio geometric transform
    df['log_ratio_geo_distribution'] = np.log(df['distribution'] / df.groupby('trip_end_month')['distribution'].transform(gmean))
    return df

# Install some autoformatters to make your life easy, we can discuss
# Function to find the optimal order and seasonal order
# Determine based on the lowest average mae from the rolling validation window
def find_optimal_sarima_order(
    segment_df,
    test_window,
    p=range(0, 3),
    d=range(0, 3),
    q=range(0, 3),
    sp=range(0, 2),
    sd=range(0, 2),
    sq=range(0, 2),
    s=12,
):
    best_avg_mae = np.inf
    validation_window = 3
    # We conduct 4 folds for a rolling validation window of 3 months to capture a full year of seasonality
    # I don't know how/if this makes sense for the initial obsservations, I guess we're using data from 2021-09 so maybe it's alright
    initial_train_window = len(segment_df) - test_window - 4 * validation_window
    best_order = None
    best_seasonal_order = None

    def evaluate_order(param, seasonal_param):
        errors = []
        for i in range(4):
            train_end = initial_train_window + i * validation_window
            train_data = segment_df[:train_end]
            val_data = segment_df[train_end:train_end + validation_window]

            try:
                model = sm.tsa.statespace.SARIMAX(train_data,
                                                  order=param,
                                                  seasonal_order=seasonal_param)
                results = model.fit(disp=False)
                forecast = results.forecast(steps=validation_window)
                mae = mean_absolute_error(val_data, forecast)
                errors.append(mae)
            except:
                continue

        if len(errors) > 0:
            avg_error = np.mean(errors)
            return avg_error, param, seasonal_param
        else:
            return np.inf, None, None

    # I appreciate the rigor here, I feel like there must be a more efficient way of doing this
    # Could we maybe look to restrict the search space based on outright bad options?
    results = Parallel(n_jobs=-1)(delayed(evaluate_order)(param, seasonal_param)
                                  for param in [(x[0], x[1], x[2]) for x in list(itertools.product(p, d, q))]
                                  for seasonal_param in [(x[0], x[1], x[2], s) for x in list(itertools.product(sp, sd, sq))])

    # Re the above, you should include a bool param that allows you to return all results, not just the best one
    for avg_error, param, seasonal_param in results:
        if avg_error < best_avg_mae:
            best_avg_mae = avg_error
            best_order = param
            best_seasonal_order = seasonal_param

    return best_order, best_seasonal_order

# Again, I would look for ways to incorporate this into the above, really all that's happening is you're supplying exog in one case and setting it to None in the other
def find_optimal_sarimax_order(
    segment_df,
    exog,
    test_window,
    p=range(0, 3),
    d=range(0, 3),
    q=range(0, 3),
    sp=range(0, 2),
    sd=range(0, 2),
    sq=range(0, 2),
    s=12,
):
    best_avg_mae = np.inf
    validation_window = 3
    initial_train_window = len(segment_df) - test_window - 4 * validation_window
    best_order = None
    best_seasonal_order = None

    def evaluate_order(param, seasonal_param):
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
            return avg_error, param, seasonal_param
        else:
            return np.inf, None, None

    results = Parallel(n_jobs=-1)(delayed(evaluate_order)(param, seasonal_param)
                                  for param in [(x[0], x[1], x[2]) for x in list(itertools.product(p, d, q))]
                                  for seasonal_param in [(x[0], x[1], x[2], s) for x in list(itertools.product(sp, sd, sq))])

    for avg_error, param, seasonal_param in results:
        if avg_error < best_avg_mae:
            best_avg_mae = avg_error
            best_order = param
            best_seasonal_order = seasonal_param

    return best_order, best_seasonal_order


# Function to fit SARIMA model for each segment
def fit_sarima(df, 
               segment, 
               test_window
               ) -> sm.tsa.statespace.MLEResults:
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

## Could this function be integrated into the above function? Both are off a SARIMAX object just with different arguments, could avoid repeating yourself a bit
def fit_sarimax(df, segment, test_window):
    segment_df = df[df['monaco_bin'] == segment].set_index('trip_end_month')
    # Create an indicator variable for dates before '2023-04-01'
    # Shouldn't this be June 2023? That's when the real change occurred, but not sure how much that'll change results
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

# Example of docstrings, recommend installing AutoDocstring extension in VSCode, type hinting will also be helpful to provide auto-complete options on variables. Also bit of a misnomer on "forecast_3_month_ahead" as the window is paramaterized.
# Function to forecast 3 months ahead for each segment
def forecast_3_month_ahead(
        df: pd.DataFrame,
        test_window: int, 
        forecast_window: int
        ) -> Tuple[dict, pd.DataFrame, dict]:
    """
    Function to forecast n months ahead for each segment, excluding 'NA'

    Args:
        df: Initial distribution df
        test_window: Length of holdout for testing of model
        forecast_window: Number of period to predict

    Returns:
        Tuple of SARIMA results, forecasted distribution, and MAE for each segment
    """
    # Get unique segments
    segments = df['monaco_bin'].unique()
    sarima_results = {}
    forecast_df = pd.DataFrame()
    mae_dict = {}

    # Fit SARIMA model for each segment and save the model and forecast results
    for segment in segments:
        if segment != 'NA': # Redundant given filtering in cpd_pipeline no?
            # I would maybe retitle "results" to "model" to be more clear on object type
            results = fit_sarima(df, segment, test_window) # full data is used for training
            sarima_results[segment] = results
            forecast = results.get_forecast(steps=forecast_window).predicted_mean.reset_index()
            forecast.columns = ['trip_end_month', 'distribution_forecast']
            forecast['monaco_bin'] = segment
            forecast_df = pd.concat([forecast_df, forecast], ignore_index=True)

            # Calculate MAE for training and prediction datasets
            segment_df = df[df['monaco_bin'] == segment].set_index('trip_end_month')
            # I'm a bit confused here, where are we predicting during the training interval? The `get_forecast` method invokes `get_prediction` starting at the end of the prediction interval, so I'm not sure how we're calculating the MAE for the training interval
            train_mae = mean_absolute_error(results.fittedvalues, segment_df['distribution'][:len(results.fittedvalues)])
            pred_mae = np.nan
            # I'm going to need you to walk me through this I think
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
    # This date should be coded into settings or included in a higher function, seeing it popping up across multiple util functions
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
    pred_mae = np.nan
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

def seasonal_ratio(df_forecast, df_distribution):
    # df_forecast -> |trip_end_month | monaco_bin | distribution_forecast_final
    # df_distribution -> |trip_end_month | monaco_bin | distribution_full
    # result -> trip_end_month (forecast_month) | monaco_bin | ratio (ratio compared to the previous 3 month average)
    
    df_distribution['total_paid_days'] = df_distribution.groupby('trip_end_month')['paid_days'].transform('sum')

    def weighted_rolling_avg(group, window=3):
        weights = group['total_paid_days']
        values = group['distribution_full']
        result = values.rolling(window=window, min_periods=1).apply(lambda x: np.sum(x * weights.loc[x.index]) / np.sum(weights.loc[x.index]), raw=False)
        return result

    # Calculate the 3-month weighted rolling average of distribution_full
    df_distribution['rolling_avg_distribution'] = df_distribution.groupby('monaco_bin').apply(weighted_rolling_avg).reset_index(level=0, drop=True)
    df_distribution['forecast_month'] = df_distribution['trip_end_month'] + pd.DateOffset(months=1)

    # Merge forecast and distribution dataframes on trip_end_month and monaco_bin
    df_merged = df_forecast.merge(df_distribution[['forecast_month', 'monaco_bin', 'rolling_avg_distribution']],
                                  left_on=['trip_end_month', 'monaco_bin'],
                                  right_on=['forecast_month', 'monaco_bin'],
                                  how='left')

    # Calculate the ratio of distribution_forecast_final to the 3-month rolling average
    df_merged['ratio'] = df_merged['distribution_forecast_final'] / df_merged['rolling_avg_distribution']
    
    # Select the relevant columns for the result
    result = df_merged[['trip_end_month', 'monaco_bin', 'rolling_avg_distribution', 'distribution_forecast_final', 'ratio']]
    return result

def forecast_distribution_channel(df_ratio, df_distribution_channel):
    # df_ratio -> |trip_end_month (forecast_month) | monaco_bin | ratio
    # df_distribution_channel -> |trip_end_month | monaco_bin | channels | distribution_full
    
    df_distribution_channel['total_paid_days'] = df_distribution_channel.groupby(['channels', 'trip_end_month'])['paid_days'].transform('sum')

    def weighted_rolling_avg_channel(group, window=3):
        weights = group['total_paid_days']
        values = group['distribution_full']
        result = values.rolling(window=window, min_periods=1).apply(lambda x: np.sum(x * weights.loc[x.index]) / np.sum(weights.loc[x.index]), raw=False)
        return result

    # Calculate the 3-month rolling average of distribution_full
    df_distribution_channel['rolling_avg_distribution_channel'] = df_distribution_channel.groupby(['channels', 'monaco_bin'], as_index=False).apply(weighted_rolling_avg_channel).reset_index(level=0, drop=True)
    df_distribution_channel['forecast_month'] = df_distribution_channel['trip_end_month'] + pd.DateOffset(months=1)

    # Merge ratio and distribution_channel dataframes on trip_end_month and monaco_bin
    df_merged = df_ratio.merge(df_distribution_channel[['forecast_month', 'monaco_bin', 'channels', 'rolling_avg_distribution_channel']], 
                               left_on=['trip_end_month', 'monaco_bin'],
                               right_on=['forecast_month', 'monaco_bin'],
                               how='left')

    # Calculate the forecasted distribution for each channel
    df_merged['distribution_forecast_channel'] = df_merged['ratio'] * df_merged['rolling_avg_distribution_channel']
    
    # Normalize to make the sum of distribution_forecast_channel equal to 1 for each trip_end_month
    df_merged['distribution_forecast_channel_adj'] = df_merged.groupby(['trip_end_month', 'channels'])['distribution_forecast_channel'].transform(lambda x: x / x.sum())
   
    # Select the relevant columns for the result
    result = df_merged[['trip_end_month', 'monaco_bin', 'channels', 'ratio', 'rolling_avg_distribution_channel',  'distribution_forecast_channel_adj']]
    return result

def cpd_forecast(df_forecast, df_cpd):
    # Merge with cpd data to produce cpd per channel
    # df_forecast -> |trip_end_month | monaco_bin | distribution_forecast_final | ratio| channels | total_cost_per_trip_day |
    df_cpd['forecast_month'] = df_cpd['analytics_month'] + pd.DateOffset(months=1)

    df_forecast = df_forecast.merge(
        df_cpd[['forecast_month', 'channels', 'monaco_bin', 'total_cost_per_trip_day']],
        left_on=['trip_end_month', 'monaco_bin', 'channels'],
        right_on=['forecast_month', 'monaco_bin', 'channels'],
        how='left'
    )

    df_forecast['cost_per_day'] = df_forecast['distribution_forecast_channel_adj'] * df_forecast['total_cost_per_trip_day']
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
