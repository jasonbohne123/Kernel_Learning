from datetime import datetime
import pandas as pd
import numpy as np

from Features.generators import *


def generate_features_from_quotes(quotes, time_agg=1, single_dt=None, save=False, partition_dt=False):
    """Generate features from quotes data, aggregates to time_agg and labels outcome (future direction)"""

    quotes_copy = quotes.copy()
    quotes_copy.index = pd.to_datetime(quotes_copy.index)
    quotes_copy = quotes_copy.sort_index()

    quotes_copy['Bid_Size_Diff'] = quotes_copy['Bid_Size'].diff(periods=1)
    quotes_copy['Offer_Size_Diff'] = quotes_copy['Offer_Size'].diff(periods=1)

    quotes_copy[['Spread', 'Spread_Change']] = get_spread(quotes_copy)

    quotes_copy[['WBP', 'WAP', 'VWAP']] = get_WAP(quotes_copy)

    quotes_copy['AWS'] = get_AWS(quotes_copy)

    quotes_copy['Anomaly'] = get_Outliers(quotes_copy)

    quotes_copy['Rolling_Imbalance'] = get_rolling_imbalance(quotes_copy)

    quotes_copy = quotes_copy.dropna()

    # aggregation by time
    intervals = gen_interval(quotes_copy, time_agg)
    quotes_copy['last_interval'] = pd.Series(pd.to_datetime(
        quotes_copy.index)).apply(lambda x: intervals[intervals < x][-1]).values
    quotes_copy['p_time'] = quotes_copy.index
    agg_fun = {'Exchange': 'first', 'Bid_Price': np.mean, 'Offer_Price': np.mean,
               'Bid_Size': np.mean, 'Offer_Size': np.mean, 'Bid_Size_Diff': np.mean, 'Offer_Size_Diff': np.mean,
               'Spread': np.mean, 'Spread_Change': np.mean, 'WBP': np.mean, 'WAP': np.mean,
               'VWAP': np.mean, 'AWS': np.mean, 'Anomaly': np.mean, 'Rolling_Imbalance': np.mean,
               'p_time': 'last'}

    grouped_quotes = quotes_copy.groupby('last_interval').agg(agg_fun)

    # outcome labels
    def classify_mid(x):

        if x['Next_Bid'] > x['Offer_Price']:
            return 1
        elif x['Next_Offer'] < x['Bid_Price']:
            return -1
        else:
            return 0

    grouped_quotes['Next_Bid'] = grouped_quotes['Bid_Price'].shift(
        -1)
    grouped_quotes['Next_Offer'] = grouped_quotes['Offer_Price'].shift(
        -1)
    grouped_quotes['outcome'] = grouped_quotes.apply(
        lambda x: classify_mid(x), axis=1)
    grouped_quotes['outcome'].value_counts(
    )/len(grouped_quotes['outcome'].values)

    if save:
        if partition_dt:
            grouped_quotes['date'] = [i.date() for i in grouped_quotes.index]
            dt_grouped_quotes = grouped_quotes.groupby('date').groups

            for dt in dt_grouped_quotes:
                grouped_quotes.loc[dt_grouped_quotes[dt]].to_csv(
                    '/home/jbohn/jupyter/personal/Kernel_Learning/Features/Cleaned_Features/labeled_data_'+str(dt)+'.csv')
        else:
            if single_dt is not None:
                grouped_quotes.to_csv(
                    f'/home/jbohn/jupyter/personal/Kernel_Learning/Features/Cleaned_Features/labeled_data_{single_dt}.csv')
            else:
                grouped_quotes.to_csv(
                    '/home/jbohn/jupyter/personal/Kernel_Learning/Features/Cleaned_Features/labeled_data.csv')
    if partition_dt:
        return dt_grouped_quotes
    else:
        return grouped_quotes


def gen_interval(quotes, freq):
    """Generate intervals for aggregation"""
    start = datetime.strptime(
        str(str(quotes.index[0].date()) + " 09:30:00"), "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(
        str(str(quotes.index[-1].date()) + " 16:00:00"), "%Y-%m-%d %H:%M:%S")

    intervals = np.arange(start, end, np.timedelta64(
        freq, 's'), dtype='datetime64[s]')
    return intervals
