# Feature Engineering Generators

def get_spread(data):
    """ Get the spread of the best bid and best offer"""
    data_copy = data.copy()
    data_copy['Spread'] = data_copy['Offer_Price'] - data_copy['Bid_Price']
    data_copy['Spread_Change'] = data_copy['Spread'].diff(periods=1)

    return data_copy[['Spread', 'Spread_Change']]


def get_WAP(data, n_buckets=5):
    """ Get the weighted average price and label weighted bid and ask size of the best bid and best offer"""

    data_copy = data.copy()

    weighted_bid_size_rolling_sum = (
        data_copy['Bid_Size']*data_copy['Bid_Price']).rolling(n_buckets).sum()
    weighted_ask_size_rolling_sum = (
        data_copy['Offer_Size']*data_copy['Offer_Price']).rolling(n_buckets).sum()

    data_copy['WBP'] = weighted_bid_size_rolling_sum/data_copy['Bid_Price']
    data_copy['WAP'] = weighted_ask_size_rolling_sum/data_copy['Offer_Price']

    data_copy['VWAP'] = (data_copy['WBP']+data_copy['WAP'])/2

    return data_copy[['WBP', 'WAP', 'VWAP']]


def get_AWS(data):
    """ Get the average weighted spread and label weighted bid and ask size of the best bid and best offer"""
    data_copy = data.copy()
    data_copy['AWS'] = data_copy['WAP'] - data_copy['WBP']

    return data_copy['AWS']


def get_Outliers(data, n_buckets=5):
    """Binary Classification on outlier cumulative and weigthed bid and ask size of the best bid and best offer"""

    data_copy = data.copy()

    def compute_quantile(x, q=0.95):
        return x.quantile(q)

    # custom rolling callable to label anomalies based on quantile exceedance
    data_copy['Anomaly'] = data_copy['AWS'].rolling(n_buckets).apply(
        lambda x: 1 if x.iloc[-1] > compute_quantile(x) else 0, raw=False)
    return data_copy['Anomaly']


def get_rolling_imbalance(data, n_buckets=5):
    """ Get the rolling imbalance of the best bid and best offer"""
    data_copy = data.copy()
    data_copy['Rolling_Imbalance'] = (
        data_copy['Offer_Size']/data_copy['Bid_Size']).rolling(n_buckets).mean().fillna(0)

    return data_copy['Rolling_Imbalance']
