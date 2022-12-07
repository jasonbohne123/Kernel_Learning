from datetime import datetime
import pandas as pd
import numpy as np  


def generate_features_from_quotes(quotes,time_agg=60):
    """Generate features from quotes data, aggregates to time_agg and labels outcome (future direction)"""
    
    simple_quotes=quotes[['Exchange','Symbol','Best_Bid_Price','Best_Bid_Size','Best_Offer_Price', 'Best_Offer_Size']]

    # linear features
    simple_quotes=simple_quotes
    simple_quotes.index=pd.to_datetime(simple_quotes.index)
    simple_quotes=simple_quotes.rename(columns={'Best_Bid_Size':'FB0','Best_Offer_Size':'FA0'})
    simple_quotes['FB2']=simple_quotes['FB0'].diff(periods=1)
    simple_quotes['FA2']=simple_quotes['FA0'].diff(periods=1)
    simple_quotes=simple_quotes.dropna()

    # aggregation by time
    intervals=gen_interval(simple_quotes,time_agg)
    simple_quotes['last_interval']=pd.Series(pd.to_datetime(simple_quotes.index)).apply(lambda x: intervals[intervals<x][-1]).values
    simple_quotes['p_time']=simple_quotes.index
    agg_fun={'Exchange':'first','Symbol':'first','Best_Bid_Price':'first','FB0':'first','Best_Offer_Price':'first','FA0':'first','FB2':'first' , 'FA2':'first', 'p_time':'first'}

    grouped_quotes=simple_quotes.groupby('last_interval').agg(agg_fun)

    # outcome labels
    def classify_mid(x):
    
        if x['Next_Best_Bid']>x['Best_Offer_Price']:
            return 1
        elif x['Next_Best_Offer']<x['Best_Bid_Price']:
            return -1
        else:
            return 0

    grouped_quotes['Next_Best_Bid']=grouped_quotes['Best_Bid_Price'].shift(-1)
    grouped_quotes['Next_Best_Offer']=grouped_quotes['Best_Offer_Price'].shift(-1)
    grouped_quotes['outcome']=grouped_quotes.apply(lambda x: classify_mid(x),axis=1)
    grouped_quotes['outcome'].value_counts()/len(grouped_quotes['outcome'].values)

    return grouped_quotes


def gen_interval(quotes,freq):
    """Generate intervals for aggregation"""
    start=datetime.strptime( str(str(quotes.index[0].date()) +" 09:30:00") ,"%Y-%m-%d %H:%M:%S")
    end=datetime.strptime( str(str(quotes.index[-1].date()) +" 16:00:00") ,"%Y-%m-%d %H:%M:%S")

    intervals=np.arange(start, end, np.timedelta64(freq ,'s'), dtype='datetime64[s]')
    return intervals