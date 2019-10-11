
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
from sklearn.linear_model import LinearRegression
from Basic_function import parallel_apply
import warnings
warnings.filterwarnings('ignore')

def _add_bureau_trend_feature(features,gr,feature_name, prefix):
    y = gr[feature_name].values
    try:
        x = np.arange(0,len(y)).reshape(-1,1)
        lr = LinearRegression()
        lr.fit(x,y)
        trend = lr.coef_[0]
        predict = lr.predict(x.shape[0])[0]
    except:
        trend=np.nan
        predict = np.nan
    features['{}{}'.format(prefix,feature_name)] = trend
    features['{}{}_predict'.format(prefix,feature_name)] = predict
    return features
def trend_in_last_k_bureau_features(gr):
    gr_ = gr.copy()
    gr_.sort_values(['DAYS_CREDIT'],ascending=False, inplace=True)

    features = {}
    features = _add_bureau_trend_feature(features, gr_, 'AMT_CREDIT_DEBT_RATE', 'all_trend_')
    features = _add_bureau_trend_feature(features, gr_, 'AMT_CREDIT_DEBT_REMAIN', 'all_trend_')
    features = _add_bureau_trend_feature(features, gr_, 'AMT_CREDIT_SUM', 'all_trend_')
    features = _add_bureau_trend_feature(features, gr_, 'AMT_CREDIT_SUM_DEBT', 'all_trend_')
    return features

def bureau_trend_features(df, bureau):
    groupby = bureau.groupby(['SK_ID_CURR'])
    g = parallel_apply(groupby, trend_in_last_k_bureau_features,
                        index_name = 'SK_ID_CURR', num_workers = 8).reset_index()
    df = pd.merge(df, g, on = ['SK_ID_CURR'], validate = 'one_to_one')
    return df
