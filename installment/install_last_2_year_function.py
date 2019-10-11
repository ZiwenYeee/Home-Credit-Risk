import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_model import ARMA

from Basic_function import parallel_apply
from scipy.stats import kurtosis, iqr, skew
import warnings
warnings.filterwarnings('ignore')

def installment_year_add_trend_feature(features,gr,feature_name, prefix):
    gr[feature_name].fillna(0, inplace = True)
    y = gr[feature_name].values
    try:
        x = np.arange(0,len(y)).reshape(-1,1)
        lr = LinearRegression()
        lr.fit(x,y)
        trend = lr.coef_[0]
        predict = lr.predict(x.shape[0])[0]
    except:
        trend = np.nan
        predict = np.nan
    features['{}{}'.format(prefix,feature_name)] = trend
    features['{}{}'.format(prefix,feature_name) + "_predict"] = predict
    try:
        y = gr.loc[gr[feature_name] != 0, feature_name].values
        x = np.arange(0, len(y)).reshape(-1,1)
        lr.fit(x,y)
        tr = lr.coef_[0]
        pr = lr.predict(y.shape[0])[0]
    except:
        tr = np.nan
        pr = np.nan
    features['{}{}'.format(prefix,feature_name) + "_non"] = tr
    features['{}{}'.format(prefix,feature_name) + "_non_predict"] = pr
    try:
        y = gr.loc[gr[feature_name] != 0, feature_name].values
        arma = ARMA(y, order = (0,1)).fit()
        ma_tr = arma.maparams[0]
        ma_pr = arma.forecast(steps = 1)[0][0]
    except:
        ma_tr = np.nan
        ma_pr = np.nan
    features['{}{}'.format(prefix,feature_name) + "_ma"] = ma_tr
    features['{}{}'.format(prefix,feature_name) + "_ma_predict"] = ma_pr
    try:
        y = gr.loc[gr[feature_name] != 0, feature_name].values
        arma = ARMA(y, order = (1, 0) ).fit()
        ar_tr = arma.arparams[0]
        ar_pr = arma.forecast(steps = 1)[0][0]
    except:
        ar_tr = np.nan
        ar_pr = np.nan
    features['{}{}'.format(prefix,feature_name) + "_ar"] = ar_tr
    features['{}{}'.format(prefix,feature_name) + "_ar_predict"] = ar_pr

    return features

def trend_in_last_2_year_k_instalment_features(gr):
    gr_ = gr.copy()
    gr_.sort_values(['DAYS_INSTALMENT'],ascending=False, inplace=True)
    features = {}
    features = installment_year_add_trend_feature(features, gr,
                                                         'instalment_paid_late_in_annuity',
                                                        '2_year_period_trend_')
    features = installment_year_add_trend_feature(features, gr,
                                                         'instalment_paid_late_in_days',
                                                        '2_year_period_trend_')
    features = installment_year_add_trend_feature(features, gr,
                                                         'instalment_paid_late_over_amount',
                                                        '2_year_period_trend_')
    return features
