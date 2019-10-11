import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import warnings
warnings.filterwarnings("ignore")

from Basic_function import safe_div
from Basic_function import timer
from Basic_function import correlation_reduce
from sklearn.linear_model import LinearRegression
from bureau.bureau_function import bureau_trend_features

def bureau_trend(df, bureau):
    features = pd.DataFrame({'SK_ID_CURR': df['SK_ID_CURR'].unique()})
    temp = bureau.loc[~(bureau.AMT_CREDIT_SUM.isnull() ) &
                      ~(bureau.AMT_CREDIT_SUM_DEBT.isnull() ) &
                      (bureau.AMT_CREDIT_SUM != 0) &
                      (bureau.AMT_CREDIT_SUM_DEBT != 0), ['AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT']]
    lr = LinearRegression()
    lr.fit(temp.AMT_CREDIT_SUM.values.reshape(-1, 1), temp.AMT_CREDIT_SUM_DEBT )
    bureau['AMT_CREDIT_SUM_DEBT'] = list(map(lambda x,y: lr.predict(x)[0] if np.isnan(y) else y,
                                         bureau.AMT_CREDIT_SUM,
                                         bureau.AMT_CREDIT_SUM_DEBT ) )
    bureau['AMT_CREDIT_DEBT_RATE'] = list(map(lambda x,y: safe_div(x,y),
                                                   bureau.AMT_CREDIT_SUM_DEBT,
                                                   bureau.AMT_CREDIT_SUM))
    bureau['AMT_CREDIT_DEBT_REMAIN'] = bureau.AMT_CREDIT_SUM - bureau.AMT_CREDIT_SUM_DEBT
    temp = bureau.loc[~(bureau.DAYS_CREDIT.isnull() ) &
                  ~(bureau.DAYS_CREDIT_ENDDATE.isnull() ), ['DAYS_CREDIT','DAYS_CREDIT_ENDDATE']]
    temp = temp.groupby(['DAYS_CREDIT']).DAYS_CREDIT_ENDDATE.mean().reset_index()
    lr_days = LinearRegression()
    lr_days.fit(temp.DAYS_CREDIT.values.reshape(-1,1), temp.DAYS_CREDIT_ENDDATE.values)
    bureau['DAYS_CREDIT_ENDDATE'] = list(map(lambda x,y : lr_days.predict(x)[0] if np.isnan(y) else y,
                                             bureau.DAYS_CREDIT,
                                             bureau.DAYS_CREDIT_ENDDATE) )
    bureau['CREDIT_DAYS_LONG'] = bureau.DAYS_CREDIT_ENDDATE - bureau.DAYS_CREDIT
    with timer("trend feature:"):
        features = bureau_trend_features(features, bureau)
    with timer("basic stat"):
        temp = bureau.groupby(['SK_ID_CURR']).agg({
        k:['mean','min','max','median','sum','std']
        for k in ['AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT' ,
                  'AMT_CREDIT_DEBT_RATE', 'AMT_CREDIT_DEBT_REMAIN',
                  'DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE', 'CREDIT_DAYS_LONG']
        })
        temp.columns = ["bureau_trend_perspective_" + "_".join(j) for j in temp.columns.ravel()]
        temp.reset_index(inplace = True)
        features = pd.merge(features, temp, on = ['SK_ID_CURR'], how = 'left')
    features = correlation_reduce(features)
    df = pd.merge(df, features, on = ['SK_ID_CURR'], how = 'left')
    return df
