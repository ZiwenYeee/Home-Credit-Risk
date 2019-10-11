import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from Basic_function import reduce_mem_usage
from Basic_function import correlation_reduce

def install_first_k_record(df, installment):
    number_of_installment = [2,4,6,8,10, 15, 20, 25, 30]
    installment['instalment_paid_late_in_days'] = installment['DAYS_ENTRY_PAYMENT'] - installment['DAYS_INSTALMENT']
    installment['instalment_paid_late'] = installment['instalment_paid_late_in_days'].apply(lambda x: 1 if x > 0 else 0)
    installment['instalment_paid_over_amount'] = installment['AMT_PAYMENT'] - installment['AMT_INSTALMENT']
    installment['instalment_paid_over'] = installment['instalment_paid_over_amount'].apply(lambda x: 1 if x > 0 else 0)
    features = pd.DataFrame({'SK_ID_CURR': df['SK_ID_CURR'].unique()})
    for i in number_of_installment:
        temp = installment.sort_values(['SK_ID_CURR','SK_ID_PREV','DAYS_ENTRY_PAYMENT']).groupby(['SK_ID_CURR','SK_ID_PREV']).head(i)
        groupby = temp.groupby(['SK_ID_CURR','SK_ID_PREV']).agg({
        'instalment_paid_late_in_days':['mean','min','max','sum'],
        'instalment_paid_late_over_amount':['mean','min','max','sum'],
        'instalment_paid_late':['mean','sum','count'],
        'instalment_paid_late_over':['mean','sum','count']
        })
        groupby.columns = ['_'.join(col) for col in groupby.columns.ravel()]
        groupby.reset_index(inplace = True)
        groupby = groupby.groupby(['SK_ID_CURR']).agg({
        k:['mean','min','max','median','sum'] for k in groupby.columns if k not in ['SK_ID_PREV']
        })
        groupby.columns = ["ins_first_{}_record_".format(i) + "_".join(col) for col in groupby.columns.ravel()]
        groupby.reset_index(inplace = True)
        groupby = correlation_reduce(groupby)
        features = pd.merge(features, groupby, on = ['SK_ID_CURR'], how = 'left')
    features = correlation_reduce(features)
    features.fillna(0, inplace = True)
    df = pd.merge(df, features, on = ['SK_ID_CURR'], how = 'left', validate = 'one_to_one')
    return df
