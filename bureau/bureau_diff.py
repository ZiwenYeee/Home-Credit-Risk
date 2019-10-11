import os
import numpy as np
import pandas as pd
import sys
from Basic_function import safe_div
from contextlib import contextmanager
import multiprocessing as mp
from tqdm import tqdm_notebook as tqdm
from sklearn.externals import joblib
import warnings
warnings.filterwarnings('ignore')

def bureau_diff(df, bureau, prefix):
    bureau['AMT_CREDIT_DEBT_RATE'] = list(map(lambda x,y: safe_div(x,y),
                                                   bureau.AMT_CREDIT_SUM_DEBT,
                                                   bureau.AMT_CREDIT_SUM))
    bureau['AMT_CREDIT_DAYS_LONG'] = bureau.DAYS_CREDIT_ENDDATE - bureau.DAYS_CREDIT
    bureau['New_credit_type'] = bureau.CREDIT_TYPE.apply(lambda x: 'Others' if x not in ['Consumer credit','Credit card'] else x)
    bureau['AMT_CREDIT_DEBT_REMAIN'] = bureau.AMT_CREDIT_SUM - bureau.AMT_CREDIT_SUM_DEBT
    bureau_col = [(['CREDIT_CURRENCY'],[('AMT_CREDIT_DEBT_RATE','mean'),
                                    ('AMT_CREDIT_DEBT_RATE','max'),
                                    ('DAYS_CREDIT','mean'),
                                    ('DAYS_CREDIT_UPDATE','mean'),
                                    ('DAYS_CREDIT_UPDATE','max'),
                                    ('DAYS_CREDIT_ENDDATE', 'mean'),
                                    ('AMT_CREDIT_SUM_DEBT', 'mean'),
                                    ('AMT_CREDIT_MAX_OVERDUE', 'mean'),
                                    ('AMT_CREDIT_SUM', 'mean'),
                                    ('AMT_CREDIT_DEBT_REMAIN', 'mean')

                                   ]),
              (['CREDIT_ACTIVE'],[('AMT_CREDIT_DEBT_RATE','mean'),
                                    ('AMT_CREDIT_DEBT_RATE','max'),
                                    ('DAYS_CREDIT','mean'),
                                    ('DAYS_CREDIT_ENDDATE', 'mean'),
                                    ('AMT_CREDIT_SUM_DEBT', 'mean'),
                                    ('AMT_CREDIT_MAX_OVERDUE', 'mean'),
                                    ('AMT_CREDIT_DEBT_REMAIN','mean')

                                    ]),
              (['New_credit_type'],[('AMT_CREDIT_DEBT_RATE','mean'),
                                    ('AMT_CREDIT_DEBT_RATE','max'),
                                    ('DAYS_CREDIT','mean'),
                                    ('DAYS_CREDIT_ENDDATE', 'mean'),
                                    ('AMT_CREDIT_SUM_DEBT', 'mean'),
                                    ('AMT_CREDIT_SUM','mean'),
                                    ('AMT_CREDIT_MAX_OVERDUE', 'mean'),
                                    ('AMT_CREDIT_DEBT_REMAIN','mean')

                                    ]),
              (['New_credit_type','CREDIT_CURRENCY'],[('AMT_CREDIT_DEBT_RATE','mean'),
                                    ('AMT_CREDIT_DEBT_RATE','max'),
                                    ('DAYS_CREDIT','mean'),
                                    ('DAYS_CREDIT_ENDDATE', 'mean'),
                                    ('AMT_CREDIT_SUM_DEBT', 'mean'),
                                    ('AMT_CREDIT_SUM','mean'),
                                    ('AMT_CREDIT_MAX_OVERDUE', 'mean'),
                                    ('AMT_CREDIT_DEBT_REMAIN','mean')

                                    ])
             ]

    features = bureau.copy()
    groupby_aggregate_names = []
    for groupby_cols, specs in tqdm(bureau_col):
        group_object = features.groupby(groupby_cols)
        for select, agg in tqdm(specs):
            groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
            features = features.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
            groupby_aggregate_names.append(groupby_aggregate_name)
    diff_feature_names = []
    for groupby_cols, specs in tqdm(bureau_col):
        for select, agg in tqdm(specs):
            if agg in ['mean','median','max','min']:
                groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
                diff_name = '{}_diff'.format(groupby_aggregate_name)
                features[diff_name] = features[select] - features[groupby_aggregate_name]
                diff_feature_names.append(diff_name)
    features[diff_feature_names]
    features = features.groupby(['SK_ID_CURR']).agg({
    k:['mean', 'max', 'sum', 'std', 'median'] for k in diff_feature_names
    })
    features.columns = [prefix + "_bureau_" + "_".join(j) for j in features.columns.ravel()]
    features = correlation_reduce(features)
    features.reset_index(inplace = True)
    features.fillna(0, inplace = True)
    df = pd.merge(df, features, on = ['SK_ID_CURR'], how = 'left', validate = 'one_to_one')
    return df
