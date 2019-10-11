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


def previous_diff(df, prev, prefix):
    prev_columns = [(['NAME_CONTRACT_TYPE'], [('amt_goods_ratio', 'mean'),
                                          ('amt_goods_ratio', 'max'),
                                          ('DAYS_LAST_DUE_1ST_VERSION','max'),
                                          ('DAYS_DECISION','mean'),
                                          ('DAYS_DECISION','max')
                                         ])
               ]
    features = prev.copy()
    groupby_aggregate_names = []
    for groupby_cols, specs in tqdm(prev_columns):
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
    for groupby_cols, specs in tqdm(prev_columns):
        for select, agg in tqdm(specs):
            if agg in ['mean','median','max','min']:
                groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
                diff_name = '{}_diff'.format(groupby_aggregate_name)
                features[diff_name] = features[select] - features[groupby_aggregate_name]
                diff_feature_names.append(diff_name)
    features = features.groupby(['SK_ID_CURR']).agg({
    k:['mean', 'min', 'max', 'median', 'std'] for k in ['NAME_CONTRACT_TYPE_mean_amt_goods_ratio_diff',
                                                        'NAME_CONTRACT_TYPE_max_amt_goods_ratio_diff',
                                                        'NAME_CONTRACT_TYPE_max_DAYS_LAST_DUE_1ST_VERSION_diff',
                                                        'NAME_CONTRACT_TYPE_max_DAYS_DECISION_diff',
                                                       'NAME_CONTRACT_TYPE_mean_DAYS_DECISION_diff']
                                                       })
    features.columns = [prefix + "_prev_" + "_".join(j) for j in features.columns.ravel()]
    features.reset_index(inplace = True)
    features.fillna(0, inplace = True)
    df = pd.merge(df, features, on = ['SK_ID_CURR'], how = 'left')
    return df
