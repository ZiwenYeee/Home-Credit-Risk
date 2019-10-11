import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import correlation_reduce
from scipy.stats import kurtosis, iqr, skew
from Basic_function import parallel_apply
from functools import partial
from installment.install_function import last_k_instalment_features_with_fractions

def install_last_k_fraction_feature(df, origin):
    func = partial(last_k_instalment_features_with_fractions,
               periods=[1,5,10,20,50],
               fraction_periods=[(5,20),(5,50)])
    groupby = origin.groupby(['SK_ID_CURR'])
    g = parallel_apply(groupby, func, index_name='SK_ID_CURR',
                   num_workers=4).reset_index()
    # g = correlation_reduce(g)
    df = pd.merge(df, g, on = ['SK_ID_CURR'], how = 'left')
    return df
