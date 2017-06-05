import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from features import *
from utility import *

## filter functions
def distortion_filter(data, column_key="_ifft_0", label='target', std_thresh=100.):
    """
    replaces distortions in the data by checking for high std
    replacement is performed by using previous data entry (duplication)

    Returns
    -------
    * corrected data
    * conducted number of replacements in each trx

    Remark
    ------
    run create label before
    """

    std_thresh_ = std_thresh
    lst = data.columns[data.columns.str.contains(column_key)]
    groups = [[x[:-2]] for x in lst]
    r_data = data.copy()

    n_errors = {}

    # for each groups of trxs
    for trx_group in groups:

        trx = trx_group[0]
        #print(trx)
        data_diff = cf_diff(data, column_key=trx, label='target')
        data_sum_groups = rf_grouped(data_diff, groups=[[trx]],
                                     fn=rf_sum_abs_single,
                                     label='target')
        data_std = cf_std_window(data_sum_groups, label='target', window=3)


        sel_col = data_std.columns[data_std.columns.str.contains(trx)]

        # in abs sum this is the 2nd index which needs to be corrected
        df_thresh_ix = np.where(data_std[sel_col] > std_thresh_ )[0]

        # list empty?
        if len(df_thresh_ix) == 0:
            n_errors[trx] = 0
            continue

        df_thresh_ix_groups = get_contigous_borders(df_thresh_ix)

        n_errors[trx] = len(df_thresh_ix_groups)

        # note: we assume that each threshold exceeded is triggered by a single problematic measurement
        # in the same window, hence there are 3 high stds |0 0 1|, |0 1 0| and |1 0 0| where 1 indicates the problematic measurement
        # of course we only want to correct on measurement
        for g in df_thresh_ix_groups:

            # only a single element
            if g[1] - g[0] == 0:
                # replace only this element in original data
                # select columns
                r_data = cf_replace(data, column_key=trx, label="target", dest_index = g[0], src_index = g[0]-1)

            else:
                # replace 2nd element in original data
                r_data = cf_replace(data, column_key=trx, label="target", dest_index = g[0]+1, src_index = g[0]-1)


    return r_data, n_errors

    