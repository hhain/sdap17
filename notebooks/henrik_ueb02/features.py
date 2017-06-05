import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## feature functions

### element-wise feature functions
def ef_log(data, column_key="ifft", label=None):
    """
    an element-wise feature

    log10 on each element
    """
    if label is not None:
        target = data[label]

    sel_cols = data.columns.values[data.columns.str.contains("ifft")]
    r_data = pd.DataFrame(np.log10(data.loc[:, sel_cols].values))
    # leave column names untouched
    #u_cols = ["col_std_"+str(x) for x in sel_cols]
    r_data.columns = data.columns[data.columns.str.contains("ifft")]

    if label is not None:
        r_data = pd.concat([r_data, target], axis=1)

    return r_data

### column feature functions    
def cf_mean_window(data, window=3, column_key="ifft", label=None ):
    """
    compute rolling std for all columns which contain "ifft"

    Parameters
    ----------

    data : table to operate on
    window : size of rolling/sliding window
    column_key : identifier for columns to use for computation
    label : name of class variable column (which is kept if label is not None)

    """

    if label is not None:
        target = data[label]

    sel_cols = data.columns.values[data.columns.str.contains("ifft")]
    r_data = data.loc[:,sel_cols].rolling(window=window, axis=0).mean()
    # rename
    u_cols = ["col_mean_"+str(x) for x in sel_cols]
    r_data.columns = u_cols

    if label is not None:
        r_data = pd.concat([ r_data, target ], axis=1)

    return r_data


def cf_std_window(data, window=3, column_key="ifft", label=None ):
    """
    compute rolling std for all columns which contain "ifft"

    Parameters
    ----------
    
    data : table to operate on
    window : size of rolling/sliding window
    column_key : identifier for columns to use for computation
    label : name of class variable column (which is kept if label is not None)
        
    """

    if label is not None:
        target = data[label]
   
    sel_cols = data.columns.values[data.columns.str.contains(column_key)]
    r_data = data.loc[:,sel_cols].rolling(window=window, axis=0).std()
    # rename
    u_cols = ["col_std_"+str(x) for x in sel_cols]
    r_data.columns = u_cols
    
    if label is not None:
        r_data = pd.concat([ r_data, target ], axis=1)
    
    return r_data

def cf_var_window(data, window=3, column_key="ifft", label=None ):
    """
    compute rolling var for all columns which contain "ifft"
    
    Parameters
    ----------
    
    data : table to operate on
    window : size of rolling/sliding window
    column_key : identifier for columns to use for computation
    label : name of class variable column (which is kept if label is not None)
        
    """

    if label is not None:
        target = data[label]
   
    sel_cols = data.columns.values[data.columns.str.contains("ifft")]
    r_data = data.loc[:,sel_cols].rolling(window=window, axis=0).var()
    # rename
    u_cols = ["col_var_"+str(x) for x in sel_cols]
    r_data.columns = u_cols
    
    r_data = pd.concat([ r_data, target ], axis=1)
    
    return r_data



def cf_diff(data, column_key="ifft", label=None):
    """
    computes the diff for each column
    
    useful for removing signal distortions which are system error artifacts and not an actual
    environmental effect in the signal
    """
    
    if label is not None:
        target = data[label]
    
    sel_cols = data.columns.values[data.columns.str.contains(column_key)]
    
    r_data = pd.DataFrame(np.diff(data.loc[:,sel_cols].values, axis=0))

    u_cols = ["col_diff_"+str(x) for x in sel_cols]
    r_data.columns = u_cols

    if label is not None:
        r_data = pd.concat([ r_data, target ], axis=1)
        
        
    return r_data


# todo
def cf_ptp(data, column_key="ifft", label=None):
    """
    computes the diff for each column
    
    useful for removing signal distortions which are system error artifacts and not an actual
    environmental effect in the signal
    """
    
    if label is not None:
        target = data[label]
    
    sel_cols = data.columns.values[data.columns.str.contains("ifft")]
    
    r_data = pd.DataFrame(np.diff(data.loc[:,sel_cols].values, axis=0))

    u_cols = ["col_diff_"+str(x) for x in sel_cols]
    r_data.columns = u_cols
    
    r_data = pd.concat([ r_data, target ], axis=1)
    
    return r_data
    
    
def cf_replace(data,  dest_index, src_index, column_key="ifft",label=None):
    """
    replaces an entry in the defined columns using dest and src index (i.e. basically replaces a single row with another one)
    
    """
    if label is not None:
        target = data[label]
    
    sel_cols = data.columns.values[data.columns.str.contains("ifft")]
    
    r_data = data.loc[:,sel_cols]
    
    r_data.iloc[dest_index,:] = r_data.iloc[src_index,:]

    #u_cols = ["col_diff_"+str(x) for x in sel_cols]
    #r_data.columns = u_cols
    
    r_data = pd.concat([ r_data, target ], axis=1)
    
    return r_data



def rf_grouped(data, groups, fn, label=None, subgroups = None):
    """
    generic function which applies row features on column groups and subgroups, see example use-case for more info
    
    Parameters
    ----------
    groups : list of lists of strings which identify columns which should be summed together
    
    subgroups : list of string which identify columns of each subgroup
    
    fn : function has p1 and p2 to execute for each pair of column cells / the function should return dataframe with new col names
    
    label : name of the label column; may also be None
    
    Example Use-Case
    ----------------
    
    Assume following data structure (DF):
    
        DataFrame({ 
                    'trx_1-2_ifft_1':data1, 'trx_1-2_ifft_2':data2,
                    'trx_2-1_ifft_1':data3, 'trx_2-1_ifft_1':data4,
                    'trx_3-4_ifft_1':data5, 'trx_3-4_ifft_6':data2,
                    'trx_4-3_ifft_1':data7, 'trx_4-3_ifft_8':data4
                    })

    Required computation
    
        Sum all same iffts of 1-2 and 2-1 together, ie. such that trx_1-2_ifft = trx_1-2_ifft_1 + trx_2-1_ifft_1 ..
        
    
    Remark
    ------
    
    - If subgroups is None, function is executed only on group selection
    - the new dataframe contains only the new columns which were selected + the label column (if defined), i.e. all other columns are removed
    
    """
        
    r_data = None
    
    # save label column
    if label is not None:
        r_data = pd.DataFrame(data[label])
    
    
    for g in groups:

        if subgroups is not None:
            raise NotImplementedError("Subgroups not implemented yet")
            pass

        # TODO: make case selection more elegant
        
        # filter groups based on column names
        if len(g) == 1:
            sel_p1 = g[0]
            sel_cols_p1 = data.columns.str.contains(sel_p1)
            p1 = data.loc[:,sel_cols_p1]
            tmp_data = fn( [p1] )

        elif len(g) == 2:
            sel_p1 = g[0]
            sel_p2 = g[1]

            sel_cols_p1 = data.columns.str.contains(sel_p1)
            sel_cols_p2 = data.columns.str.contains(sel_p2)

            p1 = data.loc[:,sel_cols_p1]
            p2 = data.loc[:,sel_cols_p2]

            tmp_data = fn( [p1, p2] )

        elif len(g) == 3:
            sel_p1 = g[0]
            sel_p2 = g[1]
            sel_p3 = g[2]

            sel_cols_p1 = data.columns.str.contains(sel_p1)
            sel_cols_p2 = data.columns.str.contains(sel_p2)
            sel_cols_p3 = data.columns.str.contains(sel_p3)

            p1 = data.loc[:,sel_cols_p1]
            p2 = data.loc[:,sel_cols_p2]
            p3 = data.loc[:,sel_cols_p3]

            tmp_data = fn( [p1, p2, p3] )
        
        else:
            raise NotImplementedError("Groups with %i entries not implemented yet"%len(g))

        
        if r_data is None:
            r_data = tmp_data
            
        else:
            r_data = pd.concat([ r_data, tmp_data ], axis=1)
            
    return r_data



def rf_sum_pair( p_list ):
    """
    Sums DFs p1 and p2 and provides new column names
    """
    name = 'rf_sum_pair'
    
    n_df = pd.DataFrame( p_list[0].values + p_list[1].values )
    
    # todo: new column names which indicate use of feature
    n_cols = ["["+name+":"+str(p_list[0].columns[i])+"|"+str(p_list[1].columns[i])+"]" for i,_ in enumerate(p_list[0].columns)]
    n_df.columns = n_cols
    
    return n_df


def rf_kurtosis_single( p_list ):
    """
    kurtosis accross rows
    """
    from scipy.stats import kurtosis
    n_df = pd.DataFrame( kurtosis( p_list[0].values, axis=1) )
    
    #n_df.columns = [p_list[0].columns[0]]
    n_cols = ['rf_kurt_'+str(p_list[0].columns.values[0]) ]
    n_df.columns = n_cols
    
    return n_df
    #rowFeatures=c("kurtosis",  "skewness", "sd" )
    
def rf_skew_single( p_list ):
    """
    skewness accross rows
    """
    from scipy.stats import skew
    n_df = pd.DataFrame( skew( p_list[0].values, axis=1) )
    
    #n_df.columns = [p_list[0].columns[0]]
    n_cols = ['rf_skew_'+str(p_list[0].columns.values[0]) ]
    n_df.columns = n_cols
    
    return n_df
    
    #rowFeatures=c("kurtosis",  "skewness", "sd" )
    

def rf_sum_single( p_list ):
    """
    Sums all columns of the provided DF
    """
    
    n_df = pd.DataFrame( np.sum( p_list[0].values, axis=1) )
    n_cols = ['rf_sum_'+x for x in p_list[0].columns ]
    n_df.columns = n_cols
    
    return n_df

def rf_var_single( p_list ):
    """
    Sums all columns of the provided DF
    """
    
    n_df = pd.DataFrame( np.var( p_list[0].values, axis=1) )
    
    n_df.columns = [p_list[0].columns[0]]
    
    return n_df


def rf_std_single( p_list ):
    """
    Sums all columns of the provided DF
    """
    
    n_df = pd.DataFrame( np.std( p_list[0].values, axis=1) )
    #print(n_df.columns)
    n_cols = ['rf_std_'+str(p_list[0].columns.values[0]) ]
    n_df.columns = n_cols
    
    return n_df

def rf_median_single( p_list ):
    """
    Sums all columns of the provided DF
    """
    
    n_df = pd.DataFrame( np.median( p_list[0].values, axis=1) )
    #print(n_df.columns)
    n_cols = ['rf_median_'+str(p_list[0].columns.values[0]) ]
    n_df.columns = n_cols
    
    return n_df

def rf_mean_single( p_list ):
    """
    Sums all columns of the provided DF
    """
    
    n_df = pd.DataFrame( np.mean( p_list[0].values, axis=1) )
    #print(n_df.columns)
    n_cols = ['rf_mean_'+str(p_list[0].columns.values[0]) ]
    n_df.columns = n_cols
    
    return n_df
    
def rf_max_single( p_list ):
    """
    Sums all columns of the provided DF
    """
    
    n_df = pd.DataFrame( np.max( p_list[0].values, axis=1) )
    #print(n_df.columns)
    n_cols = ['rf_max_'+str(p_list[0].columns.values[0]) ]
    n_df.columns = n_cols
    
    return n_df
    
def rf_ptp_single( p_list ):
    """
    p2p
    """
    n_df = pd.DataFrame( np.ptp( p_list[0].values, axis=1) )
    
    n_cols = ['rf_ptp_'+str(p_list[0].columns.values[0]) ]
    n_df.columns = n_cols
    
    return n_df
    
    

def rf_sum_abs_single( p_list ):
    """
    Sums all columns of the provided DF
    """
    
    n_df = pd.DataFrame( np.sum( np.abs(p_list[0].values), axis=1) )
    
    n_df.columns = [p_list[0].columns[0]]
    
    return n_df
    
    
def rf_weighted_average( p_list ):
    """
    average all columns of plist
    """

    w=[1*x for x in range(0,p_list[0].shape[1] )]
    n_df = pd.DataFrame( np.average( p_list[0].values, weights=w, axis=1 )  )
    
    n_df.columns = [p_list[0].columns[0]]
    
    return n_df
