import numpy as np
import pandas as pd
import datetime

import warnings

def prepare_dataframe(df, drop_cols, return_wrong_dates=False):
    """Preparation of dataframe for specific data
    
    Specificity is in that df have to contain data in separate columns ‘YY‘, ‘MM‘, ‘DD‘
    YY must be written in 2 digits or 4, the 1st case will be transformed as 1900 + YY if YY >= 83 else 2000 + YY
    Combinations of ‘YY‘, ‘MM‘, ‘DD‘ that don't represent real date will be dropped
    As a result ‘YY‘, ‘MM‘, ‘DD‘ will be transformed in ‘time‘ represented time object
    
    Parameters:
        df - pandas dataframe
        drop_cols - column that should be dropped
        return_wrong_dates - if return wrong dates if they'll be found in df
    
    Return:
        Transformed dataframe only or with list of wrong dates (see “return_wrong_dates“ parameter)
        order df (, wrong dates)
    """
    
    df = df.drop(drop_cols, axis=1)
    #delete rows where date is unknown
    df = df.dropna(subset=['YY','MM', 'DD'], how='any')
    df = df.reset_index(drop=True)
    #transformation of date
    df.loc[:, 'YY':'DD'] = df.loc[:, 'YY':'DD'].astype(int)
    for i, x in enumerate(df['YY']):
        if x < 100:
            df.loc[i, 'YY'] = 1900 + x if x >= 83 else 2000 + x

    date_column, index_error = [], []
    wrong_dates = []
    for i in range(df.shape[0]):
        try:
            date_column.append(datetime.date(df.iloc[i,0],df.iloc[i,1],df.iloc[i,2]))
        except ValueError:
            warnings.warn(f"\nGot wrong date YY MM DD, it'll be dropped", stacklevel=2)
            index_error.append(i) 
    
    wrong_dates.append(df.iloc[index_error, 0 : 3].values[0])
    df = df.drop(index_error, axis = 0)
    df = df.reset_index(drop=True)

    df.insert(0, 'time', date_column)
    df = df.set_index(pd.DatetimeIndex(df['time']))
    df = df.drop(['time', 'YY', 'MM', 'DD'],axis=1)

    #change obgect type to numeric if possible
    df = df.apply(pd.to_numeric, errors='coerce')
    
    if return_wrong_dates:
        return df, wrong_dates
    else:
        return df

def count_missing(df, is_print=True, return_df=False):
    """Print count of missed values in dataframe's columns between first and last valid value for the specific
    columns
    
    First and last NaN are omitted
    
    Parameters:
        df - pandas dataframe
    
    Return:
        None
    """
    
    nans_l = {}
    fir_valid = dict().fromkeys(df.columns)
    last_valid = dict().fromkeys(df.columns)
    
    for col in df.columns:
        ts = df[[col]].copy()
        fir_ix = ts.first_valid_index()
        if fir_ix is not None:
            fir_ix = ts.index.get_loc(fir_ix)
            last_ix = ts.index.get_loc(ts.last_valid_index())
            ts = ts.iloc[fir_ix:last_ix+1, :]
        #tuple for cnt nan and no-nan
        nans_l[col] = (ts.isna().sum().values[0], ts.notnull().sum().values[0])
        #to print first and last valid indexes
        fir_valid[col] = ts.first_valid_index()
        last_valid[col] = ts.last_valid_index()
    if is_print:    
        columns_names = 'Element | Count NaN | Count no-NaN | First valid date| Last valid date'
        print(' '*(len(columns_names)//3) + 'Some information about data')
        print('-'*len(columns_names))
        print(columns_names)
        print('-'*len(columns_names))
        for col, val in nans_l.items():
            fir_val_ix = str(fir_valid[col]).split(' ')[0]
            last_val_ix = str(last_valid[col]).split(' ')[0]
            cnt_nan = val[0]
            cnt_nonan = val[1]
            print(f'{col:<7}{cnt_nan:>8}{cnt_nonan:>15}{fir_val_ix:>18}{last_val_ix:>18}')

        print("\nP.S. valid date means that before or after this date there're only missing values.\n\
Of course nonvalid dates are omitted")
    if return_df:
        i = 0
        df = pd.DataFrame(columns=['Element', 'Count NaN', 'Count no-NaN', 'First valid date', 'Last valid date'])
        for col, val in nans_l.items():
            fir_val_ix = str(fir_valid[col]).split(' ')[0]
            last_val_ix = str(last_valid[col]).split(' ')[0]
            cnt_nan = val[0]
            cnt_nonan = val[1]
            
            df.loc[i] = [col, cnt_nan, cnt_nonan, fir_val_ix, last_val_ix]
            i += 1
            
        return df
    
def find_borders_nan_intervals(df, col):
    """Finding borders of NaN intervals
    
    First and last NaN are omitted
    
    Parameters:
        df - pandas dataframe
        col - column's name where to search borders of nan intervals
        
    Return:
        list of tuple where tuplu consist of:
            t[0] - left border, ix of 1-st nan
            t[1] - right border where first non-missing value was met
            t[2] - length of the interval
    Example:
        [1, 1, nan, nan, 3]
        t[0] = 2, t[1] = 4, t[2] = 2
    
    Note:
        Since for dataframe with several columns for some of them there can be nan first, we would start
        from first valid index for this specific column
    """
    
    nans_ix = []
    fir_ix, last_ix = 0, 0
    flag = False
    
    fir_val_ix = df.index.get_loc(df[col].first_valid_index())
    last_val_ix = df.index.get_loc(df[col].last_valid_index())

    for ix, val in enumerate(df[col].values):
        #to pass first nan values
        if ix < fir_val_ix:
            continue
        if ix > last_val_ix:
            break
        
        if np.isnan(val) and not flag:
            fir_ix = ix
            flag = True
        elif flag and not np.isnan(val):
            flag = False
            last_ix = ix
            nans_ix.append((fir_ix, last_ix, last_ix - fir_ix))
            fir_ix, last_ix = 0, 0
            
    return nans_ix

def count_frequency(l):
    """Count of repeated values in list
    
    Parameters:
        l - list of repeated values
    
    Return:
        list of tuple where:
            t[0] - value
            t[1] - count of freqquency
    """
    
    d = {}
    
    for v in l:
        d.setdefault(v, 0)
        d[v] += 1
        
    return sorted(d.items(), key=lambda x: x[0])

def get_metrics(y_true, y_pred, metrics=['rmse', 'mae', 'mape']):
    """Compute given metrics
    
    Parameters:
        y_true - actual values
        y_pred - forecasted values
        metrics - list of metrics or one metrics in string format, deault=['rmse', 'mae', 'mape']
        
    Return:
        dict with calculated metrics
    
    """
    allowed_metrics = ['rmse', 'mae', 'mape', 'corr', 'mse']
    
    d = {}
    
    if isinstance(metrics, str):
        metrics = metrics.split() #to remake in a list 
    
    for m in metrics:
        m = m.lower()
        
        if m not in allowed_metrics:
            raise ValueError(f'{m} is not found. Only {", ".join(allowed_metrics)} can be calculated')
            
        if m == 'mse':
            d[m.upper()] = np.mean((y_pred - y_true)**2)
        elif m == 'rmse':
            d[m.upper()] = np.mean((y_pred - y_true)**2)**0.5
        elif m == 'mae':
            d[m.upper()] = np.mean(np.abs(y_pred - y_true))
        elif m == 'mape':
            d[m.upper()] = np.mean(np.abs((y_pred - y_true) / y_true))
        elif m == 'corr':
            d[m.upper()] = np.corrcoef(y_pred, y_true)[0,1]
    
    return d
      