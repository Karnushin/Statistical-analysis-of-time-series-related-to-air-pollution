import numpy as np
import pandas as pd
import datetime

import warnings

import re
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp, kstest, chisquare
import scipy.stats as st

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

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

def get_metrics(y_true, y_pred, metrics=['rmse', 'mae', 'mape'], is_norm=False):
    """Compute given metrics
    
    Parameters:
        y_true - actual values
        y_pred - forecasted values
        metrics - list of metrics or one metrics in string format, deault=['rmse', 'mae', 'mape']
        is_norm - if True then normalize metrics dividing on (y_true_max - y_true_min)
        Acceptable for mse/rmse and mae
        
    Return:
        dict with calculated metrics
    
    """
    allowed_metrics = ['rmse', 'mae', 'mape', 'corr', 'mse', 'accuracy', 'roc auc', 'auc'
                       , 'precision', 'recall', 'f1', 'f1 score']
    
    d = {}
    
    if isinstance(metrics, str):
        metrics = metrics.split() #to remake in a list 
    
    for m in metrics:
        m = m.lower()
        
        if m not in allowed_metrics:
            raise ValueError(f'{m} is not found. Only {", ".join(allowed_metrics)} can be calculated')
            
            
        y_min, y_max = min(y_true), max(y_true)
        uniq_pred = len(np.unique(y_pred))
        if m == 'mse':
            d[m.upper()] = np.mean((y_pred - y_true)**2)
            if is_norm:
                d[m.upper()] /= (y_max - y_min)
        elif m == 'rmse':
            d[m.upper()] = np.mean((y_pred - y_true)**2)**0.5
            if is_norm:
                d[m.upper()] /= (y_max - y_min)
        elif m == 'mae':
            d[m.upper()] = np.mean(np.abs(y_pred - y_true))
            if is_norm:
                d[m.upper()] /= (y_max - y_min)
        elif m == 'mape':
            d[m.upper()] = np.mean(np.abs((y_pred - y_true) / y_true))
        elif m == 'corr':
            d[m.upper()] = np.corrcoef(y_pred, y_true)[0,1]
        
        elif m == 'auc' or m == 'roc auc':
            d[m.upper()] = roc_auc_score(y_true, y_pred)
        elif m == 'accuracy':
            if uniq_pred > 2:
                y_pred = np.where(y_pred > 0.5, 1, 0)
            d[m.upper()] = accuracy_score(y_true, y_pred)      
        elif m == 'precision':
            if uniq_pred > 2:
                y_pred = np.where(y_pred > 0.5, 1, 0)
            d[m.upper()] = precision_score(y_true, y_pred)
        elif m == 'recall':
            if uniq_pred > 2:
                y_pred = np.where(y_pred > 0.5, 1, 0)
            d[m.upper()] = recall_score(y_true, y_pred)
        elif m == 'f1' or m == 'f1 score':
            if uniq_pred > 2:
                y_pred = np.where(y_pred > 0.5, 1, 0)
            d[m.upper()] = f1_score(y_true, y_pred) 
          
    return d
def chi_square_on_test(test, distr, params):
    histo, bin_edges = np.histogram(test, bins='sqrt', density=False)
    n_bins = len(bin_edges) - 1
    f_ops = histo
    fdist = getattr(st, distr)
    cdf = fdist.cdf(bin_edges, *params)
    f_exp = len(test) * np.diff(cdf)
    
    return chisquare(f_ops, f_exp, ddof=len(params))

def get_best_distribution(data):
    train, test = train_test_split(data, test_size=0.2)
    dist_names = ['expon', 'genexpon', 'gamma','gengamma','invgamma','loggamma','lognorm']
            
    #dist_results_KS = []
    dist_results_Chi2 = []
    params_chi2 = {}
    #params_ks = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        params1 = dist.fit(train, floc=0)
        params2 = dist.fit(train, loc=0)
        
        S1, p_chi2_1 = chi_square_on_test(test, dist_name, params1)
        S2, p_chi2_2 = chi_square_on_test(test, dist_name, params2)
        S, p_chi2, param_chi2 = (S1, p_chi2_1, params1) if p_chi2_1 > p_chi2_2 else (S2, p_chi2_2, params2)
        
        params_chi2[dist_name] = param_chi2
        
        #D1, p_ks_1 = st.kstest(test, dist_name, args=params1)
        #D2, p_ks_2 = st.kstest(test, dist_name, args=params2)
        #D, p_ks, param_ks = (D1, p_ks_1, params1) if p_ks_1 > p_ks_2 else (D2, p_ks_2, params2)
        
        #params_ks[dist_name] = param_ks
        
        #print("p value on K-S for "+dist_name+" = "+str(p_ks))
        #print("p value on Chi2 for "+dist_name+" = "+str(p_chi2))
        #dist_results_KS.append((dist_name, p_ks))
        dist_results_Chi2.append((dist_name,p_chi2))
    
    #select the best fitted distribution on Chi2
    best_dist_chi2, best_p_chi2 = (max(dist_results_Chi2, key=lambda item: item[1]))
    #select the best fitted distribution on KS
    #best_dist_ks, best_p_ks = (max(dist_results_KS, key=lambda item: item[1]))
    
    f1, f2 = False, False
    it_ks, it_chi2 = 0,0
    #if best_p_ks < 0.05:
    #    f1 = True
    #    l1_dist_param = {}
    #    l1_dist_pv = []
    #    while p_ks < 0.05 and it_ks < 2:
    #        train, test = train_test_split(data, test_size=0.1)
    #        for dist_name in dist_names:
    #            name = dist_name + '_' + str(it_ks)
    #            dist = getattr(st, dist_name)
    #            params1 = dist.fit(train, floc=0)
    #            params2 = dist.fit(train, loc=0)
    #            D1, p_ks_1 = st.kstest(test, dist_name, args=params1)
    #            D2, p_ks_2 = st.kstest(test, dist_name, args=params2)
    #            D, p_ks, param_ks = (D1, p_ks_1, params1) if p_ks_1 > p_ks_2 else (D2, p_ks_2, params2)
    #            l1_dist_param[name] = param_ks
    #            l1_dist_pv.append((name, p_ks))
    #        it_ks += 1
    #        l1_best_dist, l1_best_pv = (max(l1_dist_pv, key=lambda item: item[1]))
    #    if l1_best_pv > best_p_ks:
    #        best_p_ks = l1_best_pv
    #        best_dist_ks = re.split('[_]', l1_best_dist)[0]
    #        new_params_ks = l1_dist_param[l1_best_dist]
    #    else:
    #        f1 = False

    if best_p_chi2 < 0.05:
        f2 = True
        l2_dist_param = {}
        l2_dist_pv = []
        while p_chi2 < 0.05 and it_chi2 < 2:
            train, test = train_test_split(data, test_size=0.2)
            for dist_name in dist_names:
                name = dist_name + '_' + str(it_chi2)
                dist = getattr(st, dist_name)
                params1 = dist.fit(train, floc=0)
                params2 = dist.fit(train, loc=0)
                S1, p_chi2_1 = chi_square_on_test(test, dist_name, params1)
                S2, p_chi2_2 = chi_square_on_test(test, dist_name, params2)
                S, p_chi2, param_chi2 = (S1, p_chi2_1, params1) if p_chi2_1 > p_chi2_2 else (S2, p_chi2_2, params2)
                l2_dist_param[name] = param_chi2
                l2_dist_pv.append((name, p_chi2))
            it_chi2 += 1
            l2_best_dist, l2_best_pv = (max(l2_dist_pv, key=lambda item: item[1]))
        if l2_best_pv > best_p_chi2:
            best_p_chi2 = l2_best_pv
            best_dist_chi2 = re.split('[_]', l2_best_dist)[0]
            new_params_chi2 = l2_dist_param[l2_best_dist]
        else:
            f2 = False
            
    print('\nChi2')
    print("Best fitting distribution: "+str(best_dist_chi2))
    print("Best p value: "+ str(best_p_chi2))
    if f2 == False:
        print("Parameters for the best fit: "+ str(params_chi2[best_dist_chi2]))
    elif best_p_chi2 >= 0.05:
        print("Parameters for the best fit: "+ str(new_params_chi2))
    elif best_p_chi2 <= 0.05:
        print('!!!Optimal Distribution on Chi Square was not found!!!')
        
    #print('\nK-S')
    #print("Best fitting distribution: "+str(best_dist_ks))
    #print("Best p value: "+ str(best_p_ks))
    #if f1 == False:
    #    print("Parameters for the best fit: "+ str(params_ks[best_dist_ks]))
    #elif best_p_ks >= 0.05:
    #    print("Parameters for the best fit: "+ str(new_params_ks))
    #elif best_p_ks <= 0.05:
    #    print('!!!Optimal Distribution on K-S was not found!!!')
    
    
    if f2 == False:
        return best_dist_chi2, params_chi2[best_dist_chi2], best_p_chi2
    else:
        return best_dist_chi2, new_params_chi2, best_p_chi2
    
    #закомментил ks test => то, что ниже не имеет смысла
    
    if f1 == False and f2 == False:
        return best_dist_ks, params_ks[best_dist_ks], best_p_ks, \
               best_dist_chi2, params_chi2[best_dist_chi2], best_p_chi2
    elif f1 == True and f2 == False:
        return best_dist_ks, new_params_ks, best_p_ks, \
               best_dist_chi2, params_chi2[best_dist_chi2], best_p_chi2
    elif f1 == False and f2 == True:
        return best_dist_ks, params_ks[best_dist_ks], best_p_ks, \
               best_dist_chi2, new_params_chi2, best_p_chi2
    elif f1 == True and f2 == True:
        return best_dist_ks, new_params_ks, best_p_ks, \
               best_dist_chi2, new_params_chi2, best_p_chi2
    
def distribution_by_season(values, element, params_dict, is_print_info=False):
            
    if is_print_info:
        print('Гистограмма для выборки {0}, сезон - {1}'.format(element, season))
        print('Поверх нее строятся теоритические функции плотности вероятности с вычисленными параметрами')
    
    fig = plt.figure(figsize=(8,8))
    
    x = np.linspace(-0.2, max(values),1000)
    colors = ['red', 'blue', 'forestgreen', 'orange']
    
    for i, distr in enumerate(list(params_dict.keys())):
        if distr == 'f':
            info_distr = distr + ' pdf\n dfn= {0:.4g},\n dfd= {1:.4g},\n loc= {2:.4g},\n scale= {3:.4g}'. \
            format(params_dict[distr][0],params_dict[distr][1],params_dict[distr][2],params_dict[distr][3])
        if distr == 'expon':
            info_distr = distr + ' pdf\n loc= {0:.4g},\n scale= {1:.4g}'. \
            format(params_dict[distr][0],params_dict[distr][1])
        if distr == 'genexpon':
            info_distr = distr + ' pdf\n a= {0:.4g},\n b= {1:.4g},\n c={2:.4g},\n loc= {3:.4g},\n scale= {4:.4g}'. \
            format(params_dict[distr][0],params_dict[distr][1],params_dict[distr][2],params_dict[distr][3],params_dict[distr][4])
        if distr == 'gamma':
            info_distr = distr + ' pdf\n shape= {0:.4g},\n loc= {1:.4g},\n scale= {2:.4g}'. \
            format(params_dict[distr][0],params_dict[distr][1], params_dict[distr][2])
        if distr == 'gengamma':
            info_distr = distr + ' pdf\n a= {0:.4g},\n c= {1:.4g},\n loc= {2:.4g},\n scale= {3:.4g}'. \
            format(params_dict[distr][0],params_dict[distr][1],params_dict[distr][2],params_dict[distr][3])
        if distr == 'invgamma':
            info_distr = distr + ' pdf\n shape= {0:.4g},\n loc= {1:.4g},\n scale= {2:.4g}'. \
            format(params_dict[distr][0],params_dict[distr][1],params_dict[distr][2])
        if distr == 'loggama':
            info_distr = distr + ' pdf\n c= {0:.4g},\n loc= {1:.4g},\n scale= {2:.4g}'. \
            format(params_dict[distr][0],params_dict[distr][1],params_dict[distr][2])
        if distr == 'lognorm':
            info_distr = distr + ' pdf\n shape= {0:.4g},\n loc= {1:.4g},\n scale= {2:.4g}'. \
            format(params_dict[distr][0],params_dict[distr][1],params_dict[distr][2])
        if distr == 'mielke':
            info_distr = distr + ' pdf\n k= {0:.4g},\n s= {1:.4g},\n loc= {2:.4g},\n scale= {3:.4g}'. \
            format(params_dict[distr][0],params_dict[distr][1],params_dict[distr][2],params_dict[distr][3])
        if distr == 'ncf':
            info_distr = distr + ' pdf\n dfn= {0:.4g},\n dfd= {1:.4g},\n nc= {2:.4g},\n loc= {3:.4g},\n scale= {4:.4g}'. \
            format(params_dict[distr][0],params_dict[distr][1],params_dict[distr][2],params_dict[distr][3],params_dict[distr][4])
        if distr == 'powerlognorm':
            info_distr = distr + ' pdf\n c= {0:.4g},\n s= {1:.4g},\n loc= {2:.4g},\n scale= {3:.4g}'. \
            format(params_dict[distr][0],params_dict[distr][1],params_dict[distr][2],params_dict[distr][3])
        if distr == 'recipinvgauss':
            info_distr = distr + ' pdf\n mu= {0:.4g},\n loc= {1:.4g},\n scale= {2:.4g}'. \
            format(params_dict[distr][0],params_dict[distr][1],params_dict[distr][2])
        if distr == 'johnsonsu':
            info_distr = distr + ' pdf\n a= {0:.4g},\n b= {1:.4g},\n loc= {2:.4g},\n scale= {3:.4g}'. \
            format(params_dict[distr][0],params_dict[distr][1],params_dict[distr][2],params_dict[distr][3])
        if distr == 'dgamma':
            info_distr = distr + ' pdf\n a= {0:.4g},\n loc= {1:.4g},\n scale= {2:.4g}'. \
            format(params_dict[distr][0],params_dict[distr][1],params_dict[distr][2])
        if distr == 'foldcauchy':
            info_distr = distr + ' pdf\n shape= {0:.4g}'. \
            format(params_dict[distr][0])
        if distr == 'loglaplace':
            info_distr = distr + ' pdf\n shape= {0:.4g},\n loc= {1:.4g},\n scale= {2:.4g}'. \
            format(params_dict[distr][0],params_dict[distr][1],params_dict[distr][2])
        if distr == 'wald':
            info_distr = distr + ' pdf\n loc= {0:.4g},\n scale= {1:.4g}'. \
            format(params_dict[distr][0],params_dict[distr][1])
        if distr == 'genhalflogistic':
            info_distr = distr + ' pdf\n shape= {0:.4g},\n loc= {1:.4g},\n scale= {2:.4g}'. \
            format(params_dict[distr][0],params_dict[distr][1],params_dict[distr][2])
        if distr == 'invgauss':
            info_distr = distr + ' pdf\n mu= {0:.4g},\n loc= {1:.4g},\n scale= {2:.4g}'. \
            format(params_dict[distr][0],params_dict[distr][1],params_dict[distr][2])
        if distr == 'alpha':
            info_distr = 'dsfdfsdf'
            
        fdist = getattr(st, distr)
        y1 = fdist.pdf(x, *params_dict[distr])
        plt.plot(x, y1, color=colors[i], linewidth=2, label=info_distr)
        
    plt.hist(values, bins='sqrt', density=True,
            # label='{0}\n size of sample: {1}'.format(season,len(values))
            )
    plt.title(element, fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.ylabel('$Density$', fontsize=10)
    plt.xlabel('$x$', fontsize=10)
    plt.show()
    #return fig
    
def samples_split_by_order(X, y=None, date_col='time', date=None, split_by_date=False, sec_size=0.2):
    '''Divide samples by order
    
    Parameters:
        X - dataframe
        y - series or dataframe
        X, y must be already ordered by some order
        If y isn't given thеn only X is divided
        
        date_col - name of column that indicate monthly date, default='month_camp'
        
        date - 1-st sample created strictly till (before!) this date, 2-d after including the 'date' if 'date' is not None
        
        split_by_date - bool, split samples by date if True (date must be given)
        
        sec_size - fraction of second size of sample
        
        If split_by_date is True then division makes by date and column "month_camp must be" else by frac
        
    Return:
        samples divided by order in format X_first, X_second (optionally y_first, y_second)
    '''
    
    if date is not None and not split_by_date:
        raise ValueError('Get date but parameter split_by_date is False. It should be True')
    elif date is None and split_by_date:
        raise ValueError('Get splitby_date=True but date is not found')
    
    cnt_rows_1 = np.where(X[date_col]>=date)[0][0] if split_by_date else int(X.shape[0] * (1 - sec_size))
    
    X_fir, X_sec = X.iloc[:cnt_rows_1, :], X.iloc[cnt_rows_1:, :]
    
    return (X_fir, X_sec, y[:cnt_rows_1], y[cnt_rows_1:]) if y is not None else (X_fir, X_sec)
      