# Script containing utility and analysis functions

# Author: Virginia Casasnovas
# Date: 2021-10-07

import numpy as np
import pandas as pd
import math
import scipy.stats
import matplotlib.pyplot as plt

### General utility functions ###

def angle_wrap(a):

    if a.size == 1:
        if not np.isnan(a):
            b = (360+a) % 360
        else:
            b = np.nan
    else:
        b = np.empty(a.shape)
        for c, i in enumerate(a):
            if not np.isnan(i):
                b[c] = (360+i) % 360
            else:
                b[c] = np.nan
    
    return b


def angle_between(v1, v2):
    
    b = np.empty((v1.shape[0],1))
    for i in range(0,v1.shape[0]):
        vector1 = v1[i,:]
        vector2 = v2[i,:]
        
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        
        magnitude1 = math.sqrt(sum(a**2 for a in vector1))+0.00001
        magnitude2 = math.sqrt(sum(b**2 for b in vector2))+0.00001
        
        cos_theta = dot_product / (magnitude1 * magnitude2)
        
        angle_rad = math.acos(cos_theta)
        
        cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
        if cross_product < 0:
            angle_rad = -angle_rad
        
        b[i] = math.degrees(angle_rad)
    
    return b


def txt2df(file_path):

    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            data_list.append(line.split())
        
    data_df = pd.DataFrame(data_list[1:], columns=data_list[0])
    
    return data_df


def str2bool(v):

    b = []
    for i in v:
        b.append(i.lower() in ("yes", "true", "t", "1", "True"))

    return b

def change_mky_name(df):
    """
    Changes name of monkey in DataFrame.
    """  

    name_map = {'yod': 'MY', 'zep': 'MZ'}
    df['monkey'] = df['monkey'].replace(name_map)
    
    return df

### Uncertainty-related utility functions ###

def stat_test(data, group, paired):
    
    g1 = np.where(group == 'CRS')[0]
    g2 = np.where(group == 'CLD')[0]
    
    data_val = data.values
    d1 = data_val[g1]
    d2 = data_val[g2]
    
    n1 = scipy.stats.shapiro(d1)
    n2 = scipy.stats.shapiro(d2)
    v1 = scipy.stats.levene(d1, d2)
    
    print('SW test: CRS stat={}, p={}; CLD stat={}, p={}'.format(np.round(n1.statistic, 3), np.round(n1.pvalue, 4), np.round(n2.statistic, 3), np.round(n2.pvalue, 4)))
    print('Lev test: stat={}, p={}'.format(np.round(v1.statistic, 3), np.round(v1.pvalue, 4)))
    print('CRS: {}±{}; CLD: {}±{}'.format(np.round(np.mean(d1), 3), np.round(np.std(d1), 3), np.round(np.mean(d2), 3), np.round(np.std(d2), 3)))
          
    if paired:
        # if (n1.pvalue>0.05) & (n2.pvalue>0.05) & (v1.pvalue>0.05):
        #     res = scipy.stats.ttest_rel(d1,d2)
        #     print('Paired t-test: stat={}, p={}'.format(np.round(res.statistic,3),np.round(res.pvalue,5)))     
        # else:
        res = scipy.stats.wilcoxon(d1, d2)
        print('Paired sign-rank: stat={}, p={}'.format(np.round(res.statistic, 3), np.round(res.pvalue,5)))  
    else:
        if (n1.pvalue > 0.05) & (n2.pvalue > 0.05) & (v1.pvalue > 0.05):
            res = scipy.stats.ttest_ind(d1, d2)
            print('Unpaired t-test: stat={}, p={}'.format(np.round(res.statistic, 3), np.round(res.pvalue, 5)))     
        else:
            res = scipy.stats.ranksums(d1, d2)
            print('Unpaired rank-sum: stat={}, p={}'.format(np.round(res.statistic, 3), np.round(res.pvalue, 5)))  
        
    print('')
    
def missing_unc(df, m):
    
    gr = df['uncertainty'].values
    x = df[m].values
    
    x1 = np.where((gr == 'CRS') & (np.isnan(x)))
    x2 = np.where((gr == 'CLD') & (np.isnan(x)))
    
    
    if np.where(gr == 'CRS')[0][0] < np.where(gr == 'CLD')[0][0]:
        na1 = np.append(x1, x2-sum(gr == 'CRS'))
        na2 = np.append(x2, x1+sum(gr == 'CRS'))

    else:
        na1 = np.append(x1, x2+sum(gr == 'CRS'))
        na2 = np.append(x2, x1-sum(gr == 'CRS'))  
    
    na = np.concatenate((na1, na2))

    return na
  

def equal_cond_df(df):

    new_index = pd.MultiIndex.from_product([df[i].unique() for i in list(df.columns[:-1])], names=df.columns[:-1])

    df1 = df.groupby(list(df.columns[:-1]))[df.columns[-1]].mean().reindex(new_index).reset_index()
    na = missing_unc(df1, df.columns[-1])
    
    df1 = df1.drop(na)
    
    return df1


### Plotting-related utility functions ###

def add_median(x, **kwargs):
    plt.axvline(x.median(), c='k', ls='-', lw=1.5)

        

    
    
    
    
    
    
    
    
    
    
    
    