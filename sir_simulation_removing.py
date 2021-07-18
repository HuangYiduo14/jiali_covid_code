from our_simulation import mcmc_result, CPR_group_test, calculate_v_load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import statsmodels.api as sm
lowess = sm.nonparametric.lowess
n_list = [1, 2, 3, 4, 5,6,7,8,9,10, 15,20,25,30]
color_table = {1:'b', 2:'r', 3:'g', 4:'c', 30:'m', 10:'y'}
df_se = pd.read_csv('se_data.csv')
#df_cpr = pd.read_csv('cpr1_data.csv')
df_se.sort_values(by='p',inplace=True)
#df_cpr.sort_values(by='p',inplace=True)


def lowess_data(n_list,df_se):
    for n in n_list:
        x = df_se['p'].values
        y = df_se[str(n)].values
        z = lowess(y,x,return_sorted=False,frac=1./10)
        df_se[str(n)+'_lws'] = z
        if n in list(color_table.keys()):
            #plt.scatter(x,y,alpha=0.4,color=color_table[n])
            plt.plot(x,z,color =color_table[n],label=n)
    plt.legend()
    return df_se

df_se = lowess_data(n_list,df_se)
#df_se.to_csv('se_data_lws.csv',index=False)
#df_se = pd.read_csv('se_data_lws.csv').drop('Unnamed: 0',axis=1)
n_test = df_se.shape[0]
cpr_matrix = np.zeros((n_test, len(n_list)))
for i, n in enumerate(n_list):
    if n==1:
        cpr_matrix[:,0] = 1./df_se[str(n)+'_lws'].values/df_se['p'].values
    else:
        se_vect = df_se[str(n)+'_lws'].values
        p_vect = df_se['p'].values
        cpr_matrix[:,i] = 1. / se_vect / df_se['1_lws'].values / p_vect * (1. / n + se_vect - se_vect * (1 - p_vect) ** n)
df_cpr = pd.DataFrame(cpr_matrix,columns=n_list)
df_cpr['p'] = df_se['p']
df_cpr.set_index('p',inplace=True)
df_cpr['n_star'] = df_cpr.idxmin(axis=1)
df_cpr['n_star'].plot()

print('n_star curve generated')