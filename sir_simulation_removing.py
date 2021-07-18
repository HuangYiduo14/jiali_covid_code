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
df_cpr = pd.read_csv('cpr1_data.csv')
df_se.sort_values(by='p',inplace=True)
df_cpr.sort_values(by='p',inplace=True)


def lowess_data(n_list,df_se):
    for n in n_list:
        x = df_se['p'].values
        y = df_se[str(n)].values
        z = lowess(y,x,return_sorted=False,frac=1./10)
        if n in list(color_table.keys()):
            plt.scatter(x,y,alpha=0.4,color=color_table[n])
            plt.plot(x,z,color =color_table[n],label=n)
    plt.legend()
    return

def ols_curve_fit_cpr(n_list,df_cpr):
    for n in n_list:
        # we will discard high variance values because variance is high
        df_cpr_cp = df_cpr.copy()
        df_cpr_cp = df_cpr_cp.loc[df_cpr_cp['p']>1e-3]
        x = df_cpr_cp['p'].values
        y = df_cpr_cp[str(n)].values
        func = lambda x,a,b,c: 1./a/x*(b+c*(1-x)**n)
        parm_guess,_ = curve_fit(func,x,y,p0=[0.8**2,0.8,-0.8])
        a0,b0,c0 = parm_guess
        z = func(df_cpr['p'],a0,b0,c0)
        plt.scatter(df_cpr['p'],df_cpr[str(n)],alpha=0.4,color=color_table[n])
        plt.plot(df_cpr['p'],z,color = color_table[n])
    return df_cpr

lowess_data(n_list,df_se)
#df_cpr_down = lowess_data(n_list,df_cpr_down)