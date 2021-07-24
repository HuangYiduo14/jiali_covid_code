"""
do lowess and draw se/cpr and n_star graphs
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
sp = 0.986
lowess = sm.nonparametric.lowess

def lowess_data(n_list,df_se):
    # fit curve se
    for n in n_list:
        x = df_se['p'].values
        y = df_se[str(n)].values
        z = lowess(y,x,return_sorted=False,frac=1./10)
        z[z>1.] = 1.
        df_se[str(n)+'_lws'] = z
    return df_se
n_list  = [1, 2, 3, 4, 5, 10, 15, 30]
#n_list = [1]
df_all_se = pd.read_csv('pcr_se_all_data.csv')
df_all_se.sort_values(by='p',inplace=True)
df_all_se = lowess_data(n_list,df_all_se)

df_se = pd.read_csv('pcr_se_data.csv')
df_se.sort_values(by='p',inplace=True)
df_se = lowess_data(n_list,df_se)

df_anti_se = pd.read_csv('anti3_se_data.csv')
df_anti_se.sort_values(by='p',inplace=True)
df_anti_se = lowess_data([1],df_anti_se)


plt.plot(df_anti_se['p'],df_anti_se['1_lws'],label='anti',color = 'k')
#for n in n_list:
#    plt.plot(df_se['p'],df_se[str(n)+'_lws'],label=n,linestyle='dashed')
for n in n_list:
    plt.plot(df_all_se['p'], df_all_se[str(n) + '_lws'], label=n)

plt.legend(ncol=len(n_list)//5,bbox_to_anchor=(1,1),loc='upper left', )

# plot: PCR over n
from matplotlib.font_manager import FontProperties
# ax_5 = fig.add_subplot(2, 4, 5)
# ax_6 = fig.add_subplot(2, 4, 6)
# ax_7 = fig.add_subplot(2, 4, 7)

