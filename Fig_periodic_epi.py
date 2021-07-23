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
#detectable_load = 3
n_list  = [1, 2, 3, 4, 5, 10, 15, 30]
df_se = pd.read_csv('pcr3_se_data.csv')
df_se.sort_values(by='p',inplace=True)
df_se = lowess_data(n_list,df_se)
n_test = df_se.shape[0]
cpr_matrix = np.zeros((n_test, len(n_list)))
for i, n in enumerate(n_list):
    if n==1:
        cpr_matrix[:,0] = 1./df_se[str(n)+'_lws'].values/df_se['p'].values
    else:
        se_vect = df_se[str(n)+'_lws'].values
        p_vect = df_se['p'].values
        cpr_matrix[:,i] = 1. / se_vect / df_se['1_lws'].values / p_vect * (1. / n + se_vect - (se_vect+sp-1) * (1 - p_vect) ** n)
df_cpr = pd.DataFrame(cpr_matrix,columns=n_list)
df_cpr['p'] = df_se['p']
df_cpr.sort_values(by='p',inplace=True)
df_cpr.set_index('p',inplace=True)
df_cpr['n_star'] = df_cpr.idxmin(axis=1)

plt.rcParams.update({'font.size':13})
plt.rcParams.update({'axes.labelpad':9})
plt.rc('font', family = 'Times New Roman')
N=100000
for n in n_list:
    plt.plot(df_se[str(n)+'_lws'],label=n)
plt.legend()


fig = plt.figure(figsize=(15, 6))
ax_1 = fig.add_subplot(1, 2 , 1)
ax_2 = fig.add_subplot(1, 2, 2)

# plot: PCR over n
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('xx-small')
for n in n_list:
    ax_1.plot(df_cpr[n], label=n)
ax_1.set_title('(A) CPR')
ax_1.set_xlim([0.1,0.3])
ax_1.set_ylim([5,10])
#ax_1.set_ylabel('CPR')
ax_1.set_xlabel('p')
ax_1.legend(ncol=len(n_list)//5,title='pool size', bbox_to_anchor=(0.6,1),loc='upper left', prop=fontP)

# plot
ax_2.plot(df_cpr['n_star'])
ax_2.set_xscale('log')
#ax_2.set_ylabel('n')
ax_2.set_xlabel('p')
ax_2.set_title('(B) Optimal pool size')


# ax_5 = fig.add_subplot(2, 4, 5)
# ax_6 = fig.add_subplot(2, 4, 6)
# ax_7 = fig.add_subplot(2, 4, 7)

