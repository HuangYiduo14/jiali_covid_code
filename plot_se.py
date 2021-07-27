"""
do lowess and draw se/cpr and n_star graphs
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
sp = 0.986
lowess = sm.nonparametric.lowess
plt.rcParams.update({'font.size':13})
plt.rcParams.update({'axes.labelpad':9})
plt.rc('font', family = 'Times New Roman')
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


n_test = df_se.shape[0]
cpr_matrix = np.zeros((n_test, len(n_list)))
for i, n in enumerate(n_list):
    if n==1:
        cpr_matrix[:,0] = 1./df_se[str(n)+'_lws'].values/df_se['p'].values
    else:
        se_vect = df_se[str(n)+'_lws'].values
        se_all_vect = df_all_se[str(n)+'_lws'].values
        p_vect = df_se['p'].values
        cpr_matrix[:,i] = 1. / se_all_vect / p_vect * (1. / n + se_vect - (se_vect+sp-1) * (1 - p_vect) ** n)
df_cpr = pd.DataFrame(cpr_matrix,columns=n_list)
df_cpr['p'] = df_se['p']
df_cpr.set_index('p',inplace=True)
df_cpr['n_star'] = df_cpr.idxmin(axis=1)


fig = plt.figure(1, figsize=(15, 6))
ax1 = fig.add_subplot(1,2,1)
ax1.plot(df_anti_se['p'],df_anti_se['1_lws'],label='antigen',color = 'k')
for n in n_list:
    ax1.plot(df_se['p'],df_se[str(n)+'_lws'],label='PCR n='+str(n))
#ax1.legend(ncol=1,bbox_to_anchor=(1,1),loc='upper left')
ax1.set_xscale('log')
ax1.set_title('(A) Sep for poll test only')
ax1.set_xlabel('prevalence')
ax2 = fig.add_subplot(1,2,2)
ax2.plot(df_anti_se['p'],df_anti_se['1_lws'],label='antigen',color = 'k')
for n in n_list:
    ax2.plot(df_all_se['p'], df_all_se[str(n) + '_lws'], label='PCR n='+str(n))
ax2.set_xscale('log')
ax2.set_title('(B) Sed for two test processes')
ax2.set_xlabel('prevalence')
ax2.legend(ncol=1,bbox_to_anchor=(1,1),loc='upper left',fancybox=False)

fig = plt.figure(2, figsize=(15, 6))
ax3 = fig.add_subplot(1,2,1)
for n in n_list:
    ax3.plot(df_se['p'],df_cpr[n],label='n='+str(n))
#ax1.legend(ncol=1,bbox_to_anchor=(1,1),loc='upper left')
ax3.set_xlim([0.1,0.5])
ax3.set_ylim([2.5,13])
ax3.set_title('(A) CPR')
ax3.set_xlabel('prevalence')
ax3.legend(ncol=4,bbox_to_anchor=(1,1))


ax4 = fig.add_subplot(1,2,2)
ax4.plot(df_se['p'],df_cpr['n_star'],color='tab:blue')
ax4.set_xscale('log')
ax4.set_title('(B) Optimal n')
ax4.set_xlabel('prevalence')
ax4.set_ylabel('n')
ax4.tick_params(axis='y', labelcolor='tab:blue')

ax5 = ax4.twinx()
names = df_cpr['n_star'].astype(str).values+'_lws'
ax5.plot(df_se['p'],[df_all_se.loc[index,names[index]] for index in df_all_se.index],color='tab:red')
ax5.set_xscale('log')
ax5.set_ylabel('Sed')
ax2.tick_params(axis='y', labelcolor='tab:red')




