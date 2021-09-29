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

n_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
cmap= {1:'purple', 2:'deeppink',3:'crimson', 4:'orange', 5:'gold',
            6:'yellow',7:'greenyellow',8:'lawngreen', 9:'green', 10:'lightseagreen', 15:'cyan',20: 'teal',
            25:'skyblue',30: 'blue'}
#n_list = [1]
df_all_se = pd.read_csv('se_d_pcr_data.csv')
df_all_se.sort_values(by='p',inplace=True)
df_all_se = lowess_data(n_list,df_all_se)

df_se = pd.read_csv('se_p_pcr_data.csv')
df_se.sort_values(by='p',inplace=True)
df_se = lowess_data(n_list,df_se)

df_anti_se = pd.read_csv('se_d_antigen_data.csv')
df_anti_se.sort_values(by='p',inplace=True)
df_anti_se = lowess_data([1],df_anti_se)

#df_vl = pd.read_csv('ind_viral_load_data.csv')
#df_vl.sort_values(by='p',inplace=True)
vl_data = pd.read_csv('ind_v_load.csv')
vl_data = vl_data['log10vload'].values
vl_data_non_zero = vl_data[vl_data>0]
percentage = 1-len(vl_data_non_zero)/len(vl_data)

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
ax1 = fig.add_subplot(2,2,3)
ax1.plot(df_anti_se['p'],df_anti_se['1_lws'],label='antigen',color = 'k')
for n in n_list:
    ax1.plot(df_se['p'],df_se[str(n)+'_lws'],label='PCR n='+str(n),color=cmap[n])
#ax1.legend(ncol=1,bbox_to_anchor=(1,1),loc='upper left')
ax1.set_xscale('log')
ax1.set_title('(B) Sensitivity for pooled samples with different pool sizes (first stage)')
ax1.set_ylabel('Average sensitivity')

#ax1.set_xlabel('prevalence')
ax2 = fig.add_subplot(2,2,4)
ax2.plot(df_anti_se['p'],df_anti_se['1_lws'],label='antigen',color = 'k')
for n in n_list:
    ax2.plot(df_all_se['p'], df_all_se[str(n) + '_lws'], label='PCR n='+str(n),color=cmap[n])
ax2.set_xscale('log')
ax2.set_title('(C) Overall sensitivity for two-stage pooled testing')
ax2.set_ylabel('Average sensitivity')

#ax2.set_xlabel('prevalence')
ax2.legend(ncol=1,bbox_to_anchor=(1,1),loc='center left',fancybox=False)

axvl = fig.add_subplot(2,1,1)
#p = df_vl['p'].values
#vl = df_vl[str(1)]
#a = np.polyfit(np.log(p),np.log(vl),1)

axvl.hist(vl_data,density=True,bins=1000,cumulative=True,histtype='step')
axvl.text(10,0.1,'{0:.2f}% of the patients have 0 viral load'.format(100*percentage))
axvl.plot([3,3],[0,1.1],label='$LOD=10^3$',color='red')
axvl.plot([5,5],[0,1.1],label='$LOD=10^5$',color='blue')
axvl.text(3.1,0.409124,'(3,0.429)',color='red')
axvl.text(5.1,0.64252,'(5,0.663)',color='blue')
axvl.set_ylim(0,1.01)
axvl.set_xlim(0,15)
axvl.legend()
axvl.set_ylabel('Probability')
axvl.set_xlabel('log10 (viral load (copies/ml))')
#axvl.set_xscale('log')
#axvl.set_yscale('log')
axvl.set_title('(A) Cumulative distribution function of viral load for infected patients')
fig.text(0.5, 0.05,'Prevalence', ha='center',fontsize=15)
#axvl.legend(ncol=1,bbox_to_anchor=(1,1),loc='upper left',fancybox=False)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
fig = plt.figure(2, figsize=(15, 6))
ax3 = fig.add_subplot(1,2,1)
for n in n_list:
    ax3.plot(df_se['p'],df_cpr[n],label='n='+str(n), color=cmap[n])
#ax1.legend(ncol=1,bbox_to_anchor=(1,1),loc='upper left')
ax3.set_xlim([0.1,0.5])
ax3.set_ylim([2.5,13])
ax3.set_title('(A) CPR')
#ax3.set_xlabel('prevalence')
ax3.set_ylabel('Fraction')
ax3.legend(ncol=4)


ax4 = fig.add_subplot(1,2,2)
ax4.plot(df_se['p'],df_cpr['n_star'],color='tab:blue')
ax4.set_xscale('log')
ax4.set_title('(B) Optimal pool size')
#ax4.set_xlabel('prevalence')
ax4.set_ylabel('Pool size')
ax4.tick_params(axis='y', labelcolor='tab:blue')

ax5 = ax4.twinx()
names = df_cpr['n_star'].astype(str).values+'_lws'
ax5.plot(df_se['p'],[df_all_se.loc[index,names[index]] for index in df_all_se.index],color='tab:red')
ax5.set_xscale('log')
ax5.set_ylabel('Overall sensitivity for two-stage tests')
ax5.set_ylim([0,1])
ax5.tick_params(axis='y', labelcolor='tab:red')

fig.text(0.5, 0.05,'Prevalence', ha='center',fontsize=15)


