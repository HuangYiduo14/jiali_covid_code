"""
do lowess and draw se/cpr and n_star graphs
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sir_simulation_for_se import get_n_star

n_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
cmap= {1:'purple', 2:'deeppink',3:'crimson', 4:'orange', 5:'gold',
            6:'yellow',7:'greenyellow',8:'lawngreen', 9:'green', 10:'lightseagreen', 15:'cyan',20: 'teal',
            25:'skyblue',30: 'blue'}
#n_list = [1]
se_up_pcr = pd.read_csv('100_pcr_se_up.csv')
se_down_pcr = pd.read_csv('100_pcr_se_down.csv')
se_up_antigen = pd.read_csv('100_antigen_se_up.csv')
se_down_antigen = pd.read_csv('100_antigen_se_down.csv')
parameter_set_pcr = {'LOD': 2, 'n_list': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30], 'sp': 0.986, 't_lead': 2}
parameter_set_antigen = {'LOD': 5, 'n_list': [1], 'sp': 0.984, 't_lead': 0}
df_cpr_up = get_n_star(se_up_pcr,parameter_set_pcr['n_list'], parameter_set_pcr['sp'])
df_cpr_down = get_n_star(se_down_pcr,parameter_set_pcr['n_list'], parameter_set_pcr['sp'])

fig = plt.figure(1, figsize=(15, 6))
ax11 = fig.add_subplot(2,2,1)
ax11.scatter(se_up_antigen['Unnamed: 0'],se_up_antigen['sep1_lws'],label='antigen',color = 'k')
for n in n_list:
    ax11.scatter(se_up_pcr['Unnamed: 0'],se_up_pcr['sep'+str(n)+'_lws'],label='PCR n='+str(n),color=cmap[n])
#ax11.set_xscale('log')
#ax11.set_xlim([max(se_up_pcr['p'].min(),se_up_antigen['p'].min()),min(se_up_pcr['p'].max(),se_up_antigen['p'].max())])
ax11.set_title('(A) $S_{ep}$ for increasing phase')
ax11.set_ylabel('Average sensitivity')

#ax1.set_xlabel('prevalence')
ax12 = fig.add_subplot(2,2,2)
ax12.scatter(se_up_antigen['Unnamed: 0'],se_up_antigen['sed1_lws'],label='antigen',color = 'k')
for n in n_list:
    ax12.scatter(se_up_pcr['Unnamed: 0'], se_up_pcr['sed'+str(n) + '_lws'], label='PCR n='+str(n),color=cmap[n])
#ax12.set_xscale('log')
#ax12.set_xlim([max(se_up_pcr['p'].min(),se_up_antigen['p'].min()), min(se_up_pcr['p'].max(), se_up_antigen['p'].max())])
ax12.set_title('(B) $S_{ed}$ for increasing phase')
ax12.set_ylabel('Average sensitivity')
ax12.legend(ncol=1,bbox_to_anchor=(1,1),loc='upper left')

#ax11.set_xscale('log')
#axvl.set_xscale('log')
#axvl.set_yscale('log')
#axvl.legend(ncol=1,bbox_to_anchor=(1,1),loc='upper left',fancybox=False)

ax21 = fig.add_subplot(2,2,3)
ax21.plot(se_down_antigen['p'],se_down_antigen['sep1_lws'],label='antigen',color = 'k')
for n in n_list:
    ax21.plot(se_down_pcr['p'],se_down_pcr['sep'+str(n)+'_lws'],label='PCR n='+str(n),color=cmap[n])
#ax1.legend(ncol=1,bbox_to_anchor=(1,1),loc='upper left')
ax21.set_xscale('log')
ax21.set_xlim([0,min(se_down_pcr['p'].max(),se_down_antigen['p'].max())])
ax21.set_title('(C) $S_{ep}$ for decreasing phase')
ax21.set_ylabel('Average sensitivity')

#ax1.set_xlabel('prevalence')
ax22 = fig.add_subplot(2,2,4)
ax22.plot(se_down_antigen['p'],se_down_antigen['sed1_lws'],label='antigen',color = 'k')
for n in n_list:
    ax22.plot(se_down_pcr['p'], se_down_pcr['sed'+str(n) + '_lws'], label='PCR n='+str(n),color=cmap[n])
ax22.set_xscale('log')
ax22.set_xlim([0, min(se_down_pcr['p'].max(), se_down_antigen['p'].max())])
ax22.set_title('(D) $S_{ed}$ for decreasing phase')
ax22.set_ylabel('Average sensitivity')





plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
fig = plt.figure(2, figsize=(15, 6))
ax3 = fig.add_subplot(1,2,1)
for n in n_list:
    ax3.plot(se_up_pcr['p'],df_cpr_up[n],label='n='+str(n), color=cmap[n])
#ax1.legend(ncol=1,bbox_to_anchor=(1,1),loc='upper left')
ax3.set_xlim([0.005,0.045])
ax3.set_ylim([0,100])
ax3.set_title('(A) CPR for increasing')
#ax3.set_xlabel('prevalence')
ax3.set_ylabel('Fraction')
ax3.legend(ncol=4)



ax4 = fig.add_subplot(1,2,2)
ax4.plot(se_up_pcr['p'],df_cpr_up['n_star'],color='tab:blue',label='up')
ax4.plot(se_down_pcr['p'],df_cpr_down['n_star'],color='tab:red',label='down')
ax4.set_xscale('log')
ax4.set_title('(B) Optimal pool size')
#ax4.set_xlabel('prevalence')
ax4.set_ylabel('Pool size')
ax4.tick_params(axis='y')
ax4.legend()
fig.text(0.5, 0.05,'Prevalence', ha='center',fontsize=15)


