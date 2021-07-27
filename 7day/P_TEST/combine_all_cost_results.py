import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def stack_bar_plot(ax,names,df):
    datas = []
    cats = df.columns.values[:-2]
    bottom = np.zeros(df.shape[1]-2)
    for name in names:
        current_data = df.loc[name].values[:-2]
        datas.append(current_data)
        ax.bar(cats, current_data, bottom = bottom, label=name)
        bottom += current_data
    ax.legend()
    return

p_starts = [0.1,0.01,0.001,0.05,0.005]
cwd = os.getcwd()
folders = glob.glob(cwd+'/p_start*')
df_lists = []
k = 0
for folder in folders:
    csv_file = folder+'/cost analysis.csv'
    df = pd.read_csv(csv_file,index_col=0)
    df_lists.append(pd.read_csv(csv_file))
    fig = plt.figure(k)
    ax = fig.add_subplot(1,1,1)
    stack_bar_plot(ax,['Total labor cost','Reagents and Consumables'],df)
    ax.set_title('start from '+str(p_starts[k]))
    k+=1
plt.show()