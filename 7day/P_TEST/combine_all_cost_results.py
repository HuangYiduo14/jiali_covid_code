import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def stack_bar_plot(ax,names,df):
    datas = []
    cats = df.columns.values[:-3]
    bottom = np.zeros(df.shape[1]-3)
    for name in names:
        current_data = df.loc[name].values[:-3]
        datas.append(current_data)
        ax.bar(cats, current_data, bottom = bottom, label=name)
        bottom += current_data
    return

def plot_for_names(names, k=0):
    p_starts = [0.1, 0.01, 0.001, 0.05, 0.005]
    cwd = os.getcwd()
    folders = glob.glob(cwd + '/p_start*')
    df_lists = []

    for i, folder in enumerate(folders):
        csv_file = folder + '/cost analysis.csv'
        df = pd.read_csv(csv_file, index_col=0)
        df_lists.append(pd.read_csv(csv_file))
        fig = plt.figure(k)
        ax = fig.add_subplot(1, 1, 1)
        stack_bar_plot(ax, names, df)
        ax.set_title(folder)
        ax.legend()
        k += 1
    plt.show()
    return k

k=0
#k = plot_for_names(['RNA extraction consumables','RT-PCR consumables','Antigen test kits',
#                    'Pool test setup labor','RNA extraction labor','RT-PCR labor','Reporting labor'],k)

#k = plot_for_names(['Total infections'],k)
#k = plot_for_names(['Testing period'],k)
k = plot_for_names(['Cost of one reduction in total infection ($ per person)'],k)

