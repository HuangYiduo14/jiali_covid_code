import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11})
plt.rcParams.update({'axes.labelpad': 9})
plt.rc('font', family='Times New Roman')


def plot_for_each_var(exp_names, data_list, variable_interested, x_var_interested, exp_interested,
                      subplot_titles,
                      subplot_x_names,
                      subplot_y_names,
                      markers,
                      xlims,
                      ylims):
    # each subplot is a var
    fig = plt.figure(figsize=(15, 6))
    subplot_number = len(variable_interested)
    width = 2
    height = int(np.ceil(subplot_number / width))
    sub_list = []
    for var_i, var in enumerate(variable_interested):
        sub_list.append(fig.add_subplot(height, width, var_i + 1))
        for exp_index, exp_i in enumerate(exp_interested):
            exp_name = exp_names[exp_i]
            if x_var_interested[exp_index] is None:
                sub_list[var_i].plot(data_list[exp_i][var],
                                     markers[exp_index], label=exp_name)
            else:
                sub_list[var_i].plot(data_list[exp_i][x_var_interested[exp_index]], data_list[exp_i][var], markers[exp_index],
                                     label=exp_name)
            sub_list[var_i].set_xlabel(subplot_x_names[var_i])
            sub_list[var_i].set_ylabel(subplot_y_names[var_i])
            if xlims[var_i] is None:
                sub_list[var_i].autoscale(axis='x')
            else:
                sub_list[var_i].set_xlim(xlims[var_i])
            if ylims[var_i] is None:
                sub_list[var_i].autoscale(axis='y')
            else:
                sub_list[var_i].set_ylim(ylims[var_i])
            sub_list[var_i].set_title(subplot_titles[var_i])
        if var_i == 0:
            sub_list[var_i].legend()
    return fig, sub_list


def plot_for_each_exp(exp_names, data_list, variable_interested, x_var_interested, exp_interested,
                      subplot_titles,
                      subplot_x_names,
                      subplot_y_names,
                      markers,
                      xlims,
                      ylims,label_list):
    # each subplot is a var
    fig = plt.figure(figsize=(15, 6))
    subplot_number = len(exp_interested)
    width = 2
    height = int(np.ceil(subplot_number / width))
    sub_list = []
    for exp_index, exp_i in enumerate(exp_interested):
        sub_list.append(fig.add_subplot(height, width, exp_index + 1))
        for var_i, var in enumerate(variable_interested):
            var_name = var
            if x_var_interested[var_i] is None:
                sub_list[exp_index].plot(data_list[exp_i][var],
                                         markers[exp_i], label=label_list[var_i])
            else:
                sub_list[exp_index].plot(data_list[exp_i][x_var_interested[var_i]], data_list[exp_i][var],
                                         markers[var_i], label=label_list[var_i])
            sub_list[exp_index].set_xlabel(subplot_x_names[exp_index])
            sub_list[exp_index].set_ylabel(subplot_y_names[exp_index])
            if xlims[exp_index] is None:
                sub_list[exp_index].autoscale(axis='x')
            else:
                sub_list[exp_index].set_xlim(xlims[exp_index])
            if ylims[exp_index] is None:
                sub_list[exp_index].autoscale(axis='y')
            else:
                sub_list[exp_index].set_ylim(ylims[exp_index])
            sub_list[exp_index].set_title(subplot_titles[exp_index])
        if exp_index == 0:
            sub_list[exp_index].legend()
    return fig, sub_list




################## change file name here #################################################################
exp_names = ['Flexiable PCR', 'Individual PCR', 'Antigen', 'No test']
file_names = ['exp_nstar1_periodical.csv',
              'exp_individual_periodical.csv',
              'exp_antigen_individual_periodical.csv',
              'exp_no_testing.csv']
data_list = [pd.read_csv(name) for name in file_names]
for df in data_list:
    df['total_I'] = df['I'] + df['Q'] + df['SQ'] + df['R']
    df['total_I_frac'] = (df['I'] + df['Q'] + df['SQ'] + df['R']) / (df['I'] + df['Q'] + df['SQ'] + df['R'] + df['S'])
    df['I_frac'] = df['I'] / (df['I'] + df['Q'] + df['SQ'] + df['R'] + df['S'])
    df['Q_frac'] = df['Q'] / (df['I'] + df['Q'] + df['SQ'] + df['R'] + df['S'])
    df['SQ_frac'] = df['SQ'] / (df['I'] + df['Q'] + df['SQ'] + df['R'] + df['S'])
    df['R_frac'] = df['R'] / (df['I'] + df['Q'] + df['SQ'] + df['R'] + df['S'])
    df['total_test_cumsum'] = df['number_of_total_tests'].cumsum()
    df['FN_CS'] = df['FN'].cumsum()
    df['FP_CS'] = df['FP'].cumsum()

fig_var, sub_list_var = plot_for_each_var(exp_names=exp_names, data_list=data_list,
                                          exp_interested=[0, 1, 2],  # len(exp_interested) = number of lines on each fig
                                          markers=['--', ':', 'm-.'],  # len(markers) =number of lines on each fig
                                          x_var_interested=['day', 'day', 'day'],
                                          # len(varible_interesed) = number of figs
                                          variable_interested=['total_I_frac', 'I_frac', 'Q_frac', 'SQ_frac'],
                                          subplot_titles=['(A) Total infections', '(B) Infection', '(C) Q after test',
                                                          '(D) Q after symptoms'],
                                          subplot_x_names=[None, None, None, None],
                                          subplot_y_names=['Fractions', 'Fractions', 'Fractions', 'Fractions'],
                                          xlims=[None, None, None, None],
                                          ylims=[None, None, None, None])
fig_var.text(0.5, 0.05,'Time (days)', ha='center',fontsize=15)

fig_exp, sub_list_exp = plot_for_each_exp(exp_names=exp_names,
                                          data_list=data_list,
                                          variable_interested=['total_I_frac', 'I_frac', 'Q_frac', 'SQ_frac'],
                                          # len(variable_interesed) =number of lines on each fig
                                          markers=['--', ':', 'm-.', 'k-'],  # len(markers) =number of lines on each fig,
                                          label_list=['total I','I','Q','SQ'],
                                          x_var_interested=['day', 'day', 'day','day'],
                                          # len(exp_interested) = number of figs
                                          exp_interested=[0, 1, 2],
                                          subplot_titles=['(A) Flexiable PCR', '(B) Individual PCR', '(C) Antigen'],
                                          subplot_x_names=[None, None, None],
                                          subplot_y_names=['Fractions', 'Fractions', 'Fractions'],
                                          xlims=[None, None, None],
                                          ylims=[None, None, None])
fig_exp.text(0.5, 0.05,'Time (days)', ha='center',fontsize=15)

# example 2. for fig 2
fig_var2, sub_list_var2 = plot_for_each_var(exp_names=exp_names, data_list=data_list,
                                          exp_interested=[0, 1, 2],  # len(exp_interested) = number of lines on each fig
                                          markers=['.', '.', '.'],  # len(markers) =number of lines on each fig
                                          x_var_interested=['day', 'day', 'day'],
                                          # len(varible_interesed) = number of figs
                                          variable_interested=['n_star', 'total_test_cumsum', 'FN_CS', 'FP_CS','NPV','PPV'],
                                          subplot_titles=['(A) n', '(B) total test', '(C) FN',
                                                          '(D) FP', '(E) NPV', '(F) PPV'],
                                          subplot_x_names=[None, None, None, None,None,None],
                                          subplot_y_names=['Counts','Counts','Counts','Counts','Prob', 'Prob'],
                                          xlims=[None, None, None, None,None,None],
                                          ylims=[None, None, None, None,None,None])
fig_var2.text(0.5, 0.05,'Time (days)', ha='center',fontsize=15)

fig_exp2, sub_list_exp2 = plot_for_each_exp(exp_names=exp_names,
                                          data_list=data_list,
                                          variable_interested=['FN_CS', 'FP_CS'],
                                          # len(variable_interesed) =number of lines on each fig
                                          markers=['--','m-.'],  # len(markers) =number of lines on each fig,
                                          label_list=['FN','FP'],
                                          x_var_interested=['day', 'day'],
                                          # len(exp_interested) = number of figs
                                          exp_interested=[0, 1, 2],
                                          subplot_titles=['(A) Flexiable PCR', '(B) Individual PCR', '(C) Antigen'],
                                          subplot_x_names=[None, None, None],
                                          subplot_y_names=['Counts','Counts','Counts'],
                                          xlims=[None, None, None],
                                          ylims=[None, None, None])
fig_exp2.text(0.5, 0.05,'Time (days)', ha='center',fontsize=15)