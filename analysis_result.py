import pandas as pd
import matplotlib.pyplot as plt

result_ind_period = pd.read_csv('exp_individual_periodical.csv')
result_ind_round = pd.read_csv('exp_individual_round.csv')
result_nstar1_period = pd.read_csv('exp_nstar1_periodical.csv')
result_nstar1_round = pd.read_csv('exp_nstar1_round.csv')
result_nstar7_period = pd.read_csv('exp_nstar7_periodical.csv')
result_nstar7_round = pd.read_csv('exp_nstar7_round.csv')


def plot_curve(col,is_cum,fig_number,title):
    plt.figure(fig_number)
    if is_cum:
        plt.plot(result_ind_round[col].cumsum(), label='individual, round')
        plt.plot(result_nstar1_round[col].cumsum(), label='n1, round')
        plt.plot(result_nstar7_round[col].cumsum(), label='n7, round')
        plt.plot(result_ind_period[col].cumsum(), label='individual, period')
        plt.plot(result_nstar1_period[col].cumsum(), label='n1, period')
        plt.plot(result_nstar7_period[col].cumsum(), label='n7, period')
    else:
        plt.plot(result_ind_round[col], label='individual, round')
        plt.plot(result_nstar1_round[col], label='n1, round')
        plt.plot(result_nstar7_round[col], label='n7, round')
        plt.plot(result_ind_period[col], label='individual, period')
        plt.plot(result_nstar1_period[col], label='n1, period')
        plt.plot(result_nstar7_period[col], label='n7, period')
    plt.legend()
    plt.title(title)
    plt.show()

plot_curve('I',False,0,'I')
plot_curve('number_of_total_tests',True,1,'test number total')
plot_curve('TP',True,1,'TP')
plot_curve('FP',True,2,'FP')
plot_curve('TN',True,3,'TN')
plot_curve('FN',True,4,'FN')