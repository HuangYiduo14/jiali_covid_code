import pandas as pd
import matplotlib.pyplot as plt

result_ind_period = pd.read_csv('exp_individual_periodical.csv')
result_ind_round = pd.read_csv('exp_individual_round.csv')
result_nstar1_period = pd.read_csv('exp_nstar1_periodical.csv')
result_nstar1_round = pd.read_csv('exp_nstar1_round.csv')
#result_nstar7_period = pd.read_csv('exp_nstar7_periodical.csv')
#result_nstar7_round = pd.read_csv('exp_nstar7_round.csv')

def plot_curve(col,is_cum,fig_number,title):
    plt.figure(fig_number)
    if is_cum:
        plt.plot(result_ind_round[col].cumsum(), label='individual, round')
        plt.plot(result_nstar1_round[col].cumsum(), label='n1, round')
        #plt.plot(result_nstar7_round[col].cumsum(), label='n7, round')
        plt.plot(result_ind_period[col].cumsum(), label='individual, period')
        plt.plot(result_nstar1_period[col].cumsum(), label='n1, period')
       # plt.plot(result_nstar7_period[col].cumsum(), label='n7, period')
    else:
        plt.plot(result_ind_round[col], label='individual, round')
        plt.plot(result_nstar1_round[col], label='n1, round')
        #plt.plot(result_nstar7_round[col], label='n7, round')
        plt.plot(result_ind_period[col], label='individual, period')
        plt.plot(result_nstar1_period[col], label='n1, period')
        #plt.plot(result_nstar7_period[col], label='n7, period')
    plt.legend()
    plt.title(title)
    plt.show()
#result_ind_period[['S','I','R','SQ','Q']].plot()
#plt.show()

plot_curve('S',False,0,'S')
plot_curve('I',False,1,'I')
plot_curve('n_star',False,2,'N')
plot_curve('TP',True,3,'TP')
plot_curve('FP',True,4,'FP')
plot_curve('TN',True,5,'TN')
plot_curve('FN',True,6,'FN')