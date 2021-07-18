import pandas as pd
import matplotlib.pyplot as plt

result_ind_period = pd.read_csv('exp_individual_periodical.csv')
result_ind_round = pd.read_csv('exp_individual_round.csv')
result_nstar1_period = pd.read_csv('exp_nstar1_periodical.csv')
result_nstar1_round = pd.read_csv('exp_nstar1_round.csv')
result_nstar7_period = pd.read_csv('exp_nstar7_periodical.csv')
result_nstar7_round = pd.read_csv('exp_nstar7_round.csv')

plt.figure(0)
plt.plot(result_ind_round['S'],label='individual, round')
plt.plot(result_nstar1_round['S'],label='n1, round')
plt.plot(result_nstar7_round['S'],label='n7, round')
plt.plot(result_ind_period['S'],label='individual, period')
plt.plot(result_nstar1_period['S'],label='n1, period')
plt.plot(result_nstar7_period['S'],label='n7, period')
plt.legend()
plt.title('S curve for all')

plt.figure(1)
plt.plot(result_ind_round['I'],label='individual, round')
plt.plot(result_nstar1_round['I'],label='n1, round')
plt.plot(result_nstar7_round['I'],label='n7, round')
plt.plot(result_ind_period['I'],label='individual, period')
plt.plot(result_nstar1_period['I'],label='n1, period')
plt.plot(result_nstar7_period['I'],label='n7, period')
plt.legend()
plt.title('I curve for all')


plt.figure(2)
plt.plot(result_ind_round['number_of_total_tests'].cumsum(),label='individual, round')
plt.plot(result_nstar1_round['number_of_total_tests'].cumsum(),label='n1, round')
plt.plot(result_nstar7_round['number_of_total_tests'].cumsum(),label='n7, round')
plt.plot(result_ind_period['number_of_total_tests'].cumsum(),label='individual, period')
plt.plot(result_nstar1_period['number_of_total_tests'].cumsum(),label='n1, period')
plt.plot(result_nstar7_period['number_of_total_tests'].cumsum(),label='n7, period')
plt.legend()
plt.title('test number total for all')