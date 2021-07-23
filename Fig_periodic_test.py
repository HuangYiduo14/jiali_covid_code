import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':13})
plt.rcParams.update({'axes.labelpad':9})
plt.rc('font', family = 'Times New Roman')


SIM_i_alpha = pd.read_csv('.//Period-7days/exp_individual_periodical.csv')
SIM_g_alpha = pd.read_csv('.//Period-7days/exp_nstar1_periodical.csv')
SIM_0_alpha = pd.read_csv('.//Period-7days/exp_no_testing.csv')
SIM_antigen_alpha = pd.read_csv('.//Period-7days/exp_antigen_individual_periodical.csv')


fig = plt.figure(figsize=(15, 9))

ax_1 = fig.add_subplot(3, 2, 1)
ax_2 = fig.add_subplot(3, 2, 2)
ax_3 = fig.add_subplot(3, 2, 3)
ax_4 = fig.add_subplot(3, 2, 4)
ax_5 = fig.add_subplot(3, 2, 5)
ax_6 = fig.add_subplot(3, 2, 6)



ax_1.plot(SIM_g_alpha.day, SIM_g_alpha.n_star, label='PCR Flexible Testing', linestyle='dashed')
ax_1.plot(SIM_i_alpha.day, SIM_i_alpha.n_star, label='PCR Individual Testing', linestyle='dotted')
ax_1.plot(SIM_antigen_alpha.day, SIM_antigen_alpha.n_star, label='Antigen Individual Testing', linestyle='dashdot',color ='magenta')

ax_1.set_ylabel('Count')
ax_1.set_ylim(-1, 33)
ax_1.set_title('(C) Optimal pool sizes')

ax_2.plot(SIM_g_alpha.day, SIM_g_alpha.number_of_total_tests.cumsum(), label='FT_TC', linestyle='dashed')
ax_2.plot(SIM_i_alpha.day, SIM_i_alpha.number_of_total_tests.cumsum(), label='IT_TC', linestyle='dotted')
ax_2.plot(SIM_antigen_alpha.day, SIM_antigen_alpha.number_of_total_tests.cumsum(), label='IT_TC', linestyle='dashdot',color ='magenta')

ax_2.set_ylabel('Count')
ax_2.set_title('(D) Total tests')

ax_3.plot(SIM_g_alpha.day, SIM_g_alpha.FN.cumsum(), label='FT_TC', linestyle='dashed')
ax_3.plot(SIM_i_alpha.day, SIM_i_alpha.FN.cumsum(), label='IT_TC', linestyle='dotted')
ax_3.plot(SIM_antigen_alpha.day, SIM_antigen_alpha.FN.cumsum(), label='IT_TC', linestyle='dashdot',color ='magenta')

ax_3.set_ylabel('Count')
ax_3.set_title('(E) Total false negative results')

ax_4.plot(SIM_g_alpha.day, SIM_g_alpha.FP.cumsum(), label='FT_TC', linestyle='dashed')
ax_4.plot(SIM_i_alpha.day, SIM_i_alpha.FP.cumsum(), label='IT_TC', linestyle='dotted')
ax_4.plot(SIM_antigen_alpha.day, SIM_antigen_alpha.FP.cumsum(), label='IT_TC', linestyle='dashdot',color ='magenta')

ax_4.set_ylabel('Count')
ax_4.set_title('(F) Total false positive results')

ax_5.scatter(SIM_g_alpha.day, SIM_g_alpha.NPV, label='PCR Flexible Testing',s=3,  )
ax_5.scatter(SIM_i_alpha.day, SIM_i_alpha.NPV, label='PCR Individual Testing',s=3,)
ax_5.scatter(SIM_antigen_alpha.day, SIM_antigen_alpha.NPV, label='Antigen Individual Testing',s=3,color ='magenta')

ax_5.set_ylabel('Probability')
ax_5.set_title('(A) Negative predictive values')

ax_6.scatter(SIM_g_alpha.day, SIM_g_alpha.PPV, label='FT',s=3)
ax_6.scatter(SIM_i_alpha.day, SIM_i_alpha.PPV, label='IT',s=3)
ax_6.scatter(SIM_antigen_alpha.day, SIM_antigen_alpha.PPV, label='IT',s=3,color ='magenta')

ax_6.set_ylabel('Probability')
ax_6.set_title('(B) Positive predictive values')



ax_1.legend(ncol=1,loc='lower right')

fig.text(0.5, 0.05,'Time (days)', ha='center',fontsize=15 )
plt.subplots_adjust(wspace=0.27, hspace=0.4)
# plt.show()
plt.savefig('.//Period-7days/periodical_test_7.pdf', bbox_inches='tight')