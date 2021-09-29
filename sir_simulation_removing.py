from our_simulation import get_mcmc_result, calculate_v_load
import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys
import time

import matplotlib.pyplot as plt
lowess = sm.nonparametric.lowess

#df_cpr.sort_values(by='p',inplace=True)
big_M=999

EPS = 1e-12
##########################################################
# change detectable load and n_list here
#########################################################
# set 1: antigen
#detectable_load = 5
#n_list = [1]
#df_se = pd.read_csv('se_anti_data.csv')
#sp = 0.984
#t_lead = 0

# # set 2: pcr
#
detectable_load = 3
n_list = [1, 2, 3, 4, 5, 6 ,7 ,8, 9, 10, 15, 20, 25, 30]
df_se = pd.read_csv('se_pcr_3_data.csv')
df_all_se  = pd.read_csv('se_pcr_3_all_data.csv')
sp = 0.986
# # t_lead = 1

##########################################################
# change detectable load and n_list here: end
#########################################################
df_se.sort_values(by='p',inplace=True)
df_all_se.sort_values(by='p',inplace=True)

mcmc_result = get_mcmc_result()
color_table = {1:'b', 2:'r', 3:'g', 4:'c', 30:'m', 10:'y'}


def lowess_data(n_list,df_se):
    # fit curve se
    for n in n_list:
        x = df_se['p'].values
        y = df_se[str(n)].values
        z = lowess(y,x,return_sorted=False,frac=1./10)
        z[z>1.] = 1.
        df_se[str(n)+'_lws'] = z
        #if n in list(color_table.keys()):
            # plt.scatter(x,y,alpha=0.4,color=color_table[n])
        #    plt.plot(x,z,color =color_table[n],label=n)
    #plt.legend()
    return df_se

def group_test(t,df_trajs, n, daily_test_cap, test_policy='periodical'):
    # do group test for each group with given n and curve
    # make groupname
    # the input df_trajs must have cols:
    #     need_test_today: bool indicate that if this person need to be tested
    #     log10vload and is_I: v load and indicate if this is infected
    if n==1:
        df_v_load = df_trajs.loc[
            df_trajs['need_test_today'], ['log10vload', 'is_I']].copy()  # we select people that need test only
        df_v_load['vload'] = 10 ** df_v_load['log10vload']
        df_v_load.loc[df_v_load['vload'] <= 1 + EPS, 'vload'] = 0
        df_v_load['number_of_test_group'] = 1
        df_v_load['get_test'] = False
        if test_policy == 'round':
            if daily_test_cap<df_v_load.shape[0]:
                tested_individual = np.random.choice(df_v_load.index.values, daily_test_cap,replace=False)
                df_v_load.loc[tested_individual,'get_test'] = True
            else:
                df_v_load['get_test'] = True
        else:
            df_v_load['get_test'] = True
        df_v_load['true_positive_ind'] = df_v_load['get_test'] & (df_v_load['vload'] > 10 ** detectable_load)
        df_v_load['false_positive_ind'] = df_v_load['get_test']&(np.random.rand(df_v_load.shape[0])>=sp)&(~df_v_load['is_I'])
        df_v_load['test_positive_ind'] = df_v_load['true_positive_ind'] | df_v_load['false_positive_ind']
        TP = df_v_load['true_positive_ind'].sum()
        TN = (df_v_load['get_test'] & (~df_v_load['is_I']) & (~df_v_load['test_positive_ind'])).sum()
        FP = df_v_load['false_positive_ind'].sum()
        FN = (df_v_load['get_test'] & df_v_load['is_I'] & (~df_v_load['test_positive_ind'])).sum()
        # assert TP + TN + FP + FN == df_v_load['get_test'].sum()
        number_of_total_tests = df_v_load['get_test'].sum()
        number_of_group_tests = 0
        # if test_policy == 'round':
        #     assert number_of_total_tests <= daily_test_cap
        # import pdb; pdb.set_trace()

        return df_v_load, number_of_total_tests, number_of_group_tests, TP, TN, FP, FN

    df_v_load = df_trajs.loc[df_trajs['need_test_today'],['log10vload', 'is_I']].copy() # we select people that need test only
    N_test = df_v_load.shape[0] # number of people that need test
    group_name = np.arange(N_test)
    np.random.shuffle(group_name)
    group_name = group_name // n
    df_v_load['group_name'] = group_name
    # calculate each load
    df_v_load['vload'] = 10 ** df_v_load['log10vload']
    df_v_load.loc[df_v_load['vload'] <= 1 + EPS, 'vload'] = 0
    # calculate mean load for each group
    df_group_vl = df_v_load.groupby('group_name').agg({'vload': 'mean', 'is_I': 'sum'}).rename({'is_I': 'num_I_group'},
                                                                                               axis=1)
    df_group_vl['positive_group'] = df_group_vl['num_I_group'] > 0
    df_group_vl['true_positive_group'] = df_group_vl['vload'] > 10 ** detectable_load
    # here we add false positive cases
    # possible error
    # df_group_vl['false_positive_group'] = (np.random.rand(df_group_vl.shape[0])>=sp)&(~df_group_vl['true_positive_group'])


    df_group_vl['false_positive_group'] = (np.random.rand(df_group_vl.shape[0])>=sp)&(~df_group_vl['positive_group'])
    df_group_vl['test_positive_group'] = df_group_vl['true_positive_group']|df_group_vl['false_positive_group']
    df_group_vl['number_of_test_group'] = 1 + n * df_group_vl['test_positive_group']

    # import pdb; pdb.set_trace()

    # only selected group can be tested
    if test_policy=='round':
        df_group_vl['cumsum_test'] = df_group_vl['number_of_test_group'].cumsum()
        df_group_vl['get_test'] = df_group_vl['cumsum_test'] <= daily_test_cap
        df_group_vl.drop('cumsum_test', axis=1, inplace=True)
    else:
        df_group_vl['get_test'] = True
    df_group_vl.reset_index(inplace=True)  # make group number an col instead of index
    df_v_load = df_v_load.join(df_group_vl, on='group_name', how='left', rsuffix='_group')  # join table on group names
    df_v_load['true_positive_ind'] = df_v_load['get_test'] & df_v_load['test_positive_group'] & (
            df_v_load['vload'] > 10 ** detectable_load)  # test_positive_ind: if this person get tested and the result is positive
    # possible error YJL
    df_v_load['false_positive_ind'] = df_v_load['get_test'] & df_v_load['test_positive_group'] & (
                np.random.rand(df_v_load.shape[0])>=sp) & (~df_v_load['is_I'])

    df_v_load['test_positive_ind'] = df_v_load['true_positive_ind']|df_v_load['false_positive_ind']
    TP = df_v_load['true_positive_ind'].sum()
    TN = (df_v_load['get_test']&(~df_v_load['is_I'])&(~df_v_load['test_positive_ind'])).sum()
    FP = df_v_load['false_positive_ind'].sum()
    FN = (df_v_load['get_test']&df_v_load['is_I']&(~df_v_load['test_positive_ind'])).sum()
    # assert TP+TN+FP+FN ==df_v_load['get_test'].sum()
    # assert (1 * df_v_load['is_I'] - 1 * df_v_load['true_positive_ind']).min() >= 0
    number_of_total_tests = (df_group_vl['number_of_test_group'] * df_group_vl['get_test']).sum()
    number_of_group_tests = df_group_vl['get_test'].sum()
    # if test_policy=='round':
    #     assert number_of_total_tests<=daily_test_cap
    # df_v_load.to_csv('df_v_load_%d.csv'%t)
    return df_v_load, number_of_total_tests, number_of_group_tests, TP,TN,FP,FN

# first we consider SIR model
def SIRsimulation(N, table_n_star,external_infection_rate=0,test_round=100,
                            n_star_policy='daily',test_policy = 'periodical', t_period = 7, period = 7, round_daily_test_cap=10000,fixed_n=1,p_start=0.001,
                            T_lead = 1, I0=100, R0=2.5, R02=2.5, R03=2.5, tmax=365, t_start=80,t_end=150, sym_ratio=0.4, exp_name='default_exp'):
    # << here external_rate HYD
    '''
    run SIR simulation
    :param N: population size
    :param n_list: list of n that we want in our experiments
    :param table_n_star: a table recording n star and p, must have columns ['n_star'] and index is 'p'
    :param external_infection_rate: external infection rate

    :param n_star_policy: how to get optimal {'daily','period','fixed'}
    :param test_policy: how to test people {'periodical','round'}
    :param t_period: if we update t periodically, the period
    :param period: if we use period test, the period
    :param round_daily_test_cap: if we use round test, the daily test capacity (all tested)
    :param fixed_n: if we use fixed n policy, the value of n

    :param T_lead: time needed for the test, T_lead=1 means the result will come out tomorrow
    :param I0: number of infections at day 0
    :param R0: reproduction rate
    :param R02: reproduction rate after t_start
    :param R03: reproduction rate after t_end
    :param tmax: max day
    :param t_start: policy imposed date, R0 become R02 after this day
    :param tmax: policy end date, R0 become R03 after this day
    :param sym_ratio: proportion of patients with symptoms
    :return:
    cpr_table, cpr1_table, se_table, p_list
    here cpr1 is calculated using formula (1)
    cpr is calculated using from simulation
    '''
    # file_log = open('exp0_'+exp_name+'.txt','w')
    # print(exp_name, ':', file=file_log)
    # print(sys.argv[1:],file=file_log)
    # print('='*100,file=file_log)
    # file_log.flush()
    print('S, I, R, Q, SQ, total_tested_individual, positive_results, number_of_total_tests, n_star, number_of_group_tests, TP,TN,FP,FN')
    col_names = ['S', 'I', 'R', 'Q', 'SQ', 'total_tested_individual', 'positive_results', 'number_of_total_tests', 'n_star', 'number_of_group_tests','TP','TN','FP','FN']
    log_exp_table = []
    # -- define beta as a function of time --recover time mean(tw+t_incu)
    # HERE R0 is not Rt
    gamma = 1. / 20.68
    def beta(t):
        if t < t_start:
            return R0 * gamma
        elif t < t_end:
            return R02 * gamma
        else:
            return R03 * gamma
    # -- initialize the model --
    trajs = mcmc_result.sample(N, replace=True)  # sample trajectories for these people from MCMC results
    trajs.reset_index(inplace=True, drop=True)
    trajs['log10vload'] = 0

    trajs['is_I'] = False
    trajs['is_S'] = True
    trajs['is_R'] = False
    trajs['is_Q'] = False
    trajs['is_SQ'] = False
    trajs['can_be_sym'] = (np.random.rand(trajs.shape[0]) <= sym_ratio)
    trajs['will_test'] = (np.random.rand(trajs.shape[0]) <= 0.9)
    trajs['day'] = -1  # use this column to record the current day of infection, -1 means healthy
    trajs['need_test_today'] = False # if this person will be tested today
    trajs['day_until_remove'] = big_M # how many days before the test result and removed, we will assign one T_lead to the people that we identified as tested
    # we will remove I->Q if day_until_remove=0, only positive patients have this number<big_M
    trajs['get_test'] = False # indicate if this person got test today, updated using group_test
    trajs['true_positive_ind'] = False #indicate if the result is positive for this person in today's test
    # set day 0 patients
    trajs.loc[:I0 - 1, 'is_I'] = True
    trajs.loc[:I0 - 1, 'is_S'] = False
    trajs.loc[:I0 - 1, 'day'] = (
                (trajs.loc[:I0 - 1, 'tinc'].values) * np.random.rand(I0)).astype(
        int)  # we assume all these patients are incu
    # note that in pandas loc, we have to subtract 1 to make dim correct
    # import pdb;pdb.set_trace()

    # possible error
    if test_policy == 'round':
        # trajs['round_test_needed'] = trajs['will_test']
        #                              # &(trajs['is_I']|trajs['is_S']|trajs['is_R']|trajs['is_SQ']|trajs['is_Q']) # if is round test, we will have a column to indicate if tested or not

        trajs['round_test_needed'] = trajs['will_test']&(trajs['is_I']|trajs['is_R']|trajs['is_S']) # if is round test, we will have a column to indicate if tested or not


    else:
        trajs['period_test_in_days'] = big_M # if is period test, we need to see how many days until the test day
    k = 0
    t_begin = tmax
    round_daily_test_cap_rec = round_daily_test_cap

    for t in range(tmax):
        print('day', t, '=' * 100)
        beta_t = beta(t)
        I = trajs['is_I'].sum()
        S = trajs['is_S'].sum()
        R = trajs['is_R'].sum()
        Q = trajs['is_Q'].sum()
        SQ = trajs['is_SQ'].sum()
        p_t = I / trajs.shape[0]







        #print('S:', S, ', I:', I, ', R:', R, ', Q:', Q, ', SQ:', SQ,'R+Q+SQ:',R+Q+SQ)
        # assert S + R + I+Q+SQ == trajs.shape[0]
        trajs = calculate_v_load(trajs) # calculate viral load
        # calculate n_star
        if n_star_policy=='daily' or t%t_period==0:
            idx_closest = np.searchsorted(table_n_star.index.values, p_t)
            n_star = int(table_n_star.iloc[idx_closest]['n_star'])
        if n_star_policy=='fixed':
            n_star = fixed_n
        # -- do test --
        # step 1. identify the people that 'need_test_today' according to different policies
        if test_policy=='round':
            if ((I / N) < p_start) & (t <= t_begin):
                print('-------------------possible error-------------------')

                round_daily_test_cap = 0
            elif (I == 0):
                print('-------------------test end-------------------')
                break
            else:
                print('------------test_today--------------------')
                t_begin = t
                print('t_begin',t_begin)
                round_daily_test_cap = round_daily_test_cap_rec

            trajs['need_test_today'] = trajs['round_test_needed']
        else:
            if t%period==0: # if is the first day of period, assign people to a random day
                # HYD 0723
                np.random.seed(0)
                trajs['period_test_in_days'] = np.random.choice(list(range(period)), trajs.shape[0])
                # HYD 0723
                np.random.seed(int(time.time() * 1000) % 1000)

                trajs.loc[~trajs['will_test'],'period_test_in_days'] = big_M

            # YJL possible error
            # trajs['need_test_today'] = (trajs['period_test_in_days']==0)
            #                            # &(trajs['is_I']|trajs['is_S'])

            trajs['need_test_today'] = (trajs['period_test_in_days']==0)&(trajs['is_I']|trajs['is_R']|trajs['is_S'])

        # import pdb;
        # pdb.set_trace()

        # step 2. do test
        test_result, number_of_total_tests, number_of_group_tests, TP,TN,FP,FN = group_test(t,trajs,n_star,daily_test_cap=round_daily_test_cap, test_policy=test_policy)
        # print('number_of_total_tests:',number_of_total_tests,'------------TP:',TP,'--------------FP:',FP,)
        # step 3. update info, especially day_until_remove
        trajs['get_test'] = False
        trajs['true_positive_ind'] = False
        trajs['get_test'] = test_result['get_test']
        trajs['get_test'].fillna(False,inplace=True)
        trajs['true_positive_ind'] = test_result['true_positive_ind']


        trajs['true_positive_ind'].fillna(False,inplace=True)
        will_be_Q_in_T_lead = trajs['true_positive_ind'] & (trajs['day_until_remove'] == big_M)
        trajs.loc[will_be_Q_in_T_lead, 'day_until_remove'] = T_lead


        # import pdb; pdb.set_trace()

        # -- update according to SIR --
        # S -> I
        external_number_today = int(N*external_infection_rate) #HYD: external rate
        neg_dS = round(beta_t * S * I / N)+external_number_today   # calculate new infections (-dS) # HYD: external rate
        if S - neg_dS < 0:
            neg_dS = S
        new_infected = trajs.loc[trajs['is_S']].sample(int(neg_dS), replace=False)
        trajs.loc[new_infected.index, 'day'] = 0
        trajs.loc[new_infected.index, 'is_I'] = True
        trajs.loc[new_infected.index, 'is_S'] = False

        # I -> SQ
        is_removed_symptom = trajs['can_be_sym']&(trajs['day'] > trajs['tinc'])&(trajs['is_I'])
        # trajs.loc[is_removed_symptom,'day_until_remove'] = big_M
        trajs.loc[is_removed_symptom, 'is_I'] = False
        trajs.loc[is_removed_symptom, 'is_SQ'] = True

        # I,SQ-> Q
        is_removed_from_test = trajs['day_until_remove'] == 0
        # possible error YJL
        trajs.loc[is_removed_from_test,'day_until_remove'] = big_M

        # assert (1*trajs['is_I'] - 1*is_removed_from_test).min() >= 0
        trajs.loc[trajs['day_until_remove']<big_M,'day_until_remove'] -= 1
        trajs.loc[is_removed_from_test, 'is_I'] = False
        trajs.loc[is_removed_from_test, 'is_SQ'] = False
        trajs.loc[is_removed_from_test, 'is_Q'] = True


        # I,Q,SQ -> R
        is_removed_final = trajs['day'] > trajs['tw'] + trajs['tinc']
        trajs.loc[is_removed_final, 'is_I'] = False
        trajs.loc[is_removed_final, 'is_Q'] = False
        trajs.loc[is_removed_final, 'is_SQ'] = False
        trajs.loc[is_removed_final, 'is_R'] = True

        # continue of step 1, update after the test
        if test_policy == 'round':
            #print('people need to be test this round', trajs['round_test_needed'].sum())
            trajs.loc[trajs['get_test'], 'round_test_needed'] = False
            #print('people need to be test this round 2:', trajs['round_test_needed'].sum())
            # if all round_test_needed = False, this round is over and update
            if trajs['round_test_needed'].sum() == 0:
                # trajs['round_test_needed'] = trajs['will_test']   # & (trajs['is_I'] | trajs['is_S'])

                trajs['round_test_needed'] = trajs['will_test'] & (trajs['is_I'] | trajs['is_R'] | trajs['is_S'])

                print(k, 'old round ended, new round start')
                k+=1
                print('k=',k)
            else:
                # trajs['round_test_needed'] = trajs['round_test_needed']
        #                 #                              # & ( trajs['is_I'] | trajs['is_S'])

                trajs['round_test_needed'] = trajs['round_test_needed'] & (
                            trajs['is_I'] | trajs['is_R'] | trajs['is_S'])
        else:
            trajs['period_test_in_days'] -= 1
        # add one day
        trajs.loc[trajs['is_I'], 'day'] += 1
        trajs.loc[trajs['is_R'], 'day'] += 1
        trajs.loc[trajs['is_Q'],'day']+=1
        trajs.loc[trajs['is_SQ'], 'day'] += 1
        #print(S,I,R,Q,SQ,trajs['get_test'].sum(),trajs['true_positive_ind'].sum(),number_of_total_tests,n_star,number_of_group_tests,TP,TN,FP,FN,file=file_log)
        log_exp_table.append([S,I,R,Q,SQ,trajs['get_test'].sum(),trajs['true_positive_ind'].sum(),number_of_total_tests,n_star,number_of_group_tests,TP,TN,FP,FN])
        # trajs.to_csv('trajs%d.csv' % t)

        # file_log.flush()
    # file_log.close()

        # control test round
        if test_policy == 'round':
            if (k == test_round):
                print('end test')
                break


    df = pd.DataFrame(log_exp_table,columns=col_names)

    df['day'] = df.index
    df['PPV'] = df['TP'] / (df['TP'] + df['FP'])
    df['NPV'] = df['TN'] / (df['TN'] + df['FN'])
    df['I_all'] = df['I'] + df['Q'] + df['SQ'] + df['R']
    df['total_sample'] =  (df['TP'] + df['FP'])+(df['TN'] + df['FN'])


    df.to_csv('exp_'+exp_name+'.csv')
    return df


df_se = lowess_data(n_list,df_se)
df_all_se = lowess_data(n_list, df_all_se)

# df_se.to_csv('se_data_lws.csv',index=False)
# df_se = pd.read_csv('se_data_lws.csv').drop('Unnamed: 0',axis=1)
n_test = df_se.shape[0]
cpr_matrix = np.zeros((n_test, len(n_list)))
for i, n in enumerate(n_list):
    if n==1:
        cpr_matrix[:,0] = 1./df_se[str(n)+'_lws'].values/df_se['p'].values
    else:
        se_vect = df_se[str(n)+'_lws'].values
        se_all_vect = df_all_se[str(n)+'_lws'].values
        p_vect = df_se['p'].values
        # cpr_matrix[:,i] = 1. / se_vect / df_se['1_lws'].values / p_vect * (1. / n + se_vect - (se_vect+sp-1) * (1 - p_vect) ** n)
        cpr_matrix[:,i] = 1. / se_all_vect / p_vect * (1. / n + se_vect - (se_vect+sp-1) * (1 - p_vect) ** n)

df_cpr = pd.DataFrame(cpr_matrix,columns=n_list)
df_cpr['p'] = df_se['p']
df_cpr.set_index('p',inplace=True)
df_cpr['n_star'] = df_cpr.idxmin(axis=1)

# here we have all cpr data
# df_cpr['n_star'].plot() # draw n star plot here <<<<<
print('>> n_star curve generated','**'*100)

N=100000
I0 = int(N//2000)
capcity = int(N//60)
tmax=365
test_round = 10000000
capcity_antigen = int(N//3)
t_lead_antigen = 0
# pay attention to default variables
# ##########################No testing#########################
# #
# result_no_testing = SIRsimulation(N, table_n_star=df_cpr, exp_name='no_testing',
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=0,tmax=tmax)

##########################round testing#########################
#
# result_ind_round_1delay = SIRsimulation(N, table_n_star=df_cpr, exp_name='individual_round_1delay_0.001',T_lead = 1, test_round = test_round,p_start=0.001,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=capcity,tmax=tmax)
# #
# result_nstar7_round_2delay = SIRsimulation(N, table_n_star=df_cpr, exp_name='nstar7_round_2delay_0.001',T_lead = 2,test_round = test_round,p_start=0.001,
#                             n_star_policy='period',test_policy = 'round', I0=I0,round_daily_test_cap=capcity,tmax=tmax)
# result_nstar7_round_2delay = SIRsimulation(N, table_n_star=df_cpr, exp_name='nstar7_round_2delay_0.0025',T_lead = 2,test_round = test_round,p_start=0.0025,
#                             n_star_policy='period',test_policy = 'round', I0=I0,round_daily_test_cap=capcity,tmax=tmax)
# result_nstar7_round_2delay = SIRsimulation(N, table_n_star=df_cpr, exp_name='nstar7_round_2delay_0.0075',T_lead = 2,test_round = test_round,p_start=0.0075,
#                             n_star_policy='period',test_policy = 'round', I0=I0,round_daily_test_cap=capcity,tmax=tmax)
# result_nstar7_round_2delay = SIRsimulation(N, table_n_star=df_cpr, exp_name='nstar7_round_2delay_0.0125',T_lead = 2,test_round = test_round,p_start=0.0125,
#                             n_star_policy='period',test_policy = 'round', I0=I0,round_daily_test_cap=capcity,tmax=tmax)
#
# result_nstar7_round_2delay = SIRsimulation(N, table_n_star=df_cpr, exp_name='nstar7_round_2delay_0.015',T_lead = 2,test_round = test_round,p_start=0.015,
#                             n_star_policy='period',test_policy = 'round', I0=I0,round_daily_test_cap=capcity,tmax=tmax)
#

#
#
# result_ind_round_1delay = SIRsimulation(N, table_n_star=df_cpr, exp_name='individual_round_1delay_0.0025',T_lead = 1, test_round = test_round,p_start=0.0025,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=capcity,tmax=tmax)
#
#
# result_ind_round_1delay = SIRsimulation(N, table_n_star=df_cpr, exp_name='individual_round_1delay_0.0075',T_lead = 1, test_round = test_round,p_start=0.0075,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=capcity,tmax=tmax)
#
# result_ind_round_1delay = SIRsimulation(N, table_n_star=df_cpr, exp_name='individual_round_1delay_0.0125',T_lead = 1, test_round = test_round,p_start=0.0125,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=capcity,tmax=tmax)
#
# result_ind_round_1delay = SIRsimulation(N, table_n_star=df_cpr, exp_name='individual_round_1delay_0.015',T_lead = 1, test_round = test_round,p_start=0.015,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=capcity,tmax=tmax)


########################################
result_antigen_round_3day = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_round_3day_0.0025',T_lead = 0, test_round = test_round,p_start=0.0025,
                            n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=int(N//3),tmax=tmax)

# result_antigen_round_3day = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_round_3day_0.0075',T_lead = 0, test_round = test_round,p_start=0.0075,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=int(N//3),tmax=tmax)
# result_antigen_round_3day = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_round_3day_0.0125',T_lead = 0, test_round = test_round,p_start=0.0125,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=int(N//3),tmax=tmax)
#
# result_antigen_round_3day = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_round_3day_0.015',T_lead = 0, test_round = test_round,p_start=0.015,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=int(N//3),tmax=tmax)
#########################################


############################################
#
result_antigen_round_14day = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_round_14day_0.0025',T_lead = 0, test_round = test_round,p_start=0.0025,
                            n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=int(N//14),tmax=tmax)

# result_antigen_round_14day = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_round_14day_0.0075',T_lead = 0, test_round = test_round,p_start=0.0075,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=int(N//14),tmax=tmax)
# result_antigen_round_14day = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_round_14day_0.0125',T_lead = 0, test_round = test_round,p_start=0.0125,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=int(N//14),tmax=tmax)
#
# result_antigen_round_14day = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_round_14day_0.015',T_lead = 0, test_round = test_round,p_start=0.015,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=int(N//14),tmax=tmax)

#
#
# ####################################################

# result_ind_round_1delay = SIRsimulation(N, table_n_star=df_cpr, exp_name='individual_round_1delay_0.005',T_lead = 1, test_round = test_round,p_start=0.005,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=capcity,tmax=tmax)
#
#
# result_nstar7_round_2delay = SIRsimulation(N, table_n_star=df_cpr, exp_name='nstar7_round_2delay_0.005',T_lead = 2,test_round = test_round,p_start=0.005,
#                             n_star_policy='period',test_policy = 'round', I0=I0,round_daily_test_cap=capcity,tmax=tmax)


#
# result_ind_round_1delay = SIRsimulation(N, table_n_star=df_cpr, exp_name='individual_round_1delay_0.01',T_lead = 1, test_round = test_round,p_start=0.01,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=capcity,tmax=tmax)
#
#
# result_nstar7_round_2delay = SIRsimulation(N, table_n_star=df_cpr, exp_name='nstar7_round_2delay_0.01',T_lead = 2,test_round = test_round,p_start=0.01,
#                             n_star_policy='period',test_policy = 'round', I0=I0,round_daily_test_cap=capcity,tmax=tmax)







# result_antigen_round_3day = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_round_3day_0.001',T_lead = 0, test_round = test_round,p_start=0.001,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=int(N//3),tmax=tmax)
#
# result_antigen_round_3day = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_round_3day_0.005',T_lead = 0, test_round = test_round,p_start=0.005,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=int(N//3),tmax=tmax)
#
# result_antigen_round_3day = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_round_3day_0.01',T_lead = 0, test_round = test_round,p_start=0.01,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=int(N//3),tmax=tmax)
#
# result_antigen_round_14day = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_round_14day_0.001',T_lead = 0, test_round = test_round,p_start=0.001,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=int(N//14),tmax=tmax)
#
# result_antigen_round_14day = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_round_14day_0.005',T_lead = 0, test_round = test_round,p_start=0.005,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=int(N//14),tmax=tmax)
#
# result_antigen_round_14day = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_round_14day_0.01',T_lead = 0, test_round = test_round,p_start=0.01,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=int(N//14),tmax=tmax)
#
#


# result_antigen_round_7day = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_round_7day',T_lead = 0, test_round = test_round,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=int(N//7),tmax=tmax)
#
# result_antigen_round_14day = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_round_14day_p_start0.005',T_lead = 0, test_round = test_round,p_start=0.005,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=int(N//14),tmax=tmax)
#
# result_antigen_round_14day = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_round_14day',T_lead = 0, test_round = test_round,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=int(N//14),tmax=tmax)






# result_antigen_round_1day = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_round_1day',T_lead = 0, test_round = test_round,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=int(N//1),tmax=tmax)

# result_antigen_round_14day = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_round_14day',T_lead = 0, test_round = test_round,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=int(N//14),tmax=tmax)

# result_antigen_round_3day = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_round_0delay',T_lead = t_lead_antigen, test_round = test_round,
#                             n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=capcity_antigen,tmax=tmax)



# result_nstar7_round_1delay = SIRsimulation(N, table_n_star=df_cpr, exp_name='nstar7_round_1delay',T_lead = t_lead,test_round = test_round,
#                             n_star_policy='period',test_policy = 'round', I0=I0,round_daily_test_cap=capcity,tmax=tmax)

# result_nstar7_period_2delay = SIRsimulation(N, table_n_star=df_cpr, exp_name='nstar7_round_2delay',T_lead = 2, test_round = test_round,
#                             n_star_policy='period',test_policy = 'round',I0=I0,round_daily_test_cap=capcity,tmax=tmax)
#
# result_nstar1_round = SIRsimulation(N, table_n_star=df_cpr, exp_name='nstar1_round_1delay',T_lead = t_lead,test_round = test_round,
#                             n_star_policy='daily',test_policy = 'round', I0=I0,round_daily_test_cap=capcity,tmax=tmax)
# ##########################periodical testing#########################
#
# result_ind_period_1delay = SIRsimulation(N, table_n_star=df_cpr, exp_name='individual_periodical_1delay',T_lead = 1,
#                             n_star_policy='fixed',test_policy = 'periodical', I0=I0,tmax=tmax)
#
#
# result_ind_period_2delay = SIRsimulation(N, table_n_star=df_cpr, exp_name='individual_periodical_2delay',T_lead = 2,
#                             n_star_policy='fixed',test_policy = 'periodical', I0=I0,tmax=tmax)
#
# result_nstar7_period_0delay = SIRsimulation(N, table_n_star=df_cpr, exp_name='nstar7_periodical_0delay',T_lead = 0,
#                             n_star_policy='period',test_policy = 'periodical',I0=I0,tmax=tmax)

# result_nstar7_period_1delay = SIRsimulation(N, table_n_star=df_cpr, exp_name='nstar7_periodical_1delay',T_lead = 1,
#                             n_star_policy='period',test_policy = 'periodical',I0=I0,tmax=tmax)
# #
# result_nstar7_period_2delay = SIRsimulation(N, table_n_star=df_cpr, exp_name='nstar7_periodical_2delay',T_lead = 2,
#                             n_star_policy='period',test_policy = 'periodical',I0=I0,tmax=tmax)

# #######################Antigen need change VL in our simulation###############
# result_antigen_ind_period = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_periodical_0delay_7VL',T_lead = 0,
#                             n_star_policy='fixed',test_policy = 'periodical', I0=I0,tmax=tmax)
# result_antigen_ind_period = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_periodical_1delay',T_lead = 1,
#                             n_star_policy='fixed',test_policy = 'periodical', I0=I0,tmax=tmax)
# result_antigen_ind_period = SIRsimulation(N, table_n_star=df_cpr, exp_name='antigen_individual_periodical_2delay',T_lead = 2,
#                             n_star_policy='fixed',test_policy = 'periodical', I0=I0,tmax=tmax)


"""
result_ind_round = SIRsimulation(N, table_n_star=df_cpr, exp_name='individual_round',
                            n_star_policy='fixed',test_policy = 'round', I0=I0, round_daily_test_cap=capcity,tmax=tmax)

result_nstar1_period = SIRsimulation(N, table_n_star=df_cpr, exp_name='nstar1_periodical',
                            n_star_policy='daily',test_policy = 'periodical',I0=I0,tmax=tmax)

result_nstar1_round = SIRsimulation(N, table_n_star=df_cpr, exp_name='nstar1_round',
                            n_star_policy='daily',test_policy = 'round', I0=I0,round_daily_test_cap=capcity,tmax=tmax)

result_nstar7_period = SIRsimulation(N, table_n_star=df_cpr, exp_name='nstar7_periodical',
                            n_star_policy='period',test_policy = 'periodical',I0=I0,tmax=tmax)

result_nstar7_round = SIRsimulation(N, table_n_star=df_cpr, exp_name='nstar7_round',
                            n_star_policy='period',test_policy = 'round', I0=I0,round_daily_test_cap=capcity,tmax=tmax)


"""


# def plot_curve(col,is_cum,fig_number,title):
#     plt.figure(fig_number)
#     if is_cum:
#         plt.plot(result_ind_round[col].cumsum(), label='individual, round')
#         plt.plot(result_nstar1_round[col].cumsum(), label='n1, round')
#         #plt.plot(result_nstar7_round[col].cumsum(), label='n7, round')
#         plt.plot(result_ind_period[col].cumsum(), label='individual, period')
#         plt.plot(result_nstar1_period[col].cumsum(), label='n1, period')
#        # plt.plot(result_nstar7_period[col].cumsum(), label='n7, period')
#     else:
#         plt.plot(result_ind_round[col], label='individual, round')
#         plt.plot(result_nstar1_round[col], label='n1, round')
#         #plt.plot(result_nstar7_round[col], label='n7, round')
#         plt.plot(result_ind_period[col], label='individual, period')
#         plt.plot(result_nstar1_period[col], label='n1, period')
#         #plt.plot(result_nstar7_period[col], label='n7, period')
#     plt.legend()
#     plt.title(title)
#     plt.show()
# #result_ind_period[['S','I','R','SQ','Q']].plot()
# #plt.show()
#
# plot_curve('S',False,0,'S')
# plot_curve('I',False,1,'I')
# plot_curve('n_star',False,2,'N')
# plot_curve('TP',True,3,'TP')
# plot_curve('FP',True,4,'FP')
# plot_curve('TN',True,5,'TN')
# plot_curve('FN',True,6,'FN')
# plot_curve('FP',False,7,'FP(t)')