import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPS = 1e-12
detectable_load = 3
file_name='used_pars_swab_1.csv'
mcmc_result = pd.read_csv(file_name)
mcmc_result = mcmc_result.dropna()
mcmc_result.reset_index(inplace=True,drop=True)
# t00: first time greater than 0,
# tpg: tp+tg,
# t01: last time greater than 0，
# these 3 params determines one trajectory
mcmc_result['tpg'] = mcmc_result['tp']+mcmc_result['tg']
mcmc_result['is_tg_valid'] = (mcmc_result['vp']>=detectable_load)&\
                             ((mcmc_result['vp']-detectable_load)/mcmc_result['tp']>detectable_load/mcmc_result['tg'])
is_tg_valid = mcmc_result['is_tg_valid']
# case 1: tg is valid
mcmc_result.loc[is_tg_valid,'t00'] = mcmc_result.loc[is_tg_valid,'tg'] - \
                                     mcmc_result.loc[is_tg_valid,'tp']/(
                                             mcmc_result.loc[is_tg_valid,'vp']-detectable_load)*detectable_load
mcmc_result.loc[is_tg_valid,'t01'] = mcmc_result.loc[is_tg_valid,'tw']+mcmc_result.loc[is_tg_valid,'tinc'] \
                                     + detectable_load * (mcmc_result.loc[is_tg_valid,'tw']+mcmc_result.loc[is_tg_valid,'tinc']
                                                         -mcmc_result.loc[is_tg_valid,'tpg'])/(
                                             mcmc_result.loc[is_tg_valid,'vp']-detectable_load)
# case 2: tg is not valid
mcmc_result.loc[~is_tg_valid,'t00'] = 0
mcmc_result.loc[~is_tg_valid,'t01'] = mcmc_result.loc[~is_tg_valid,'tw']+mcmc_result.loc[~is_tg_valid,'tinc']


def CPR_group_test(df_trajs, n, daily_test_cap):
    # do group test for each group with given n and curve
    # make groupname
    N = df_trajs.shape[0]
    group_name = np.arange(N)
    np.random.shuffle(group_name)
    group_name = group_name//n
    df_v_load = df_trajs[['log10vload','is_I']].copy()
    df_v_load['group_name'] = group_name
    # calculate each load
    df_v_load['vload'] = 10**df_v_load['log10vload']
    df_v_load.loc[df_v_load['vload']<=1+EPS,'vload'] = 0
    # calculate mean load for each group
    df_group_vl = df_v_load.groupby('group_name').agg({'vload':'mean','is_I':'sum'}).rename({'is_I':'num_I_group'},axis=1)
    df_group_vl['test_positive_group'] = df_group_vl['vload']>10**detectable_load
    if n>1:
        df_group_vl['number_of_test_group'] = 1+n*df_group_vl['test_positive_group']
    else:
        df_group_vl['number_of_test_group'] = 1
    # only selected group can be tested
    df_group_vl['cumsum_test'] = df_group_vl['number_of_test_group'].cumsum()
    df_group_vl['get_test'] = df_group_vl['cumsum_test']<daily_test_cap
    df_group_vl.drop('cumsum_test',axis=1,inplace=True)
    df_group_vl.reset_index(inplace=True) # make group number an col instead of index
    df_v_load = df_v_load.join(df_group_vl,on='group_name',how='left',rsuffix='_group') #join table on group names
    df_v_load['test_positive_ind'] = df_v_load['get_test'] & df_v_load['test_positive_group'] &(
        df_v_load['vload']>10**detectable_load) # test_positive_ind: if this person get tested and the result is positive
    # TODO: detectable_load for individual or group difference?
    #total_tested_infected = (df_v_load['is_I'] & df_v_load['get_test']).s
    total_patient_found = df_v_load['test_positive_ind'].sum()
    #se = total_patient_found/total_tested_infected
    cpr = total_patient_found/daily_test_cap
    return df_v_load, 0, cpr

def find_opt_n(df_trajs, daily_test_cap):
    '''
    find opt n for each given traj (with day info)
    :param df_trajs: df with columns ['vp', 'tw', 'tp', 'tg', 'tinc', 'sigma', 'tpg', 'is_tg_valid', 't00',
       't01', 'log10vload', 'is_I', 'is_S', 'is_R', 'day','is_removed']
    :param daily_test_cap: capacity, int
    :return: optimal n and df with a new col: 'is_removed'
    '''
    # TODO: what n to use? should we use optimal n for each time? or fixed?
    n_list = list(range(1,10))
    cpr_list = []
    traj_list = []
    se_list = []
    for n in n_list:
        df_v_load, se, cpr = CPR_group_test(df_trajs, n, daily_test_cap)
        cpr_list.append(cpr)
        traj_list.append(df_v_load)
        se_list.append(se)
    ind_best = cpr_list.index(max(cpr_list))
    n_star = n_list[ind_best]
    df_trajs['is_removed'] = traj_list[ind_best]['test_positive_ind']
    # plt.plot(n_list,cpr_list)
    # plt.show()
    return n_star, df_trajs

def calculate_v_load(df_trajs):
    '''
    calculate viral load using trajectory dataframe
    :param df_trajs: updated table
    :return: updated table
    '''
    mask0 = (df_trajs['day']<=df_trajs['t00'])|(df_trajs['day']>=df_trajs['t01'])
    df_trajs.loc[mask0,'log10vload'] = 0
    import pdb; pdb.set_trace()

    mask1 = (df_trajs['day']>df_trajs['t00'])&(df_trajs['day']<=df_trajs['tpg']) # incresing
    df_temp1 = df_trajs['vp']/(df_trajs['tpg']-df_trajs['t00'])*(df_trajs['day']-df_trajs['t00'])
    df_trajs.loc[mask1, 'log10vload'] = df_temp1.loc[mask1]

    mask2 = (df_trajs['day']>df_trajs['tpg'])&(df_trajs['day']<=df_trajs['t01']) # decresing
    df_temp2 = df_trajs['vp'] / (df_trajs['t01'] - df_trajs['tpg']) * (
                df_trajs['day'] - df_trajs['t00'])
    df_trajs.loc[mask2, 'log10vload'] = df_temp2.loc[mask2]
    return df_trajs

# first we consider SIR model
def SIRsimulation(N, daily_test_cap, I0=100, asymptomatic=0.65, results_delay=0, R0=2.5, R02=0.8, R03=1.5, tmax=365,t_start=80,t_end=150):
    '''
    run SIR simulation
    :param N: population size
    :param daily_test_cap: daily test capacity (all tested)
    :param I0: number of infections at day 0
    :param asymptomatic: asymptomatic rate
    :param results_delay: how long to get result
    :param R0: reproduction rate
    :param R02: reproduction rate after t_start
    :param R03: reproduction rate after t_end
    :param tmax: max day
    :param t_start: policy imposed date, R0 become R02 after this day
    :param tmax: policy end date, R0 become R03 after this day
    :return:
    '''
    # -- define beta as a function of time --
    S_list = []
    I_list = []
    R_list = []
    gamma=1./7
    def beta(t):
        if t<t_start:
            return R0*gamma
        elif t<t_end:
            return R02*gamma
        else:
            return R03*gamma
    # -- initialize the model --
    trajs = mcmc_result.sample(N,replace=True) # sample trajectories for these people from MCMC results
    trajs.reset_index(inplace=True,drop=True)
    trajs['is_removed'] = False
    trajs['log10vload'] = 0
    trajs['is_I'] = False
    trajs['is_S'] = True
    trajs['is_R'] = False
    trajs['day'] = -1 # use this column to record the current day of infection, -1 means healthy
    # set day 0 patients
    trajs.loc[:I0 - 1, 'is_I'] = True
    trajs.loc[:I0 - 1, 'is_S'] = False
    trajs.loc[:I0-1,'day'] = (trajs.loc[:I0-1,'tinc'].values * np.random.rand(I0)).astype(int) # we assume all these patients are incu
    # note that in pandas loc, we have to subtract 1 to make dim correct
    for t in range(tmax):
        print('day', t)
        beta_t = beta(t)
        I = trajs['is_I'].sum()
        S = trajs['is_S'].sum()
        R = trajs['is_R'].sum()
        print('remain pop', S+R+I)
        assert S+R+I == trajs.shape[0]
        # -- calculate viral load --
        trajs = calculate_v_load(trajs)
        n_star, trajs = find_opt_n(trajs, daily_test_cap)
        # -- update according to SIR --
        # S -> I
        neg_dS = round(beta_t*S*I) # calculate new infections (-dS)
        if S-neg_dS<0:
            neg_dS = S
        new_infected = trajs.loc[trajs['is_S']].sample(neg_dS,replace=False)
        trajs.loc[new_infected.index,'day'] = 0
        trajs.loc[new_infected.index,'is_I'] = True
        trajs.loc[new_infected.index, 'is_S'] = False
        # I -> R
        is_removed_undetected = trajs['day']>trajs['tw']+trajs['tinc']
        trajs.loc[is_removed_undetected, 'is_I'] = False
        trajs.loc[is_removed_undetected, 'is_R'] = True
        # I -> detected and removed
        trajs = trajs.loc[~trajs['is_removed']] # TODO: how to remove? I->symptom ?
        import pdb; pdb.set_trace()
        # add one day
        trajs['day']+=1
        S_list.append(S)
        I_list.append(I)
        R_list.append(R)
    plt.plot(S_list,label='S')
    plt.plot(I_list,label='I')
    plt.plot(R_list,label='R')
    plt.legend()
    plt.show()
    return

N=100000
daily_test_cap = 10000
SIRsimulation(N, daily_test_cap)


# def simulate_se_sp(p,n_list,asym_rate = 0.5,pop=10000000,L=5):
#     NI = round(pop*p)
#     NI_asym = round(NI*asym_rate)
#     NI_sym = NI - NI_asym
#     NS = pop - NI
#
#     v_load = np.zeros(pop) # v load for each person
#     v_params = sample_trajectories(NI)
#
#     np.random.shuffle(v_load)
#
#     p_true_positive_list = []
#     for n in n_list:
#         n_groups = len(v_load)//n
#         if n==1:
#             detected_pop = np.sum(v_load>L)
#             n_total_patient_asym = np.sum(v_load > EPS)
#             p_true_positive = detected_pop/n_total_patient_asym
#             p_true_positive_list.append(p_true_positive)
#             continue
#         # vectorize to speed up
#         v_load_without_sym_reshaped = np.reshape(v_load[:n_groups*n],(n_groups,n))
#         v_group_mean = np.log10(np.mean(10.**v_load_without_sym_reshaped, axis=1))
#         detected_group = np.count_nonzero(v_group_mean>L)
#         infected_group = np.count_nonzero(v_group_mean>EPS)
#         p_true_positive = detected_group / infected_group
#         p_true_positive_list.append(p_true_positive)
#     return p_true_positive_list
# result = []
# #p_list = 10**(-2+0.02*np.arange(1,100))
# p_list = .02*np.arange(1,10)
# n_list = np.arange(1,10)
# for p in p_list:
#     p_true_pt = simulate_se_sp(p,n_list,asym_rate = 0.65,pop=1000000,L=3)
#     print(p, p_true_pt)
#     result.append([p]+p_true_pt)
# result = pd.DataFrame(result,columns=['p']+n_list.tolist())
# result.to_csv('result_test_1Mpop.csv')
# #result.set_index('p',inplace=True)
# result.plot(x='p',y=n_list)
# plt.xscale('log')


