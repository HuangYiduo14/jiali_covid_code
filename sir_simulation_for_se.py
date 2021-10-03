import statsmodels.api as sm
import time
import numpy as np
import pandas as pd
import multiprocessing
import tqdm

cpu_count = multiprocessing.cpu_count()
lowess = sm.nonparametric.lowess
big_M = 999
EPS = 1e-12
interested_percentiles = [0,15,25,50,75,85,100]

# revision 2: we will use tg as t00,
def get_mcmc_result():
    # this function will generate VL trajectories from posterior distribution of parameters
    file_name = 'used_pars_swab_1.csv'  # this file should contain samples from posterior distribution of parameters tg, tp, tw, tinc, vp
    mcmc_result = pd.read_csv(file_name)
    mcmc_result = mcmc_result.dropna()
    mcmc_result.reset_index(inplace=True, drop=True)
    # t00: first time greater than 0,
    # tpg: tp+tg,
    # t01: last time greater than 0，
    # these 3 params determines one trajectory
    mcmc_result['tpg'] = mcmc_result['tg'] + mcmc_result['tp']
    mcmc_result['t00'] = mcmc_result['tg']
    mcmc_result['t01'] = mcmc_result['tw'] + mcmc_result['tinc']
    return mcmc_result


mcmc_result = get_mcmc_result()  # this is the global dataframe that will keep the posterior samples


def simulate_one_group_test(df_trajs, detectable_load, n, save_vl=False):
    # this function simulate one group test for people in df_trajs
    # df_trajs must has col ['log10vload', 'is_I']
    # n: size of each group
    # do group test for each group with given n and curve
    # make groupname
    N = df_trajs.shape[0]
    group_name = np.arange(N)
    np.random.shuffle(group_name)
    group_name = group_name // n
    df_v_load = df_trajs[['log10vload', 'is_I']].copy()
    df_v_load['get_test'] = True
    df_v_load['is_I'] = 1 * df_v_load['is_I']
    df_v_load['group_name'] = group_name
    prevalence = df_v_load['is_I'].sum() / df_v_load.shape[0]
    # calculate each vload
    df_v_load['vload'] = 10 ** df_v_load['log10vload']
    vl = df_v_load['vload'].mean()
    if vl <= 2:  # since exp(0)=1, non-infected persons should have vl of 0
        vl = 0
    df_v_load.loc[df_v_load['vload'] <= 1 + EPS, 'vload'] = 0
    # calculate mean load for each group
    # df_group_vl is the table where each group is one row
    df_group_vl = df_v_load.groupby('group_name').agg({'vload': 'mean', 'is_I': 'sum'}).rename({'is_I': 'num_I_group'},
                                                                                               axis=1)
    df_group_vl['test_positive_group'] = df_group_vl['vload'] > 10 ** detectable_load
    if n > 1:
        df_group_vl['number_of_test_group'] = 1 + n * df_group_vl['test_positive_group']
    else:
        df_group_vl['number_of_test_group'] = 1
    df_group_vl.reset_index(inplace=True)  # make group number an col instead of index
    df_v_load = df_v_load.join(df_group_vl, on='group_name', how='left',
                               rsuffix='_group')  # join table on group names to find our group test results
    df_v_load['test_positive_ind'] = df_v_load['get_test'] & df_v_load['test_positive_group'] & (
            df_v_load[
                'vload'] > 10 ** detectable_load)  # test_positive_ind: if this person get tested and the result is positive
    if save_vl:
        df_v_load.to_csv('ind_v_load.csv')
    total_tested_infected = df_v_load['is_I'].sum()
    total_patient_found = df_v_load['test_positive_ind'].sum()
    if total_tested_infected == 0:
        return df_v_load, 0, 0, vl
    se_d = total_patient_found / total_tested_infected
    # note that cpr and se are not limited by the capacity, here we calculate over all population
    total_infected_group = (df_group_vl['num_I_group'] > 0).sum()
    total_test_out_group = df_group_vl['test_positive_group'].sum()
    if total_infected_group < EPS:
        se_p = 0
    else:
        se_p = total_test_out_group / total_infected_group
    if n == 1:
        assert abs(se_p - se_d) < EPS
    return df_v_load, se_p, se_d, vl


def give_se_cpr(df_trajs, detectable_load, n_list, save_vl=False):
    # give sed and sep for a list of n
    se_p_list = []
    se_d_list = []
    vl_list = []
    for n in n_list:
        if n == 1:
            _, se_p, se_d, vl = simulate_one_group_test(df_trajs, detectable_load, n, save_vl=save_vl)
            se_i = se_p
        else:
            _, se_p, se_d, vl = simulate_one_group_test(df_trajs, detectable_load, n, save_vl=save_vl)
        # print('n=',n,'vl=',vl)
        se_d_list.append(se_d)
        se_p_list.append(se_p)
        vl_list.append(vl)
    return se_d_list, se_p_list, vl_list


def calculate_v_load(df_trajs):
    '''
    calculate viral load using trajectory dataframe
    :param df_trajs: updated table
    :return: updated table
    '''
    mask0 = (df_trajs['day'] <= df_trajs['t00']) | (df_trajs['day'] >= df_trajs['t01'])
    df_trajs.loc[mask0, 'log10vload'] = 0
    mask1 = (df_trajs['day'] > df_trajs['t00']) & (df_trajs['day'] <= df_trajs['tpg'])  # incresing
    df_temp1 = df_trajs['vp'] / (df_trajs['tpg'] - df_trajs['t00']) * (df_trajs['day'] - df_trajs['t00'])
    df_trajs.loc[mask1, 'log10vload'] = df_temp1.loc[mask1]
    mask2 = (df_trajs['day'] > df_trajs['tpg']) & (df_trajs['day'] <= df_trajs['t01'])  # decresing
    df_temp2 = df_trajs['vp'] / (df_trajs['t01'] - df_trajs['tpg']) * (
            df_trajs['t01'] - df_trajs['day'])
    df_trajs.loc[mask2, 'log10vload'] = df_temp2.loc[mask2]
    df_trajs[['log10vload']].to_csv('new_vload_distribution.csv')
    return df_trajs


def group_test(df_trajs, n, daily_test_cap, parameters):
    # do group test for each group with given n and curve
    # make groupname
    # the input df_trajs must have cols:
    #     need_test_today: bool indicate that if this person need to be tested
    #     log10vload and is_I: v load and indicate if this is infected
    sp = parameters['sp']
    test_policy = parameters['test_policy']
    detectable_load = parameters['LOD']
    if n == 1:
        # for n=1, we only do individual tests
        df_v_load = df_trajs.loc[
            df_trajs['need_test_today'] > 0, ['log10vload', 'is_I']].copy()  # we select people that need test only
        df_v_load['vload'] = 10 ** df_v_load['log10vload']
        df_v_load.loc[df_v_load['vload'] <= 1 + EPS, 'vload'] = 0
        df_v_load['number_of_test_group'] = 1
        df_v_load['get_test'] = False
        if test_policy == 'round':
            if daily_test_cap < df_v_load.shape[0]:
                tested_individual = np.random.choice(df_v_load.index.values, daily_test_cap, replace=False)
                df_v_load.loc[tested_individual, 'get_test'] = True
            else:
                df_v_load['get_test'] = True
        else:
            df_v_load['get_test'] = True
        df_v_load['true_positive_ind'] = df_v_load['get_test'] & (df_v_load['vload'] > 10 ** detectable_load)
        df_v_load['false_positive_ind'] = df_v_load['get_test'] & (np.random.rand(df_v_load.shape[0]) >= sp) & (
            ~df_v_load['is_I'])
        df_v_load['test_positive_ind'] = df_v_load['true_positive_ind'] | df_v_load['false_positive_ind']
        TP = df_v_load['true_positive_ind'].sum()
        TN = (df_v_load['get_test'] & (~df_v_load['is_I']) & (~df_v_load['test_positive_ind'])).sum()
        FP = df_v_load['false_positive_ind'].sum()
        FN = (df_v_load['get_test'] & df_v_load['is_I'] & (~df_v_load['test_positive_ind'])).sum()
        # assert TP + TN + FP + FN == df_v_load['get_test'].sum()
        number_of_total_tests = df_v_load['get_test'].sum()
        number_of_group_tests = 0
        v_load_vec = df_v_load.loc[df_v_load['is_I'], 'log10vload'].values
        if len(v_load_vec) == 0:
            percentiles = [0 for _ in interested_percentiles]
            v_mean = v_std = 0
        else:
            percentiles = np.percentile(v_load_vec, interested_percentiles)
            v_mean = np.mean(v_load_vec)
            v_std = np.std(v_load_vec)

        return df_v_load, number_of_total_tests, number_of_group_tests, TP, TN, FP, FN,  percentiles, v_mean, v_std

    df_v_load = df_trajs.loc[
        df_trajs['need_test_today'] > 0, ['log10vload', 'is_I']].copy()  # we select people that need test only
    N_test = df_v_load.shape[0]  # number of people that need test
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
    df_group_vl['false_positive_group'] = (np.random.rand(df_group_vl.shape[0]) >= sp) & (
        ~df_group_vl['positive_group'])
    df_group_vl['test_positive_group'] = df_group_vl['true_positive_group'] | df_group_vl['false_positive_group']
    df_group_vl['number_of_test_group'] = 1 + n * df_group_vl['test_positive_group']
    # only selected group can be tested
    if test_policy == 'round':
        df_group_vl['cumsum_test'] = df_group_vl['number_of_test_group'].cumsum()
        df_group_vl['get_test'] = df_group_vl['cumsum_test'] <= daily_test_cap
        df_group_vl.drop('cumsum_test', axis=1, inplace=True)
    else:
        df_group_vl['get_test'] = True
    df_group_vl.reset_index(inplace=True)  # make group number an col instead of index
    df_v_load = df_v_load.join(df_group_vl, on='group_name', how='left', rsuffix='_group')  # join table on group names
    df_v_load['true_positive_ind'] = df_v_load['get_test'] & df_v_load['test_positive_group'] & (
            df_v_load[
                'vload'] > 10 ** detectable_load)  # test_positive_ind: if this person get tested and the result is positive
    df_v_load['false_positive_ind'] = df_v_load['get_test'] & df_v_load['test_positive_group'] & (
            np.random.rand(df_v_load.shape[0]) >= sp) & (~df_v_load['is_I'])
    df_v_load['test_positive_ind'] = df_v_load['true_positive_ind'] | df_v_load['false_positive_ind']
    TP = df_v_load['true_positive_ind'].sum()
    TN = (df_v_load['get_test'] & (~df_v_load['is_I']) & (~df_v_load['test_positive_ind'])).sum()
    FP = df_v_load['false_positive_ind'].sum()
    FN = (df_v_load['get_test'] & df_v_load['is_I'] & (~df_v_load['test_positive_ind'])).sum()
    number_of_total_tests = (df_group_vl['number_of_test_group'] * df_group_vl['get_test']).sum()
    number_of_group_tests = df_group_vl['get_test'].sum()
    #import pdb; pdb.set_trace()
    v_load_vec = df_v_load.loc[df_v_load['is_I'],'log10vload'].values
    if len(v_load_vec)==0:
        percentiles = [0 for _ in interested_percentiles]
        v_mean = v_std = 0
    else:
        percentiles = np.percentile(v_load_vec, interested_percentiles)
        v_mean = np.mean(v_load_vec)
        v_std = np.std(v_load_vec)

    return df_v_load, number_of_total_tests, number_of_group_tests, TP, TN, FP, FN, percentiles, v_mean, v_std


# first we consider SIR model
def SIRsimulation_for_se(N, parameters, table_n_star_up=None, table_n_star_down=None, external_infection_rate=0,
                         test_round=100,
                         n_star_policy='daily', test_policy='periodical', n_period=7, period=7,
                         round_daily_test_cap=10000, fixed_n=1, p_start=0.001,
                         T_lead=1, I0=100, R0=2.5, R02=2.5, R03=2.5, tmax=365, t_start=80, t_end=150, sym_ratio=0.4,
                         exp_name='default_exp', use_table=False):
    '''
    run SIR simulation to evaluate Se
    :param N: population size
    :param parameters: parameters of LOD, n_list candidate, sp
    :param table_n_star_up: a table recording n star and p, must have columns ['n_star','up'] and index is 'p' up is for increasing period, down is for decreasing period
    :param external_infection_rate: external infection rate
    :param n_star_policy: how to get optimal {'daily','period','fixed'}
    :param test_policy: how to test people {'periodical','round'}
    :param n_period: if we update n periodically, the period
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
    n_list = parameters['n_list']
    detectable_load = parameters['LOD']
    sp = parameters['sp']

    print(
        'S, I, R, Q, SQ, total_tested_individual, positive_results, number_of_total_tests, n_star, number_of_group_tests, TP,TN,FP,FN')
    # initialize record table, we will add one row each day
    col_names = ['S', 'I', 'R', 'Q', 'SQ', 'total_tested_individual', 'positive_results', 'number_of_total_tests',
                 'n_star', 'number_of_group_tests', 'TP', 'TN', 'FP', 'FN', 'n_star_by_sim', 'mean_day', 'is_increasing','v_mean','v_std'
                 ]+list(['v_pt_'+str(pt) for pt in interested_percentiles])
    log_exp_table = []
    log_se_table = []
    se_col_names = ['p'] + ['sep{0}'.format(n) for n in n_list] + ['sed{0}'.format(n) for n in n_list]
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
    # all persons trajectory information and infection information is in trajs table
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
    trajs['need_test_today'] = False  # if this person will be tested today
    trajs[
        'day_until_remove'] = big_M  # how many days before the test result and removed, we will assign one T_lead to the people that we identified as tested
    # we will remove I->Q if day_until_remove=0, only positive patients have this number<big_M
    trajs['get_test'] = False  # indicate if this person got test today, updated using group_test
    trajs['true_positive_ind'] = False  # indicate if the result is positive for this person in today's test
    # set day 0 patients
    trajs.loc[:I0 - 1, 'is_I'] = True
    trajs.loc[:I0 - 1, 'is_S'] = False
    trajs.loc[:I0 - 1, 'day'] = (
            (trajs.loc[:I0 - 1, 'tinc'].values) * np.random.rand(I0)).astype(
        int)  # we assume all these patients are incu
    # note that in pandas loc, we have to subtract 1 to make dim correct
    if test_policy == 'round':
        trajs['round_test_needed'] = trajs['will_test'] & (trajs['is_I'] | trajs['is_R'] | trajs[
            'is_S'])  # if is round test, we will have a column to indicate if tested or not
    else:
        trajs['period_test_in_days'] = big_M  # if is period test, we need to see how many days until the test day
    k = 0
    t_begin = tmax
    round_daily_test_cap_rec = round_daily_test_cap

    flag1 = True  # this will be false if we start test
    increasing_phase = True
    for t in range(tmax):
        print('day', t, '=' * 100)
        beta_t = beta(t)
        I = trajs['is_I'].sum()
        S = trajs['is_S'].sum()
        R = trajs['is_R'].sum()
        Q = trajs['is_Q'].sum()
        SQ = trajs['is_SQ'].sum()
        if I + S + R > 0:
            p_t = I / (I + S + R)
        else:
            p_t = 0
        trajs = calculate_v_load(trajs)  # calculate viral load
        # calculate n_star
        print('p=', p_t)
        # calculate se_d and se_p for all candidate n by simulating group sampling
        se_d_list, se_p_list, _ = give_se_cpr(trajs.loc[trajs['is_I'] | trajs['is_R'] | trajs['is_S']],
                                              detectable_load, n_list=n_list)
        log_se_table.append([p_t] + se_p_list + se_d_list)

        if increasing_phase:
            print('increasing')
        else:
            print('decreasing')
        # see if is increasing or not
        if t > 7:
            if all([log_se_table[-i][0] > p_t for i in range(1, 8)]):
                increasing_phase = False
        if n_star_policy == 'fixed':
            n_star = fixed_n
            n_star_this = n_star
        elif (n_star_policy == 'daily' or t % n_period == 0):
            # we can also simulate the process to compare the result
            n_list_temp = n_list
            cpr_temp = np.zeros_like(n_list_temp)
            if se_d_list[0] < EPS or p_t < EPS:
                cpr_temp[0] = 0
            else:
                cpr_temp[0] = 1. / se_p_list[0] / p_t
            for i, n in enumerate(n_list_temp):
                if n > 1:
                    if se_d_list[i] < EPS or p_t < EPS:
                        cpr_temp[i] = 0
                    else:
                        cpr_temp[i] = 1. / se_d_list[i] / p_t * (
                                    1. / n + se_p_list[i] - (se_p_list[i] + sp - 1) * (1 - p_t) ** n)
            n_star_this = n_list_temp[np.argmin(cpr_temp)]
            if use_table:
                if increasing_phase:
                    idx_closest = np.searchsorted(table_n_star_up.index.values, p_t)
                    if idx_closest == len(table_n_star_up.index.values):
                        idx_closest -= 1
                    n_star = int(table_n_star_up.iloc[idx_closest]['n_star'])
                else:
                    idx_closest = np.searchsorted(table_n_star_down.index.values, p_t)
                    if idx_closest == len(table_n_star_down.index.values):
                        idx_closest -= 1
                    n_star = int(table_n_star_down.iloc[idx_closest]['n_star'])
            else:
                n_star = n_star_this

        # -- do test --
        # step 1. identify the people that 'need_test_today' according to different policies
        if test_policy == 'round':
            if ((I / N) < p_start) & (t <= t_begin) & flag1:
                print('-------------------test not start-------------------')
                round_daily_test_cap = 0
            elif (I == 0):
                print('-------------------test end-------------------')
                break
            else:
                print('------------test today--------------------')
                flag1 = False
                round_daily_test_cap = round_daily_test_cap_rec

            trajs['need_test_today'] = trajs['round_test_needed']
        else:
            if t % period == 0:  # if is the first day of period, assign people to a random day
                np.random.seed(0)
                trajs['period_test_in_days'] = np.random.choice(list(range(period)), trajs.shape[0])
                np.random.seed(int(time.time() * 1000) % 1000)
                trajs.loc[~trajs['will_test'], 'period_test_in_days'] = big_M
            trajs['need_test_today'] = (trajs['period_test_in_days'] == 0) & (
                        trajs['is_I'] | trajs['is_R'] | trajs['is_S'])
        # step 2. do test
        parameters['test_policy'] = test_policy
        test_result, number_of_total_tests, number_of_group_tests, TP, TN, FP, FN, v_percentiles, v_mean, v_std = group_test(trajs, n_star,
                                                                                               daily_test_cap=round_daily_test_cap,
                                                                                               parameters=parameters)
        # print('number_of_total_tests:',number_of_total_tests,'------------TP:',TP,'--------------FP:',FP,)
        # step 3. update info, including day_until_remove
        trajs['get_test'] = False
        trajs['true_positive_ind'] = False
        trajs['get_test'] = test_result['get_test']
        trajs['get_test'].fillna(False, inplace=True)
        trajs['true_positive_ind'] = test_result['true_positive_ind']
        trajs['true_positive_ind'].fillna(False, inplace=True)
        will_be_Q_in_T_lead = trajs['true_positive_ind'] & (trajs['day_until_remove'] == big_M)
        trajs.loc[will_be_Q_in_T_lead, 'day_until_remove'] = T_lead
        # S -> I
        external_number_today = int(N * external_infection_rate)  # HYD: external rate
        neg_dS = round(
            beta_t * S * I / N) + external_number_today  # calculate new infections (-dS) # HYD: external rate
        if S - neg_dS < 0:
            neg_dS = S
        new_infected = trajs.loc[trajs['is_S']].sample(int(neg_dS), replace=False)
        trajs.loc[new_infected.index, 'day'] = 0
        trajs.loc[new_infected.index, 'is_I'] = True
        trajs.loc[new_infected.index, 'is_S'] = False

        # I -> SQ
        is_removed_symptom = trajs['can_be_sym'] & (trajs['day'] > trajs['tinc']) & (trajs['is_I'])
        # trajs.loc[is_removed_symptom,'day_until_remove'] = big_M
        trajs.loc[is_removed_symptom, 'is_I'] = False
        trajs.loc[is_removed_symptom, 'is_SQ'] = True

        # I -> Q
        is_removed_from_test = trajs['day_until_remove'] == 0
        trajs.loc[is_removed_from_test, 'day_until_remove'] = big_M
        trajs.loc[trajs['day_until_remove'] < big_M, 'day_until_remove'] -= 1
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
            trajs.loc[trajs['get_test'], 'round_test_needed'] = False
            # if all round_test_needed = False, this round is over and update
            if trajs['round_test_needed'].sum() == 0:
                # trajs['round_test_needed'] = trajs['will_test']   # & (trajs['is_I'] | trajs['is_S'])
                trajs['round_test_needed'] = trajs['will_test'] & (trajs['is_I'] | trajs['is_R'] | trajs['is_S'])
                print(k, 'old round ended, new round start')
                k += 1
                print('k=', k)
            else:
                trajs['round_test_needed'] = trajs['round_test_needed'] & (
                        trajs['is_I'] | trajs['is_R'] | trajs['is_S'])
        else:
            trajs['period_test_in_days'] -= 1
        infected_days = trajs.loc[trajs['is_I'], 'day'] / (
                    trajs.loc[trajs['is_I'], 'tw'] + trajs.loc[trajs['is_I'], 'tinc'])
        mean_day = infected_days.values.mean()
        std_day = infected_days.values.std()

        # add one day
        trajs.loc[trajs['is_I'], 'day'] += 1
        trajs.loc[trajs['is_R'], 'day'] += 1
        trajs.loc[trajs['is_Q'], 'day'] += 1
        trajs.loc[trajs['is_SQ'], 'day'] += 1

        # col_names = ['S', 'I', 'R', 'Q', 'SQ', 'total_tested_individual', 'positive_results', 'number_of_total_tests',
        #             'n_star', 'number_of_group_tests', 'TP', 'TN', 'FP', 'FN', 'n_star_by_sim', 'mean_day', 'is_increasing','v_mean','v_std', 'v_p0',....,]
        # print(S,I,R,Q,SQ,trajs['get_test'].sum(),trajs['true_positive_ind'].sum(),number_of_total_tests,n_star,number_of_group_tests,TP,TN,FP,FN,file=file_log)
        log_exp_table.append(
            [S, I, R, Q, SQ, trajs['get_test'].sum(), trajs['true_positive_ind'].sum(), number_of_total_tests, n_star,
             number_of_group_tests, TP, TN, FP, FN, n_star_this, mean_day, increasing_phase,v_mean,v_std]+[v_pt for v_pt in v_percentiles])
        # control test round
        if test_policy == 'round':
            if (k == test_round):
                print('end test')
                break

    df = pd.DataFrame(log_exp_table, columns=col_names)
    df['day'] = df.index
    df['PPV'] = df['TP'] / (df['TP'] + df['FP'])
    df['NPV'] = df['TN'] / (df['TN'] + df['FN'])
    df['I_all'] = df['I'] + df['Q'] + df['SQ'] + df['R']
    df['total_sample'] = (df['TP'] + df['FP']) + (df['TN'] + df['FN'])

    # df.to_csv('exp_'+exp_name+'.csv')

    df_se_result = pd.DataFrame(log_se_table, columns=se_col_names)
    # df_se_result.to_csv('se_'+exp_name+'.csv')

    return df, df_se_result


def get_results_no_table(seed_para):
    seed = seed_para[0]
    parameter_set = seed_para[1]
    N = parameter_set['N']
    I0 = parameter_set['I0']
    tmax = parameter_set['tmax']
    test_round = parameter_set['test_round']
    capcity = int(N * parameter_set['capacity_frac'])
    test_policy = parameter_set['test_policy']
    p_start = parameter_set['p_start']
    n_period = parameter_set['n_period']
    n_star_policy = parameter_set['n_star_policy']
    fixed_n = parameter_set['fixed_n']
    np.random.seed(seed)
    result_nstar7_round_2delay, result_se = SIRsimulation_for_se(N, parameter_set,
                                                                 T_lead=parameter_set['t_lead'], n_period=n_period,
                                                                 test_round=test_round, p_start=p_start,
                                                                 n_star_policy=n_star_policy, test_policy=test_policy,
                                                                 I0=I0,
                                                                 round_daily_test_cap=capcity, tmax=tmax,
                                                                 use_table=False, fixed_n=fixed_n, period=parameter_set['period'])
    max_p_date = result_se['p'].idxmax()
    result_se_up = result_se.loc[:max_p_date]
    result_se_down = result_se.loc[max_p_date:]
    return [result_se_up, result_se_down]


def get_results_with_table(seed_para):
    seed = seed_para[0]
    parameter_set = seed_para[1]
    N = parameter_set['N']
    I0 = parameter_set['I0']
    tmax = parameter_set['tmax']
    test_round = parameter_set['test_round']
    capcity = int(N * parameter_set['capacity_frac'])
    test_policy = parameter_set['test_policy']
    p_start = parameter_set['p_start']
    n_star_policy = parameter_set['n_star_policy']
    n_period = parameter_set['n_period']
    fixed_n = parameter_set['fixed_n']
    up_table = seed_para[2]
    down_table = seed_para[3]
    exp_name = seed_para[4]
    np.random.seed(seed)
    result_record, result_se = SIRsimulation_for_se(N, parameter_set, table_n_star_up=up_table,
                                                    table_n_star_down=down_table,
                                                    T_lead=parameter_set['t_lead'], n_period=n_period,
                                                    test_round=test_round, p_start=p_start,
                                                    n_star_policy=n_star_policy, test_policy=test_policy, I0=I0,
                                                    round_daily_test_cap=capcity, tmax=tmax,
                                                    use_table=True, fixed_n=fixed_n, period=parameter_set['period'])
    max_p_date = result_se['p'].idxmax()
    result_se_up = result_se.loc[:max_p_date]
    result_se_down = result_se.loc[max_p_date:]
    result_record.to_csv(exp_name + '_record.csv')
    result_se.to_csv(exp_name + '_se.csv')
    return result_record, result_se_up, result_se_down


def lowess_data(n_list, df_se, suffix='sep'):
    # fit curve se
    for n in n_list:
        x = df_se['p'].values
        y = df_se[suffix + str(n)].values
        z = lowess(y, x, return_sorted=False, frac=1. / 3)
        z[z > 1.] = 1.
        df_se[suffix + str(n) + '_lws'] = z
    return df_se


def get_n_star(df_se, n_list, sp):
    n_test = df_se.shape[0]
    cpr_matrix = np.zeros((n_test, len(n_list)))
    for i, n in enumerate(n_list):
        if n == 1:
            cpr_matrix[:, 0] = 1. / df_se['sed' + str(n) + '_lws'].values / df_se['p'].values
        else:
            se_vect = df_se['sep' + str(n) + '_lws'].values
            se_all_vect = df_se['sed' + str(n) + '_lws'].values
            p_vect = df_se['p'].values
            cpr_matrix[:, i] = 1. / se_all_vect / p_vect * (1. / n + se_vect - (se_vect + sp - 1) * (1 - p_vect) ** n)
    df_cpr = pd.DataFrame(cpr_matrix, columns=n_list)
    df_cpr['p'] = df_se['p']
    df_cpr.set_index('p', inplace=True)
    df_cpr['n_star'] = df_cpr.idxmin(axis=1)
    return df_cpr


if __name__ == '__main__':
    parameter_set_pcr = {'LOD': 2,
                         'n_list': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30],
                         'sp': 0.986,
                         't_lead': 2,
                         'N': 100000,
                         'I0': int(100000 // 2000),
                         'tmax': 365,
                         'p_start': 0.01,
                         'test_round': 10000000,
                         'test_policy': 'round',
                         'capacity_frac': 1 / 60, # this parameter is for round test
                         'period': 7, # this parameter is for periodical test
                         'n_star_policy': 'period',
                         'fixed_n': 1, # this parameter is for fixed n
                         'n_period':7 # this parameter is for periodically update n
                         }
    parameter_set_antigen = {'LOD': 5,
                             'n_list': [1],
                             'sp': 0.984,
                             't_lead': 0,
                             'N': 100000,
                             'I0': int(100000 // 2000),
                             'tmax': 365,
                             'p_start':0.01,
                             'test_round': 10000000,
                             'test_policy': 'round',
                             'capacity_frac': 1 / 3,# this parameter is for round test
                             'period': 7,# this parameter is for periodical test
                             'n_star_policy': 'fixed',
                             'fixed_n': 1,# this parameter is for fixed n
                             'n_period': 7 # this parameter is for periodically update n
                             }
    # # task 1: estimate sep/sed for pcr
    # all_results = []
    # seed_params = [[i, parameter_set_pcr] for i in range(cpu_count)]
    # with multiprocessing.Pool(cpu_count) as pool:
    #     for result in tqdm.tqdm(pool.imap_unordered(get_results_no_table, seed_params), total=len(seed_params)):
    #         all_results.append(result)
    # se_up_all = []
    # se_down_all = []
    # for i in range(cpu_count):
    #     se_up_all.append(all_results[i][0])
    #     se_down_all.append(all_results[i][1])
    # se_up_all = pd.concat(se_up_all)
    # se_down_all = pd.concat(se_down_all)
    # se_up_all.sort_values('p', inplace=True)
    # se_down_all.sort_values('p', inplace=True)
    # n_list = parameter_set_pcr['n_list']
    # se_up_all = lowess_data(n_list, se_up_all, 'sep')
    # se_up_all = lowess_data(n_list, se_up_all, 'sed')
    # se_down_all = lowess_data(n_list, se_down_all, 'sep')
    # se_down_all = lowess_data(n_list, se_down_all, 'sed')
    # se_up_all.to_csv('100_pcr_se_up.csv')
    # se_down_all.to_csv('100_pcr_se_down.csv')


    # # task 2: estimate sep/sed for antigen
    # all_results = []
    # with multiprocessing.Pool(cpu_count) as pool:
    #     for result in tqdm.tqdm(pool.imap_unordered(get_results_no_table, seed_params), total=len(seed_params)):
    #         all_results.append(result)
    # se_up_all = []
    # se_down_all = []
    # for i in range(cpu_count):
    #     se_up_all.append(all_results[i][0])
    #     se_down_all.append(all_results[i][1])
    # se_up_all = pd.concat(se_up_all)
    # se_down_all = pd.concat(se_down_all)
    # se_up_all.sort_values('p', inplace=True)
    # se_down_all.sort_values('p', inplace=True)
    # n_list = parameter_set_antigen['n_list']
    # se_up_all = lowess_data(n_list, se_up_all, 'sep')
    # se_up_all = lowess_data(n_list, se_up_all, 'sed')
    # se_down_all = lowess_data(n_list, se_down_all, 'sep')
    # se_down_all = lowess_data(n_list, se_down_all, 'sed')
    # se_up_all.to_csv('100_antigen_se_up.csv')
    # se_down_all.to_csv('100_antigen_se_down.csv')

    # task 3: generate n star table for pcr and antigen
    se_up_pcr = pd.read_csv('100_pcr_se_up.csv')
    se_down_pcr = pd.read_csv('100_pcr_se_down.csv')
    se_up_antigen = pd.read_csv('100_antigen_se_up.csv')
    se_down_antigen = pd.read_csv('100_antigen_se_down.csv')

    se_up_pcr = get_n_star(se_up_pcr, parameter_set_pcr['n_list'], parameter_set_pcr['sp'])
    se_down_pcr = get_n_star(se_down_pcr, parameter_set_pcr['n_list'], parameter_set_pcr['sp'])
    seed_params_pcr = [0, parameter_set_pcr, se_up_pcr, se_down_pcr, 'pcr_test']
    result_record, result_se_up, result_se_down = get_results_with_table(seed_params_pcr)
    result_record[['v_pt_'+str(pt) for pt in [15,50,85]]].plot()




    #se_up_antigen = get_n_star(se_up_antigen, parameter_set_antigen['n_list'], parameter_set_antigen['sp'])
    #se_down_antigen = get_n_star(se_down_antigen, parameter_set_antigen['n_list'], parameter_set_antigen['sp'])
    #seed_params_antigen = [0, parameter_set_pcr, se_up_antigen, se_down_antigen, 'antigen_test']
    #result_record_antigen, result_se_up_antigen, result_se_down_antigen = get_results_with_table(seed_params_antigen)

