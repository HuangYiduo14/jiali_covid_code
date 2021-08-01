import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_mcmc_result():
    detectable_load0 = 3
    file_name = 'used_pars_swab_1.csv'
    mcmc_result = pd.read_csv(file_name)
    mcmc_result = mcmc_result.dropna()
    mcmc_result.reset_index(inplace=True, drop=True)
    # t00: first time greater than 0,
    # tpg: tp+tg,
    # t01: last time greater than 0，
    # these 3 params determines one trajectory
    mcmc_result['tpg'] = mcmc_result['tp'] + mcmc_result['tg']
    mcmc_result['is_tg_valid'] = (mcmc_result['vp'] >= detectable_load0) & \
                                 ((mcmc_result['vp'] - detectable_load0) / mcmc_result['tp'] > detectable_load0 /
                                  mcmc_result[
                                      'tg'])
    is_tg_valid = mcmc_result['is_tg_valid']
    # case 1: tg is valid
    mcmc_result.loc[is_tg_valid, 't00'] = mcmc_result.loc[is_tg_valid, 'tg'] - \
                                          mcmc_result.loc[is_tg_valid, 'tp'] / (
                                                  mcmc_result.loc[
                                                      is_tg_valid, 'vp'] - detectable_load0) * detectable_load0
    mcmc_result.loc[is_tg_valid, 't01'] = mcmc_result.loc[is_tg_valid, 'tw'] + mcmc_result.loc[is_tg_valid, 'tinc'] \
                                          + detectable_load0 * (mcmc_result.loc[is_tg_valid, 'tw'] + mcmc_result.loc[
        is_tg_valid, 'tinc']
                                                               - mcmc_result.loc[is_tg_valid, 'tpg']) / (
                                                  mcmc_result.loc[is_tg_valid, 'vp'] - detectable_load0)
    # case 2: tg is not valid
    mcmc_result.loc[~is_tg_valid, 't00'] = 0
    mcmc_result.loc[~is_tg_valid, 't01'] = mcmc_result.loc[~is_tg_valid, 'tw'] + mcmc_result.loc[~is_tg_valid, 'tinc']
    return mcmc_result

def CPR_group_test(df_trajs, n, daily_test_cap, se_i=0.9):
    # do group test for each group with given n and curve
    # make groupname
    N = df_trajs.shape[0]
    group_name = np.arange(N)
    np.random.shuffle(group_name)
    group_name = group_name // n
    df_v_load = df_trajs[['log10vload', 'is_I']].copy()
    df_v_load['is_I'] = 1 * df_v_load['is_I']
    df_v_load['group_name'] = group_name
    prevalence = df_v_load['is_I'].sum() / df_v_load.shape[0]
    # calculate each load
    df_v_load['vload'] = 10 ** df_v_load['log10vload']
    vl = df_v_load['vload'].mean()
    df_v_load.loc[df_v_load['vload'] <= 1 + EPS, 'vload'] = 0
    # calculate mean load for each group
    df_group_vl = df_v_load.groupby('group_name').agg({'vload': 'mean', 'is_I': 'sum'}).rename({'is_I': 'num_I_group'},
                                                                                               axis=1)
    df_group_vl['test_positive_group'] = df_group_vl['vload'] > 10 ** detectable_load
    print(detectable_load)
    if n > 1:
        df_group_vl['number_of_test_group'] = 1 + n * df_group_vl['test_positive_group']
    else:
        df_group_vl['number_of_test_group'] = 1

    df_group_vl['get_test'] = True
    df_group_vl.reset_index(inplace=True)  # make group number an col instead of index
    df_v_load = df_v_load.join(df_group_vl, on='group_name', how='left', rsuffix='_group')  # join table on group names
    df_v_load['test_positive_ind'] = df_v_load['get_test'] & df_v_load['test_positive_group'] & (
            df_v_load[
                'vload'] > 10 ** detectable_load)  # test_positive_ind: if this person get tested and the result is positive
    total_tested_infected = (df_v_load['is_I'] & df_v_load['get_test']).sum()
    total_patient_found = df_v_load['test_positive_ind'].sum()

    if total_tested_infected == 0:
        return None,None,None,None,None,None
    se_all = total_patient_found / total_tested_infected

    # note that cpr and se are not limited by the capacity, here we calculate over all population
    total_infected_group = (df_group_vl['num_I_group'] > 0).sum()
    total_test_out_group = df_group_vl['test_positive_group'].sum()
    # question: some R has large VL
    if total_infected_group < EPS:
        se = 1
    else:
        se = total_test_out_group / total_infected_group
    total_patient_found_theory = (df_v_load['test_positive_group'] & (df_v_load['vload'] > 10 ** detectable_load)).sum()

    if total_patient_found_theory<EPS:
        return df_v_load, se, None, 0, se_all,vl
    total_test_needed = df_group_vl['number_of_test_group'].sum()
    cpr = total_test_needed / total_patient_found_theory

    return df_v_load, se, cpr, 0, se_all,vl


def give_se_cpr(df_trajs, daily_test_cap, n_list):
    cpr_list = []
    cpr1_list = []
    se_list = []
    se_all_list = []
    vl_list = []
    for n in n_list:
        if n == 1:
            _, se, cpr, cpr1,se_all,vl = CPR_group_test(df_trajs, n, daily_test_cap)
            se_i = se
        else:
            _, se, cpr, cpr1,se_all,vl = CPR_group_test(df_trajs, n, daily_test_cap, se_i)
        print('n=',n,'vl=',vl)
        cpr_list.append(cpr)
        se_list.append(se)
        cpr1_list.append(cpr1)
        se_all_list.append(se_all)
        vl_list.append(vl)
    return cpr_list, se_list, cpr1_list,se_all_list, vl_list


def calculate_v_load(df_trajs, is_random_day=False):
    '''
    calculate viral load using trajectory dataframe
    :param df_trajs: updated table
    :return: updated table
    '''
    if is_random_day:
        df_trajs['day1'] = df_trajs['day']
        df_trajs['day'] = (df_trajs['tw'].values + df_trajs['tinc'].values) * np.random.rand(df_trajs.shape[0]).astype(
            int)
        df_trajs.loc[df_trajs['is_S'], 'day'] = -1
        df_trajs.loc[df_trajs['is_R'], 'day'] = df_trajs.loc[df_trajs['is_R'], 'day1']

    mask0 = (df_trajs['day'] <= df_trajs['t00']) | (df_trajs['day'] >= df_trajs['t01'])
    df_trajs.loc[mask0, 'log10vload'] = 0

    mask1 = (df_trajs['day'] > df_trajs['t00']) & (df_trajs['day'] <= df_trajs['tpg'])  # incresing
    df_temp1 = df_trajs['vp'] / (df_trajs['tpg'] - df_trajs['t00']) * (df_trajs['day'] - df_trajs['t00'])
    df_trajs.loc[mask1, 'log10vload'] = df_temp1.loc[mask1]

    mask2 = (df_trajs['day'] > df_trajs['tpg']) & (df_trajs['day'] <= df_trajs['t01'])  # decresing
    df_temp2 = df_trajs['vp'] / (df_trajs['t01'] - df_trajs['tpg']) * (
            df_trajs['t01'] - df_trajs['day'])  # there's a bug here << critical : HYD
    df_trajs.loc[mask2, 'log10vload'] = df_temp2.loc[mask2]

    if is_random_day:
        df_trajs['day'] = df_trajs['day1']
    return df_trajs


# first we consider SIR model
def SIRsimulation_get_curve(N, daily_test_cap, n_list, I0=100, R0=2.5, R02=0.8, R03=1.5, tmax=365, t_start=80,
                            t_end=150):
    '''
    run SIR simulation
    :param N: population size
    :param daily_test_cap: daily test capacity (all tested)
    :param n_list: list of n that we want in our experiments
    :param I0: number of infections at day 0
    :param R0: reproduction rate
    :param R02: reproduction rate after t_start
    :param R03: reproduction rate after t_end
    :param tmax: max day
    :param t_start: policy imposed date, R0 become R02 after this day
    :param tmax: policy end date, R0 become R03 after this day
    :return:
    cpr_table, cpr1_table, se_table, p_list
    here cpr1 is calculated using formula (1)
    cpr is calculated using from simulation
    '''
    # -- initialize the model --
    trajs = mcmc_result.sample(N, replace=True)  # sample trajectories for these people from MCMC results
    trajs.reset_index(inplace=True, drop=True)
    trajs['is_removed'] = False
    trajs['log10vload'] = 0
    trajs['is_I'] = False
    trajs['is_S'] = True
    trajs['is_R'] = False
    trajs['day'] = -1  # use this column to record the current day of infection, -1 means healthy
    # set day 0 patients
    trajs.loc[:I0 - 1, 'is_I'] = True
    trajs.loc[:I0 - 1, 'is_S'] = False
    trajs.loc[:I0 - 1, 'day'] = (
                (trajs.loc[:I0 - 1, 'tinc'].values + trajs.loc[:I0 - 1, 'tw'].values) * np.random.rand(I0)).astype(
        int)  # we assume all these patients are incu
    # note that in pandas loc, we have to subtract 1 to make dim correct
    cpr_table = []
    se_table = []
    p_list = []
    cpr1_table = []
    se_all_table = []
    vl_table = []
    for t in range(1):
        # -- calculate viral load --
        trajs = calculate_v_load(trajs)
        # n_star, trajs = find_opt_n(trajs, daily_test_cap)
        cpr_t_list, se_t_list, cpr1_t_list, se_all_list,vl_list = give_se_cpr(trajs, daily_test_cap, n_list)
        p_t = I0 / trajs.shape[0]
        p_list.append(p_t)
        cpr_table.append(cpr_t_list)
        se_table.append(se_t_list)
        cpr1_table.append(cpr1_t_list)
        se_all_table.append(se_all_list)
        vl_table.append(vl_list)
    return cpr_table, se_table, p_list, cpr1_table, se_all_table,vl_table


def save_data(p_list, n_list, data, name='test', save=False):
    df = pd.DataFrame(data, columns=n_list)
    df['p'] = p_list
    #df.set_index('p', inplace=True)
    #df.plot()
    df.to_csv(name + '_data.csv')
    #plt.show()
    #plt.title(name)
    # plt.xscale('log')
    #if save:
    #    plt.savefig(name + '_png')
    return df

def plot_scatter(table,n_list):
    for n in [3]:
        plt.scatter(table['p'],table[n],label=n)
    plt.legend()
    plt.show()


EPS = 1e-12
#detectable_load = 5
n_list = [1]
detectable_load = 3
#n_list = [1, 2, 3, 4, 5, 6 ,7 ,8, 9, 10, 15, 20, 25, 30]
mcmc_result = get_mcmc_result()

if __name__=='__main__':
    #p_random = 10**(-4+np.linspace(0,4,2000))
    p_random = [1. for _ in range(1000)]
    cpr_random = []
    se_random = []
    cpr1_random = []
    se_all_random = []
    vl_random = []
    N = 200000
    daily_test_cap = 1000
    k=0
    for p in p_random:
        print('exp',k,'random p', p)
        k+=1
        I0 = int(p * N)
        cpr_table1, se_table1, _, cpr1_table1,se_all_table1,vl_table = SIRsimulation_get_curve(N, daily_test_cap, n_list, tmax=1, I0=I0)
        cpr_random.append(cpr_table1[0])
        se_random.append(se_table1[0])
        cpr1_random.append(cpr1_table1[0])
        se_all_random.append(se_all_table1[0])
        vl_random.append(vl_table[0])
    #cpr1_table = save_data(p_random, n_list, cpr1_random, 'pcr_cpr1')
    #se_table = save_data(p_random, n_list, se_random, 'pcr_se')
    #se_all_table = save_data(p_random,n_list, se_all_random, 'pcr_se_all')
    vl_table = save_data(p_random,n_list,vl_random,'viral_load_mean_i_only')
    print(vl_table[1].mean())
    #cpr_table = save_data(p_random, n_list, cpr_random, 'anti_cpr')

# <<test of data initialization: PASSED
# mcmc_result1 = mcmc_result.copy()
# mcmc_result1['day'] = mcmc_result1['tg']
# mcmc_result1 = calculate_v_load(mcmc_result1)
# print('valid min and max',mcmc_result1.loc[mcmc_result1['is_tg_valid'],'log10vload'].min(),mcmc_result1.loc[mcmc_result1['is_tg_valid'],'log10vload'].max())
# mcmc_result1 = mcmc_result.copy()
# mcmc_result1['day'] = mcmc_result1['tpg']
# mcmc_result1 = calculate_v_load(mcmc_result1)
# mcmc_result1['diff'] = mcmc_result1['log10vload'] - mcmc_result1['vp']
# print('valid min and max',mcmc_result1['diff'].min(), mcmc_result1['diff'].max())
# mcmc_result1 = mcmc_result.copy()
# mcmc_result1['day'] = mcmc_result1['tinc']+mcmc_result1['tw']
# mcmc_result1 = calculate_v_load(mcmc_result1)
# print('valid min and max',mcmc_result1.loc[mcmc_result1['is_tg_valid'],'log10vload'].min(),mcmc_result1.loc[mcmc_result1['is_tg_valid'],'log10vload'].max())
