import numpy as np
import pandas as pd
import multiprocessing
cpu_count = multiprocessing.cpu_count()
import tqdm
import matplotlib.pyplot as plt

# revision 2: we will use tg as t00,
def get_mcmc_result():
    file_name = 'used_pars_swab_1.csv'
    mcmc_result = pd.read_csv(file_name)
    mcmc_result = mcmc_result.dropna()
    mcmc_result.reset_index(inplace=True, drop=True)
    # t00: first time greater than 0,
    # tpg: tp+tg,
    # t01: last time greater than 0，
    # these 3 params determines one trajectory
    mcmc_result['tpg'] = mcmc_result['tg']+mcmc_result['tp']
    mcmc_result['t00'] = mcmc_result['tg']
    mcmc_result['t01'] = mcmc_result['tw'] + mcmc_result['tinc']
    return mcmc_result

def CPR_group_test(df_trajs, detectable_load, n, save_vl = False):
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
    # calculate each load
    df_v_load['vload'] = 10 ** df_v_load['log10vload']
    vl = df_v_load['vload'].mean()
    if vl<=2:
        vl = 0

    df_v_load.loc[df_v_load['vload'] <= 1 + EPS, 'vload'] = 0
    # calculate mean load for each group
    df_group_vl = df_v_load.groupby('group_name').agg({'vload': 'mean', 'is_I': 'sum'}).rename({'is_I': 'num_I_group'},
                                                                                               axis=1)
    df_group_vl['test_positive_group'] = df_group_vl['vload'] > 10 ** detectable_load
    if n > 1:
        df_group_vl['number_of_test_group'] = 1 + n * df_group_vl['test_positive_group']
    else:
        df_group_vl['number_of_test_group'] = 1

    df_group_vl.reset_index(inplace=True)  # make group number an col instead of index
    df_v_load = df_v_load.join(df_group_vl, on='group_name', how='left', rsuffix='_group')  # join table on group names
    df_v_load['test_positive_ind'] = df_v_load['get_test'] & df_v_load['test_positive_group'] & (
            df_v_load[
                'vload'] > 10 ** detectable_load)  # test_positive_ind: if this person get tested and the result is positive

    if save_vl:
        df_v_load.to_csv('ind_v_load.csv')
    total_tested_infected = df_v_load['is_I'].sum()
    total_patient_found = df_v_load['test_positive_ind'].sum()

    if total_tested_infected == 0:
        return df_v_load,None,None,vl
    se_d = total_patient_found / total_tested_infected

    # note that cpr and se are not limited by the capacity, here we calculate over all population
    total_infected_group = (df_group_vl['num_I_group'] > 0).sum()
    total_test_out_group = df_group_vl['test_positive_group'].sum()

    if total_infected_group < EPS:
        se_p = None
    else:
        se_p = total_test_out_group / total_infected_group
    if n==1:
        assert abs(se_p-se_d)<EPS

    return df_v_load, se_p, se_d, vl


def give_se_cpr(df_trajs, detectable_load, n_list, save_vl = False):
    se_p_list = []
    se_d_list = []
    vl_list = []
    for n in n_list:
        if n == 1:
            _, se_p, se_d,vl = CPR_group_test(df_trajs,detectable_load, n, save_vl=save_vl)
            se_i = se_p
        else:
            _, se_p, se_d,vl = CPR_group_test(df_trajs,detectable_load, n, save_vl=save_vl)
        #print('n=',n,'vl=',vl)
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


# first we consider SIR model
def SIRsimulation_get_curve(N, detectable_load, n_list, I0=100, save_vl=False):
    '''
    run SIR simulation
    :param N: population size
    :param n_list: list of n that we want in our experiments
    :param I0: number of infections at day 0

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
    trajs.loc[:I0 - 1, 'day'] = trajs.loc[:I0 - 1, 'tinc'].values * np.random.rand(I0)
    #trajs.loc[:I0 - 1, 'day'] = (trajs.loc[:I0 - 1, 'tinc'].values + trajs.loc[:I0 - 1, 'tw'].values) * np.random.rand(I0)  # we assume all these patients are incu
    # note that in pandas loc, we have to subtract 1 to make dim correct
    # -- calculate viral load --
    trajs = calculate_v_load(trajs)
    # n_star, trajs = find_opt_n(trajs, daily_test_cap)
    se_d_list, se_p_list, vl_list = give_se_cpr(trajs, detectable_load, n_list, save_vl=save_vl)
    p_t = I0 / trajs.shape[0]
    return se_d_list, se_p_list, vl_list, p_t



def save_data(p_list, n_list, data, name='test'):
    df = pd.DataFrame(data, columns=n_list)
    df['p'] = p_list
    df.to_csv(name + '_data.csv')
    return df




name = 'pcr' #{'pcr','antigen'}
EPS = 1e-12

if name == 'pcr':
    detectable_load = 3
    n_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
else:
    detectable_load = 5
    n_list = [1]

mcmc_result = get_mcmc_result()
N = 100000

def get_results(p):
    se_d_list, se_p_list, vl_list, p_t = SIRsimulation_get_curve(N,detectable_load,n_list, I0=int(p * N))
    return (se_d_list, se_p_list, vl_list, p_t)


if __name__=='__main__':
    p_random = 10**(-4+np.linspace(0,4,10000))
    p_random = np.repeat(p_random, 1)
    p_random.sort()
    #p_random = [1. for _ in range(1000)]
    all_results = []

    with multiprocessing.Pool(cpu_count) as pool:
        for result in tqdm.tqdm(pool.imap_unordered(get_results, p_random),total=len(p_random)):
            all_results.append(result)

    se_d_table = [result[0] for result in all_results]
    se_p_table = [result[1] for result in all_results]
    vl_table = [result[2] for result in all_results]
    p_table = [result[3] for result in all_results]

    se_d_table = save_data(p_table, n_list, se_d_table, 'se_d_'+name)
    se_p_table = save_data(p_table, n_list, se_p_table, 'se_p_'+name)
    vl_table = save_data(p_table,n_list,vl_table,'viral_load_'+name)

    p = 1
    n_list = [1]
    if name == 'pcr':
        SIRsimulation_get_curve(N, detectable_load, n_list, I0=int(p * N), save_vl = True)

    # estimate the distribution of


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
