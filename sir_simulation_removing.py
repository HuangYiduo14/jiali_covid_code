from our_simulation import mcmc_result, CPR_group_test, calculate_v_load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import statsmodels.api as sm
n_list = [1, 2, 3, 4, 5, 10, 15]
color_table = {1:'b', 2:'r', 3:'g', 4:'c', 5:'m', 10:'y', 15:'k'}
df_cpr_up = pd.read_csv('cpr1_up_data.csv')
df_cpr_down = pd.read_csv('cpr1_down_data.csv')
df_se_up = pd.read_csv('se_table_up_data.csv')
df_se_down = pd.read_csv('se_down_data.csv')
df_se_up.sort_values(by='p',inplace=True)
df_se_down.sort_values(by='p',inplace=True)
df_cpr_up.sort_values(by='p',inplace=True)
df_cpr_down.sort_values(by='p',inplace=True)


def lowess_data(n_list,df_cpr):
    for n in n_list:
        # we will discard high variance values because variance is high
        df_cpr_cp = df_cpr.copy()
        df_cpr_cp = df_cpr_cp.loc[df_cpr_cp['p']>1e-3]
        x = df_cpr_cp['p'].values
        y = df_cpr_cp[str(n)].values

        z = func(df_cpr['p'],a0,b0,c0)
        plt.scatter(df_cpr['p'],df_cpr[str(n)],alpha=0.4,color=color_table[n])
        plt.plot(df_cpr['p'],z,color = color_table[n])
    return df_cpr

def ols_curve_fit_cpr(n_list,df_cpr):
    for n in n_list:
        # we will discard high variance values because variance is high
        df_cpr_cp = df_cpr.copy()
        df_cpr_cp = df_cpr_cp.loc[df_cpr_cp['p']>1e-2]
        x = df_cpr_cp['p'].values
        y = df_cpr_cp[str(n)].values
        func = lambda x,a,b,c: 1./a/x*(b+c*(1-x)**n)
        parm_guess,_ = curve_fit(func,x,y,p0=[0.8**2,0.8,-0.8])
        a0,b0,c0 = parm_guess
        z = func(df_cpr['p'],a0,b0,c0)
        plt.scatter(df_cpr['p'],df_cpr[str(n)],alpha=0.4,color=color_table[n])
        plt.plot(df_cpr['p'],z,color = color_table[n])
    return df_cpr




def give_se_cpr(df_trajs, daily_test_cap, n_list):
    cpr_list = []
    cpr1_list = []
    traj_list = []
    se_list = []
    for n in n_list:
        if n == 1:
            df_v_load, se, cpr, cpr1 = CPR_group_test(df_trajs, n, daily_test_cap)
            se_i = se
        else:
            df_v_load, se, cpr, cpr1 = CPR_group_test(df_trajs, n, daily_test_cap, se_i)
        cpr_list.append(cpr)
        traj_list.append(df_v_load)
        se_list.append(se)
        cpr1_list.append(cpr1)
    return cpr_list, se_list, cpr1_list

def SIRsimulation(N, daily_test_cap, n_list, I0=100, R0=2.5, R02=0.8, R03=1.5, tmax=365, t_start=80,
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
    # -- define beta as a function of time --
    S_list = []
    I_list = []
    R_list = []
    gamma = 1. / 7

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
    for t in range(tmax):
        print('day', t, '=' * 100)
        beta_t = beta(t)
        I = trajs['is_I'].sum()
        S = trajs['is_S'].sum()
        R = trajs['is_R'].sum()
        print('S:', S, ', I:', I, ', R:', R)

        assert S + R + I == trajs.shape[0]
        # -- calculate viral load --
        trajs = calculate_v_load(trajs)
        # n_star, trajs = find_opt_n(trajs, daily_test_cap)
        cpr_t_list, se_t_list, cpr1_t_list = give_se_cpr(trajs, daily_test_cap, n_list)
        p_t = I / trajs.shape[0]
        p_list.append(p_t)
        cpr_table.append(cpr_t_list)
        se_table.append(se_t_list)
        cpr1_table.append(cpr1_t_list)
        # -- update according to SIR --
        # S -> I
        neg_dS = round(beta_t * S * I / N)  # calculate new infections (-dS)
        if S - neg_dS < 0:
            neg_dS = S
        new_infected = trajs.loc[trajs['is_S']].sample(int(neg_dS), replace=False)
        trajs.loc[new_infected.index, 'day'] = 0
        trajs.loc[new_infected.index, 'is_I'] = True
        trajs.loc[new_infected.index, 'is_S'] = False
        # I -> R
        is_removed_undetected = trajs['day'] > trajs['tw'] + trajs['tinc']
        trajs.loc[is_removed_undetected, 'is_I'] = False
        trajs.loc[is_removed_undetected, 'is_R'] = True
        # I -> detected and removed
        # trajs = trajs.loc[~trajs['is_removed']] # TODO: how to remove? I->symptom ?
        # add one day
        trajs.loc[trajs['is_I'], 'day'] += 1
        trajs.loc[trajs['is_R'], 'day'] += 1
        S_list.append(S)
        I_list.append(I)
        R_list.append(R)
    # plt.plot(S_list, label='S')
    # plt.plot(I_list, label='I')
    # plt.plot(R_list, label='R')
    # plt.legend()
    # plt.show()
    return cpr_table, se_table, p_list, cpr1_table

ols_curve_fit_cpr(n_list,df_cpr_up)
#df_cpr_down = lowess_data(n_list,df_cpr_down)