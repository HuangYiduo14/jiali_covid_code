import pandas as pd
import numpy as np
##########################################################################################################
################## change file name here #################################################################
##########################################################################################################
anti_file_name = 'exp_antigen_individual_round_3day.csv'
anti_file_name_7 = 'exp_antigen_individual_round_7day.csv'
nstar_file_name = 'exp_nstar7_round_2delay.csv'
no_test_file_name = 'exp_no_testing.csv'
indi_file_name = 'exp_individual_round_1delay.csv'
##########################################################################################################
##########################################################################################################

anti = pd.read_csv(anti_file_name)
anti_7 = pd.read_csv(anti_file_name_7)
nstar = pd.read_csv(nstar_file_name)
no_test = pd.read_csv(no_test_file_name)
indi = pd.read_csv(indi_file_name)

# import pdb; pdb.set_trace()
# print(indi.head())

def calculate_cost(df_result, is_antigen=False):
    # parameters defined to calculate costs
    rna_extr_consumables_per_pcr = 9.18
    rt_pcr_consumables_per_pcr = 5.43
    antigen_per_test = 10.
    VOT = 36.5 # value of time ($/hour)
    rna_extra_labor_per_pcr = 1./14*0.5
    rt_pcr_labor_per_pcr = 1./28*0.5
    reporting_labor_per_person = 1.5/60. # this is per tested person
    setup_labor_per_n = 24/3600 # this is for pool test only, e.g. if pool size =5, then this should be setup_labor_per_n*5


    group_tests = (df_result['n_star'].values>1)*df_result['number_of_group_tests'].values
    total_tests = df_result['number_of_total_tests'].values
    tested_persons = df_result['total_tested_individual'].values
    n_stars = df_result['n_star'].values

    start_test_date = (df_result['total_tested_individual']>0).idxmax()
    with_test_days = df_result.loc[start_test_date:].shape[0]

    rna_extr_consumables = rna_extr_consumables_per_pcr*total_tests.sum()
    rt_pcr_consumables = rt_pcr_consumables_per_pcr*total_tests.sum()

    setup_labor = np.sum(VOT*setup_labor_per_n*n_stars*group_tests)
    rna_extra_labor = VOT*rna_extra_labor_per_pcr*total_tests.sum()
    rt_pcr_labor = VOT*rt_pcr_labor_per_pcr*total_tests.sum()
    reporting_labor = VOT* reporting_labor_per_person *tested_persons.sum()
    if is_antigen:
        return {'RNA extraction consumables': 0., 'RT-PCR consumables': 0.,'Antigen test kits': antigen_per_test*total_tests.sum()+reporting_labor,
                'Reagents and Consumables':antigen_per_test*total_tests.sum()+reporting_labor,
                'Pool test setup labor': 0., 'RNA extraction labor': 0.,
                'RT-PCR labor': 0., 'Reporting labor': 0,
                'Total labor cost': 0,
                'Total cost': antigen_per_test*total_tests.sum()+reporting_labor,
                'Individual tests': total_tests.sum()-group_tests.sum(),
                'Total tests': total_tests.sum(),
                'Average pool size': np.mean(n_stars),
                'Total infections': (df_result['I']+df_result['R']+df_result['Q']+df_result['SQ']).max(),
                'Identified infections': df_result['TP'].sum(),
                'False positives':df_result['FP'].sum(),
                'Testing period':with_test_days
                }
    else:
        return {'RNA extraction consumables':rna_extr_consumables,'RT-PCR consumables':rt_pcr_consumables, 'Antigen test kits':0,
                'Reagents and Consumables': rna_extr_consumables+rt_pcr_consumables,
                'Pool test setup labor':setup_labor,'RNA extraction labor':rna_extra_labor,
                'RT-PCR labor':rt_pcr_labor,'Reporting labor':reporting_labor,
                'Total labor cost':setup_labor+rna_extra_labor+rt_pcr_labor+reporting_labor,
                'Total cost':setup_labor+rna_extra_labor+rt_pcr_labor+reporting_labor+rna_extr_consumables+rt_pcr_consumables,
                'Individual tests': total_tests.sum() - group_tests.sum(),
                'Total tests': total_tests.sum(),
                'Average pool size': np.mean(n_stars),
                'Total infections': (df_result['I'] + df_result['R'] + df_result['Q'] + df_result['SQ']).max(),
                'Identified infections': df_result['TP'].sum(),
                'False positives': df_result['FP'].sum(),
                'Testing period': with_test_days
                }


def calculate_total(df):
    return (df['I']+df['R']+df['Q']+df['SQ']).max()
# calculate reduction in peak
col_names = ['Flexiable PCR','Individual PCR','Antigen test(every 3 day)', 'Antigen test(every 14 day)', ]
row_names = ['Reduction in peak','Reduction in total']
# import pdb; pdb.set_trace()



reduction_peak = [
    (no_test['I'].max()-nstar['I'].max()),
    (no_test['I'].max()-indi['I'].max()),
    (no_test['I'].max()-anti['I'].max()),
    (no_test['I'].max() - anti_7['I'].max())
]
reduction_total = [
    (calculate_total(no_test) - calculate_total(nstar)),
    (calculate_total(no_test) - calculate_total(indi)),
    (calculate_total(no_test) - calculate_total(anti)),
    (calculate_total(no_test) - calculate_total(anti_7))

]
df_reduction = pd.DataFrame([reduction_peak,reduction_total],columns=col_names,index=row_names)
#import pdb; pdb.set_trace()


df_cost = pd.DataFrame({'Flexiable PCR':calculate_cost(nstar),'Individual PCR':calculate_cost(indi), 'Antigen test(every 3 day)': calculate_cost(anti,is_antigen=True),'Antigen test(every 14 day)': calculate_cost(anti_7,is_antigen=True)})

df_cost = df_cost.append(df_reduction)
df_cost.loc['Cost of one reduction in peak infection ($ per person)'] = (df_cost.loc['Total cost'])/(df_cost.loc['Reduction in peak'])
df_cost.loc['Cost of one reduction in total infection ($ per person)'] = (df_cost.loc['Total cost'])/(df_cost.loc['Reduction in total'])
df_cost.loc['Cost of one identified and quarantined patients ($ per person)'] = (df_cost.loc['Total cost'])/(df_cost.loc['Identified infections'])

df_cost.loc['Daily cost of one reduction in peak infection ($ per person per day)'] = (df_cost.loc['Total cost'])/(df_cost.loc['Reduction in peak'])/df_cost.loc['Testing period']
df_cost.loc['Daily cost of one reduction in total infection ($ per person per day)'] = (df_cost.loc['Total cost'])/(df_cost.loc['Reduction in total'])/df_cost.loc['Testing period']
df_cost.loc['Daily cost of one identified and quarantined patients ($ per person per day)'] = (df_cost.loc['Total cost'])/(df_cost.loc['Identified infections'])/df_cost.loc['Testing period']

df_cost['Cost for each test']=None
df_cost['Cost for each individual'] = None
df_cost.loc['RNA extraction consumables','Cost for each test'] = 9.18
df_cost.loc['RT-PCR consumables','Cost for each test'] = 5.43
df_cost.loc['Antigen test kits','Cost for each test'] = 10



df_cost.loc['Pool test setup labor','Cost for each individual'] = (24/3600*36.5)
df_cost.loc['Reporting labor','Cost for each individual'] = 1.5/60.*36.5
df_cost.loc['RNA extraction labor','Cost for each test'] = 1./14*36.5
df_cost.loc['RT-PCR labor','Cost for each test'] = 1./28*0.5*36.5

df_cost.to_csv('cost analysis.csv')
