import pandas as pd
import numpy as np
##########################################################################################################
################## change file name here #################################################################
##########################################################################################################
anti_file_name = 'exp_antigen_individual_periodical.csv'
nstar_file_name = 'exp_nstar1_periodical.csv'
no_test_file_name = 'exp_no_testing.csv'
indi_file_name = 'exp_individual_periodical.csv'
##########################################################################################################
##########################################################################################################

anti = pd.read_csv(anti_file_name)
nstar = pd.read_csv(nstar_file_name)
no_test = pd.read_csv(no_test_file_name)
indi = pd.read_csv(indi_file_name)

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

    rna_extr_consumables = rna_extr_consumables_per_pcr*total_tests.sum()
    rt_pcr_consumables = rt_pcr_consumables_per_pcr*total_tests.sum()

    setup_labor = np.sum(VOT*setup_labor_per_n*n_stars*group_tests)
    rna_extra_labor = VOT*rna_extra_labor_per_pcr*total_tests.sum()
    rt_pcr_labor = VOT*rt_pcr_labor_per_pcr*total_tests.sum()
    reporting_labor = VOT* reporting_labor_per_person *tested_persons.sum()
    if is_antigen:
        return {'rna_extr_consumables': 0., 'rt_pcr_consumables': 0.,
                'setup': 0., 'rna_extr_labor': 0.,
                'rt_pcr_labor': 0., 'reporting_labor': 0,
                'labor_cost': 0,
                'Reagents and Consumables':antigen_per_test*total_tests.sum()+reporting_labor,
                'total_cost': antigen_per_test*total_tests.sum()+reporting_labor
                }
    else:
        return {'rna_extr_consumables':rna_extr_consumables,'rt_pcr_consumables':rt_pcr_consumables,
                'setup':setup_labor,'rna_extr_labor':rna_extra_labor,
                'Reagents and Consumables':rna_extr_consumables+rt_pcr_consumables,
                'rt_pcr_labor':rt_pcr_labor,'reporting_labor':reporting_labor,
                'labor_cost':setup_labor+rna_extra_labor+rt_pcr_labor+reporting_labor,
                'total_cost':setup_labor+rna_extra_labor+rt_pcr_labor+reporting_labor+rna_extr_consumables+rt_pcr_consumables
                }

def calculate_total(df):
    return (df['I']+df['R']+df['Q']+df['SQ']).max()
# calculate reduction in peak
col_names = ['Flexiable PCR','Individual PCR','Antigen']
row_names = ['reduction in peak','reduction in total']
reduction_peak = [
    (no_test['I'].max()-nstar['I'].max())/no_test['I'].max(),
    (no_test['I'].max()-indi['I'].max())/no_test['I'].max(),
    (no_test['I'].max()-anti['I'].max())/no_test['I'].max()
]
reduction_total = [
    (calculate_total(no_test) - calculate_total(nstar))/calculate_total(no_test),
    (calculate_total(no_test) - calculate_total(indi))/calculate_total(no_test),
    (calculate_total(no_test) - calculate_total(anti))/calculate_total(no_test)
]
df_reduction = pd.DataFrame([reduction_peak,reduction_total],columns=col_names,index=row_names)

df_cost = pd.DataFrame({'Flexiable PCR':calculate_cost(nstar),'Individual PCR':calculate_cost(indi), 'Antigen': calculate_cost(anti,is_antigen=True)})

df_cost = df_cost.append(df_reduction)
df_cost.loc['efficiency_in_peak'] = (df_cost.loc['reduction in peak']*100)/(df_cost.loc['total_cost']/1000000)
df_cost.loc['efficiency_in_total'] = (df_cost.loc['reduction in total']*100)/(df_cost.loc['total_cost']/100000)
df_cost.to_csv('cost analysis.csv')