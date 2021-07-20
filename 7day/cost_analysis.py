import pandas as pd

anti = pd.read_csv('exp_antigen_individual_periodical.csv')
nstar = pd.read_csv('exp_nstar1_periodical.csv')
no_test = pd.read_csv('exp_no_testing.csv')
indi = pd.read_csv('exp_individual_periodical.csv')
def calculate_total(df):
    return (df['I']+df['R']+df['Q']+df['SQ']).max()
# calculate reduction in peak
col_names = ['Flexiable PCR','Individual PCR','Antigen']
row_names = ['reduction in peak','reduction in total','sample collected','pool tests','individual tests']
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
sample_collected = [
    nstar['total_tested_individual'].sum(),
    indi['total_tested_individual'].sum(),
    anti['total_tested_individual'].sum()
]
number_pool_tests = [
    nstar['number_of_group_tests'].sum(),
    0,
    0
]
number_ind_tests = [
    nstar['number_of_total_tests'].sum() - nstar['number_of_group_tests'].sum(),
    indi['total_tested_individual'].sum(),
    anti['total_tested_individual'].sum()
]

df = pd.DataFrame([
reduction_peak,reduction_total,sample_collected,number_pool_tests,number_ind_tests
],columns=col_names,index=row_names)