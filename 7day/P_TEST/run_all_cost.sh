#!/bin/bash
for f in /Users/huangyiduo/Documents/covid_jialipaper/covid_testing-master/7day/P_TEST/p_start*;
do python $f/cost_analysis.py;
done