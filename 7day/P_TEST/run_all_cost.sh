#!/bin/bash
file0="/Users/huangyiduo/Documents/covid_jialipaper/covid_testing-master/7day/cost_analysis.py"
for f in /Users/huangyiduo/Documents/covid_jialipaper/covid_testing-master/7day/P_TEST/p_start*;
do
  cp -f $file0 $f/cost_analysis.py;
  python $f/cost_analysis.py;
done