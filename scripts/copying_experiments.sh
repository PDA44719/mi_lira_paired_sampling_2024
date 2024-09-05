#!/bin/bash
for experiment in experiment-0_16/ experiment-2_16/ experiment-4_16/ experiment-6_16/ experiment-8_16/ experiment-10_16/ experiment-12_16/ experiment-14_16/
do
    cp -R $1/standard/${experiment} $1/paired/record_$2/
done