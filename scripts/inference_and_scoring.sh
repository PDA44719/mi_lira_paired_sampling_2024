#!/bin/bash
python3 inference.py --logdir=$1/standard/
python3 score.py $1/standard/
python3 inference.py --logdir=$1/paired/record_$2/
python3 score.py $1/paired/record_$2/