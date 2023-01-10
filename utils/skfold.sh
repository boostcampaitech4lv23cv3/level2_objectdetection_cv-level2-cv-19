#!/bin/bash
python skfold.py -n 4 -f 0
python skfold.py -n 5 -f 0
python skfold.py -n 4 -f 5000
python skfold.py -n 5 -f 5000
python skfold.py -n 4 -f 10000
python skfold.py -n 5 -f 10000
python skfold.py -n 4 -f 20000
python skfold.py -n 5 -f 20000
python skfold.py -n 4 -f 30000
python skfold.py -n 5 -f 30000
