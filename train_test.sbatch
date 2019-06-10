#!/bin/bash

source ~/new_begin.sh

echo "${IN}"
python train.py -in ${IN}
python test.py -in ${IN}
