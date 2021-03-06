#!/bin/bash
date >> $HOME/logs/time_forecasthub

# >>>  conda initialize >>>
. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate forecasthub
# >>>  conda initialize >>>

# Update data
python download_new_data.py

# Run jobs on cluster
qsub ./submit.sh


date >> $HOME/logs/time_forecasthub