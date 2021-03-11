#!/bin/bash [could also be /bin/tcsh]
#$ -S /bin/bash
#$ -N COVID19-forecast
#$ -pe mvapich2-zal 32
#$ -cwd
#$ -o $HOME/logs/output-forecasthubeu
#$ -e $HOME/logs/errors-forecasthubeu
#$ -t 1:4:1

# >>>  conda initialize >>>
. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate forecasthub
# >>>  conda initialize >>>

cd $HOME/Repositories/covid19-forecast-hub-europe/MODEL

python -u ./cluster_run.py -i $SGE_TASK_ID