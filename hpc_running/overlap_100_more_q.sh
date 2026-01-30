#!/bin/bash
#PBS -N job_overlap_new
#PBS -l nodes=1:ppn=1
#PBS -l walltime=96:00:00

cd $PBS_O_WORKDIR

# Create logs directory (optional but recommended)
mkdir -p logs

# Redirect stdout & stderr using N
exec > logs/overlap_N${N}_new.out 2> logs/overlap_N${N}_new.err

PY_SCRIPT=overlap_MANY_n_100_ENSEMBLE_more_q.py
PYTHON=python3

echo "Job started on $(hostname)"
echo "Running for N=${N}"
echo "Start time: $(date)"

${PYTHON} ${PY_SCRIPT} ${N}

echo "End time: $(date)"
