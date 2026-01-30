#!/bin/bash

WORK_DIR="$(pwd)"
Ns=(144 196 256 324 400 625 1444)

for N in "${Ns[@]}"; do
    echo "Submitting job for N=${N}"
    qsub -v N=${N} "$WORK_DIR/overlap_100_more_q.sh"
done
