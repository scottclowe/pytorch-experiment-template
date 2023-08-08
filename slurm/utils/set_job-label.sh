#!/bin/bash

echo "========================================================================"
echo "-------- Setting JOB_LABEL ---------------------------------------------"
echo ""
# Decide the name of the paths to use for saving this job
if [[ "$JOB_LABEL" != "" ]];
then
    echo "Using pre-set JOB_LABEL environment variable";
elif [ "$SLURM_ARRAY_TASK_COUNT" != "" ] && [ "$SLURM_ARRAY_TASK_COUNT" -gt 1 ];
then
    JOB_LABEL="${SLURM_JOB_NAME}__${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}";
else
    JOB_LABEL="${SLURM_JOB_NAME}__${SLURM_JOB_ID}";
fi
echo "JOB_LABEL = $JOB_LABEL"
echo "========================================================================"
echo ""
