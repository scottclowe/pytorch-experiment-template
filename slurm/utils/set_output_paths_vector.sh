#!/bin/bash

echo "========================================================================"
echo "-------- Setting checkpoint and output path variables ------------------"
date
echo ""

if [[ "$JOB_LABEL" == "" ]];
then
    # Still need to set the JOB_LABEL environment variable
    source "slurm/utils/set_job-label.sh"
fi

# Vector provides a fast parallel filesystem local to the GPU nodes, dedicated
# for checkpointing. It is mounted under /checkpoint. It is strongly
# recommended that you keep your intermediary checkpoints under this directory
CKPT_DIR="/checkpoint/${USER}/${SLURM_JOB_ID}"

echo "CKPT_DIR = $CKPT_DIR"
echo ""

# Ensure the checkpoint dir exists
mkdir -p "$CKPT_DIR"

# Create a symlink to the job's checkpoint directory within a subfolder of the
# current directory (repository directory) named checkpoint.
mkdir -p "checkpoints_working"
ln -sfn "$CKPT_DIR" "$PWD/checkpoints_working/$SLURM_JOB_NAME"

# In the future, the checkpoint directory will be removed immediately after the
# job has finished. If you would like the file to stay longer, and create an
# empty "delay purge" file as a flag so the system will delay the removal for
# 48 hours
touch "$CKPT_DIR/DELAYPURGE"

# Specify an output directory to place checkpoints for long term storage once
# the job is finished, and ensure this directory is added as a symlink within
# the current directory as well.
# OUTPUT_DIR is the directory that will contain all completed jobs for this
# project.
OUTPUT_DIR="/scratch/hdd001/home/$USER/checkpoints/$PROJECT_NAME"
# JOB_OUTPUT_DIR will contain the outputs from this job.
JOB_OUTPUT_DIR="$OUTPUT_DIR/$JOB_LABEL"

echo "Current contents of ${CKPT_DIR}:"
ls -lh "${CKPT_DIR}"
echo ""
echo "JOB_OUTPUT_DIR = $JOB_OUTPUT_DIR"
if [[ -d "$JOB_OUTPUT_DIR" ]];
then
    echo "Current contents of ${JOB_OUTPUT_DIR}"
    ls -lh "${JOB_OUTPUT_DIR}"
fi
echo ""

if [[ "$start_time" != "" ]];
then
    echo "------------------------------------"
    elapsed=$(( SECONDS - start_time ))
    eval "echo Running total elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
fi
echo "========================================================================"
echo ""
