#!/bin/bash

echo "========================================================================"
echo "-------- Reporting environment configuration ---------------------------"
date
echo ""
echo "pwd:"
pwd
echo ""
echo "which python:"
which python
echo ""
echo "python version:"
python --version
echo ""
echo "which conda:"
which conda
echo ""
echo "conda info:"
conda info
echo ""
echo "which pip:"
which pip
echo ""
## Don't bother looking at system nvcc, as we have a conda installation
# echo "which nvcc:"
# which nvcc || echo "No nvcc"
# echo ""
# echo "nvcc version:"
# nvcc --version || echo "No nvcc"
# echo ""
echo "nvidia-smi:"
nvidia-smi || echo "No nvidia-smi"
echo ""
echo "torch info:"
python -c "import torch; print(f'pytorch={torch.__version__}, cuda={torch.cuda.is_available()}, gpus={torch.cuda.device_count()}')"
python -c "import torch; print(str(torch.ones(1, device=torch.device('cuda')))); print('able to use cuda')"
echo ""
if [[ "$start_time" != "" ]];
then
    echo "------------------------------------"
    elapsed=$(( SECONDS - start_time ))
    eval "echo Running total elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
fi
echo "========================================================================"
echo ""
