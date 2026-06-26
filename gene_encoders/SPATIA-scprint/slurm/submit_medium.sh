#!/bin/bash -l

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo
#      # This needs to match Trainer(devices=...)

# module load cuda/11.7  # Adjust module loading for your cluster
# module load cudnn/11.x-v8.7.0.84  # Adjust module loading for your cluster

# run script from above
srun python3 scprint/__main__.py fit --trainer.logger.offline True --data.num_workers 16 --model.lr 0.002 --config config/pretrain_small.yaml
