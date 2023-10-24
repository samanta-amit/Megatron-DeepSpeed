#!/bin/bash --login
#SBATCH -A m3957_g
#SBATCH -C 'gpu&hbm80g'
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH --nodes 128
#SBATCH --gpus 512


# TODO::
# - Add logic for catching / killing hung process at end of run to ensure
#   second run starts up (otherwise, it will wait for the hung process, which
#   will run until the job is killed)
# - This wll let us try running multiple experiments in a single slurm job
#   allocation.
# - Existing (similar implementation) from my `~/bin/kill-match`:
#   ```bash
#   #!/bin/bash --login
#   TO_KILL=$1
#   kill $(ps aux | grep -E "$USER.+($TO_KILL)" | grep -v grep | awk '{print $2}')


PPSIZE=64 \
  MODEL_SIZE_KEY="GPT1T_$(( 2 * PPSIZE ))L" \
  SEQ_LEN=2048 \
  MICRO_BATCH=2 \
  GAS=$(( 8 * PPSIZE )) \
  SP_TYPE=megatron \
  ZERO_STAGE=1 \
  USE_SEQUENCE_PARALLEL=0 \
  MPSIZE=8 \
  SPSIZE=1 \
  USE_ACTIVATION_CHECKPOINTING=1 \
  ./ALCF/train-gpt3.sh
