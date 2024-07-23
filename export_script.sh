#!/bin/bash +x
#set -e
#set -x
export PBS_O_WORKDIR=$(pwd) && source ALCF/helpers.sh && setup_python
export WANDB_MODE=disabled
export DFTRACER_LOG_LEVEL=ERROR
export DFTRACER_ENABLE=1
export DFTRACER_INC_METADATA=1
export DFTRACER_DATA_DIR=/eagle/datasets/dolma/data_v1.7_Llama2Tokenizer/
export DFTRACER_LOG_FILE=./dft_fn_posix_level.pfw
