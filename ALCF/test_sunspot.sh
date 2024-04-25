#!/bin/bash --login
#
# Run complete test of
# https://github.com/argonne-lcf/Megatron-DeepSpeed
# on Sunspot @ ALCF

# EXIT ON ERROR(s)
set -euxo pipefail

########################################################
# Setup / activate conda environment,
# mine is called q4-drop
########################################################
setup_conda() {
    if [[ "${SHELL}" = "/bin/zsh" ]]; then
        eval "$(~/miniconda3/bin/conda shell.zsh hook)"
    else
        eval "$(~/miniconda3/bin/conda shell.bash hook)"
    fi
    conda activate q4-drop
}


########################################
# Make sure ./tmp/Megatron-DeepSpeed
# does not already exist
########################################
setup_megatron_deepspeed() {
    mkdir tmp && cd tmp
    if [[ -d "Megatron-DeepSpeed" ]]; then
        # rm -rfv Megatron-DeepSpeed/
        echo "Found existing Megatron-DeepSpeed.
        Remove existing directory to run test."
        exit
    fi
    git clone https://github.com/argonne-lcf/Megatron-DeepSpeed && cd Megatron-DeepSpeed
    git checkout remove-apex-deps
}


main() {
    setup_conda
    setup_megatron_deepspeed
    # NOTE: to use OPT=adamwschedulefree, you will need to pip install schedulefree
    DEBUG=1 PBS_O_WORKDIR="$(pwd)" DATA_FILE_LIST=./ALCF/data-lists/sunspot/books.txt LR=0.0008 GRAD_ACC_STEPS=8 ZERO_STAGE=1 NUM_LAYERS=10 MICRO_BATCH=8 OPT=adamwschedulefree TIMING_LOG_LEVEL=1 bash train_llama_alcf.sh
}

main
