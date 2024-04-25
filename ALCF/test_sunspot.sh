#!/bin/bash --login
#
# Run complete test of
# https://github.com/argonne-lcf/Megatron-DeepSpeed
# on Sunspot @ ALCF
# to launch (inside an interactive `qsub -I` job) on Sirius:
#
# ```bash`
# $ git clone https://github.com/argonne-lcf/Megatron-DeepSpeed
# $ cd Megatron-DeepSpeed/ALCF
# $ bash test_sunspot.sh
# ````

# EXIT ON ERROR(s)
set -euxo pipefail

NOW="$(date "+%Y-%m-%d-%H%M%S")"

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
    OUTDIR="OUTPUTS/test-sunspot-${NOW}" && mkdir -p "${OUTDIR}" && cd "${OUTDIR}"
    echo "Running test in: ${OUTDIR}"
    echo "WORKING DIRECTORY: $(realpath $(pwd .))"
    if [[ -d "Megatron-DeepSpeed" ]]; then
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
    export DEBUG=1
    export PBS_O_WORKDIR="$(pwd)"
    export DATA_FILE_LIST=./ALCF/data-lists/sunspot/books.txt
    export ZERO_STAGE=1
    export NUM_LAYERS=10
    export MICRO_BATCH=8
    export TRAIN_ITER=20
    export TIMING_LOG_LEVEL=1
    bash train_llama_alcf.sh |& tee "test-suntpot-${NOW}".log
}

main
