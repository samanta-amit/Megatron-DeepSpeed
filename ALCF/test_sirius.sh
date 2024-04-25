#!/bin/bash --login
#
# Run complete test of
# https://github.com/argonne-lcf/Megatron-DeepSpeed
# on Sirius @ ALCF
# to launch (inside an interactive `qsub -I` job) on Sirius:
#
# ```bash`
# $ git clone https://github.com/argonne-lcf/Megatron-DeepSpeed
# $ cd Megatron-DeepSpeed/ALCF
# $ bash test_sirius.sh
# ````

# EXIT ON ERROR(s)
set -euxo pipefail

NOW="$(date "+%Y-%m-%d-%H%M%S")"

########################################################
# Setup / activate conda environment,
# mine is called q4-drop
########################################################
setup_conda() {
    export MAMBA_ROOT_PREFIX=/lus/tegu/projects/PolarisAT/foremans/micromamba
    shell_name=$(echo "${SHELL}" | tr "\/" "\t" | awk '{print $NF}')
    eval "$("${MAMBA_ROOT_PREFIX}/bin/micromamba" shell hook --shell ${shell_name})"
    micromamba activate 2024-04-23
}


########################################
# Make sure ./tmp/Megatron-DeepSpeed
# does not already exist
########################################
setup_megatron_deepspeed() {
    OUTDIR="OUTPUTS/test-sirius-${NOW}" && mkdir -p "${OUTDIR}" && cd "${OUTDIR}"
    echo "Running test in: ${OUTDIR}"
    echo "WORKING DIRECTORY: $(realpath $(pwd .))"
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
    export DEBUG=1
    export PBS_O_WORKDIR="$(pwd)"
    export DATA_FILE_LIST=./ALCF/data-lists/sirius/books.txt
    # LR=0.0008
    # GRAD_ACC_STEPS=8
    export ZERO_STAGE=1
    export NUM_LAYERS=10
    export MICRO_BATCH=8
    export TRAIN_ITERS=20
    export TIMING_LOG_LEVEL=1
    bash train_llama_alcf.sh |& tee "test-sirius-${NOW}".log
}

main
