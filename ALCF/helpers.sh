#!/bin/bash --login
#
# set -euxo pipefail

if [[ -n "${PBS_O_WORKDIR}" ]]; then
    WORKING_DIR="${PBS_O_WORKDIR}"
elif [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
    WORKING_DIR="${SLURM_SUBMIT_DIR}"
else
    echo "Unable to detect PBS or SLURM working directory info..."
    WORKING_DIR=$(python3 -c 'import os; print(os.getcwd())')
    echo "Using ${WORKING_DIR} as working directory..."
fi

export WORKING_DIR="${WORKING_DIR}"
printf "Using WORKING_DIR: %s\n" ${WORKING_DIR}


function get_machine() {
    if [[ $(hostname) == x4* ]]; then
        machine="aurora"
    elif [[ $(hostname) == x1* ]]; then
        machine="sunspot"
    elif [[ $(hostname) == x3* ]]; then
        if [[ "${PBS_O_HOST}" == sirius* ]]; then
            machine="sirius"
        else
            machine="polaris"
        fi
    elif [[ $(hostname) == nid* ]]; then
        machine="perlmutter"
    else
        echo "Unknown MACHINE. Setting MACHINE to $(hostname) and continuing..."
    fi
    export MACHINE="${machine}"
    printf "Running on: %s\n" "$(printBlue ${MACHINE})"
}


function check_and_kill_if_running() {
    # kill $(ps aux | grep -E "$USER.+(mpi|main.py)" | grep -v grep | awk '{print $2}')
    RUNNING_PIDS=$(lsof -i:29500 -Fp | head -n 1 | sed 's/^p//')
    if [[ -n "${RUNNING_PIDS}" ]];
        then echo "Caught ${RUNNING_PIDS}" && kill "${RUNNING_PIDS}";
    else
        echo "Not currently running. Continuing!"
    fi
}


function setupSrun() {
    if [[ $(hostname) == login* || $(hostname) == nid* ]]; then
        export NHOSTS="${SLURM_NNODES:-1}"
        export NGPU_PER_HOST="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
        export NGPUS="$(( NHOSTS * NGPU_PER_HOST ))"
        export SRUN_EXEC="srun --gpus ${NGPUS} --gpus-per-node ${NGPU_PER_HOST} -N ${NHOSTS} -n ${NGPUS} -l -u --verbose"
    else
        echo "Skipping setupSrun() on $(hostname)"
    fi
}


function printJobInfo() {
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "- MPICH_DIR=${MPICH_DIR:-${MPI_ROOT}}"
    echo "- Using $(which python3)"
    echo "- WORLD_SIZE:${WORLD_SIZE}"
    echo "- NCCL: ${NCCL:-nccl}"
    echo "- MODEL_TYPE: ${MODEL_TYPE}"
    echo "- Using DATA_FILE_LIST: ${DATA_FILE_LIST}"
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++"
}

function setupVenv() {
    VENV_DIR="$1"
    if [[ -d "${VENV_DIR}" ]]; then
        echo "Found venv at: ${VENV_DIR}"
        source "${VENV_DIR}/bin/activate"
    else
        echo "Skipping setupVenv() on $(hostname)"
    fi
}

function loadCondaEnv() {
    if [[ "${CONDA_EXE}" ]]; then
        echo "Already inside ${CONDA_EXE}, exiting!"
    else
        MODULE_STR="$1"
        module load "conda/${MODULE_STR}"
        nargs="$#"
        if [[ "${nargs}" -ge 2 ]]; then
            conda activate "$2"
        else
            conda activate base
        fi
    fi
}


function setupLauncher() {
    # outdir=$1
    if [[ -n "${DIST_LAUNCH}" && ${LAUNCH_CMD:-"MPICH"} != "deepspeed" ]]; then
        export LAUNCH_CMD="${DIST_LAUNCH} --genvall --cpu-bind depth -d 16 $(which python3) -Wignore ${EXEC}"
    else
        # Assert `./hostfile_deepspeed` exists
        export hfds="${WORKING_DIR}/hostfile_deepspeed" && [ -f "${hfds}" ] || exit
        export LAUNCH_CMD="deepspeed --hostfile $hfds --launcher MPICH ${EXEC}"
    fi
    printf "%s" "$(printRed 'Launching with:')"
    printf " %s" "$(printMagenta ${LAUNCH_CMD})"
}

function setDSlauncher() {
    # launcher setting
    outdir=$1
    export hfds="$outdir/hostfile_deepspeed"
    export hfmpi="$outdir/hostfile_mpich"
    [ -f "$hfds" ] || exit
    [ -f "$hfmpi" ] || exit
    export LAUNCHER=${LAUNCHER:-MPICH}
    if [[ $LAUNCHER == "deepspeed" ]]; then
        export launcher=""
    else
        export launcher="--force_multi --hostfile $hfds --launcher=${LAUNCHER} --launcher_args='-hostfile ${hfmpi}'"
    fi
}

function set_lr_args() {
    LR_ARGS="--lr ${LR} --lr-decay-style cosine"
    if [[ -n "${LR_DECAY_ITERS:-}" ]]; then
        LR_ARGS="${LR_ARGS} --lr-decay-iters ${LR_DECAY_ITERS}"
    fi
    if [[ -n "${LR_WARMUP_FRAC}" ]]; then
        LR_ARGS="${LR_ARGS} --lr-warmup-fraction ${LR_WARMUP_FRAC}"
    fi
    echo "LR_ARGS: ${LR_ARGS}"
    export LR_ARGS="${LR_ARGS}"
}

function setParams() {
    LLAMA_ARGS=""
    # +----[Parallelism Settings] -------------------------------------------+
    # +------[Aurora]--------||-------[SunSpot]-------------+
    if [[ $(hostname) == x4* || $(hostname) == x1* ]]; then
        TP=${TP:-1}                      # TP = 1
        export CCL=${CCL:-ccl}           # CCL
        export BE="${CCL}"               # COMMUNICATION BACKEND = CCL
        export DTYPE=${DTYPE:-bf16}      # DTYPE: bf16
        export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-8}     # GRADIENT_ACC_STEPS
        MICRO_BATCH=${MICRO_BATCH:-4}    # MICRO_BATCH = 4
        ##############################################################
        # NOTE: if NO_FLASH_ATTN is NON-empty; then NO FLASH ATTN !!
        if [[ -n "${NO_FLASH_ATTN-}" ]]; then
            echo "Not using flash-attn!!"
        else
            LLAMA_ARGS="${LLAMA_ARGS} --use-flash-attn-builder"
        fi
        ##############################################################
    # +--------[Polaris]-----------------------------------+
    elif [[ $(hostname) == x3* ]]; then
        # export LAUNCH_CMD="${LAUNCH_CMD:-deepspeed}"
        TP=${TP:-1}                                     # TP = 2
        export NCCL=${NCCL:-nccl}                       # NCCL
        export BE="${NCCL}"                             # BE = NCCL
        # export DTYPE=${DTYPE:-bf16}                   # DTYPE: BF16 ??
        export DTYPE=${DTYPE:-fp16}                     # DTYPE: FP16
        export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-8}     # GRADIENT_ACC_STEPS
        # NOTE: MICRO_BATCH is exported below
        MICRO_BATCH=${MICRO_BATCH:-2}    # MICRO_BATCH = 8
        if [[ -n "${NO_FLASH_ATTN-}" ]]; then
            echo "Not using flash-attn!!"
        else
            LLAMA_ARGS="${LLAMA_ARGS} --use-flash-attn-v2"
        fi
        echo "Setting up AWS NCCL OFI Plugin on Polaris..."
        source "${WORKING_DIR}/ALCF/aws_ofi_nccl_plugin.sh" || exit
    # +--------[Perlmutter]---------------------------------+
    elif [[ $(hostname) == login* || $(hostname) == nid* ]]; then
        TP="${TP:-2}"
        export NCCL="${NCCL:-nccl}"
        export BE="${NCCL}"
        export DTYPE="${DTYPE:-bf16}"
        MICRO_BATCH="${MICRO_BATCH:-8}"
        if [[ -n "${NO_FLASH_ATTN-}" ]]; then
            echo "Not using flash-attn!!"
        else
            LLAMA_ARGS="${LLAMA_ARGS} --use-flash-attn-v2"
        fi
    fi
    # +----------------------------------------------------------------------+
    export TP="${TP}"
    export PP="${PP:-1}"
    export DTYPE="${DTYPE:-bf16}"
    export OPT="${OPT:-adamw}"
    export HOSTFILE="${HOSTFILE:-${PBS_NODEFILE}}"
    NHOSTS=$(wc -l < "${HOSTFILE}")
    if [[ -z "${NGPU_PER_HOST-}" ]]; then
        NGPU_PER_HOST=$(python3 -c 'import ezpz as ez; print(ez.get_gpus_per_node())')
    fi
    export WORLD_SIZE="${WORLD_SIZE:-$(( NHOSTS * NGPU_PER_HOST ))}"
    # +---[Llama2 7B Config]--------------------------------------------------+
    export MODEL_KEY="Llama-7B"
    export HEADS=${HEADS:-${NHEADS:-32}}                # NUMBER OF ATEN HEADS
    export NLAYERS=${NLAYERS:-${NUM_LAYERS:-32}}        # NUMBER OF LAYERS
    export HIDDEN=${HIDDEN:-4096}                       # HIDDEN SIZE
    export NUM_KV_HEAD=${NUM_KV_HEAD:-8}                # GROUP ATTENTION
    export FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-11008}    # FFN HIDDEN SIZE
    # +---[Run Settings]------------------------------------------------------+
    export SEQ=${SEQ:-4096}                             # SEQ_LEN: 4096
    export ZERO_STAGE=${ZERO_STAGE:-1}                  # ZERO OFFLOADING STAGE
    export MICRO_BATCH=${MICRO_BATCH:-8}                # MICRO BATCH SIZE
    export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}          # GRADIENT ACCUMULATION STEPS
    export EVAL_ITERS="${EVAL_ITERS:-10}"               # NUMBER OF EVAL ITERS TO RUN
    export TRAIN_ITER=${TRAIN_ITER:-317892}             # NUMBER OF TRAIN ITERS
    export EVAL_INTERVAL="${EVAL_INTERVAL:-50000}"      # HOW FREQUENTLY TO RUN EVAL
    export SAVE_INTERVAL=${SAVE_INTERVAL:-200}          # HOW FREQUENTLY TO SAVE CKPTS
    export TIMING_LOG_LEVEL="${TIMING_LOG_LEVEL:-1}"    # TIMING VERBOSITY IN LOGS
    export ACT_CKPT_NUM_LAYERS="${ACT_CKPT_NUM_LAYERS:-1}"                  # NUM LAYERS TO CHECKPOINT ACTIVATIONS
    export USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING:-1}  # USE ACTIVATION CHECKPOINTING ?
    export GLOBAL_BATCH_MAX=$(( $WORLD_SIZE * $MICRO_BATCH * $GRAD_ACC_STEPS / $TP / $PP ))  # MAX GLOBAL BATCH SIZE
    export GLOBAL_BATCH="${GLOBAL_BATCH:-${GLOBAL_BATCH_MAX}}"  # WILL USE MAX IF NOT SET IN ENVIRONMENT
    tm="${WORKING_DIR}/ALCF/tokenizer.model"            # fallback: Megatron-DeepSpeed/ALCF/tokenizer.model
    export TOKENIZER_MODEL="${TOKENIZER_MODEL:-${tm}}"  # USE TOKENIZER_MODEL from env, else fallback from ^
    export MODEL_TYPE="llama-seq${SEQ}-pp${PP}-tp${TP}-${NLAYERS}layers-${HEADS}heads-${HIDDEN}hidden"  # STRING FOR IDENTIFYING MODEL
    # +----[ADDITIONAL LLAMA SPECIFIC ARGUMENTS]------------------------------
    export LLAMA_ARGS="${LLAMA_ARGS} --no-query-key-layer-scaling --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear"
    export LR=${LR:-0.0003}                             # LEARNING_RATE
    export LR_WARMUP_FRAC=${LR_WARMUP_FRAC:-0.05}       # LEARNING RATE WARMUP
    # export LR_DECAY_ITERS=${LR_DECAY_ITERS:-320000}     # LR DECAY ITERS
    export LR_DECAY_ITERS=${LR_DECAY_ITERS:-}     # LR DECAY ITERS
    set_lr_args
}


function setArgs() {
    # ---- Set DeepSpeed arguments --------------------------------
    ds_args=" "
    ds_args=" --deepspeed ${ds_args}"
    if [[ $PP == 1 ]]; then
       ds_args=" --no-pipeline-parallel ${ds_args}" 
    fi
    ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
    ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
    if [[ "$USE_ACTIVATION_CHECKPOINTING" == 1 ]]; then
        echo "!! Caught USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING} !!"
        ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
        # --checkpoint-activations \
        # --deepspeed-activation-checkpointing
    fi
    export ds_args
    # ---------------------------------------------------------------
    gpt_args=()
    # we are now using activation checkpoint provided by megatron, see below.
    # ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
    if [[ "$USE_ACTIVATION_CHECKPOINTING" == 1 ]]; then
        echo "!! Caught USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING} !!"
        gpt_args+=(
            "--checkpoint-activations"
            "--checkpoint-num-layers ${ACT_CKPT_NUM_LAYERS}"
        )
    fi
    export gpt_args
}


function make_ds_hostfile() {
    export GPUS_PER_NODE="${GPUS_PER_NODE:-${NGPU_PER_HOST:-${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}}}"
    # ---- Make MPICH hostfile ----------------
    hf="${HOSTFILE:-${PBS_NODEFILE}}"
    export hostfile_mpich=hostfile_mpich
    cat "${hf}" > "${hostfile_mpich}"
    # ---- Make DeepSpeed hostfile -------------------
    export hostfile_deepspeed=hostfile_deepspeed
    cat "${hf}" > "${hostfile_deepspeed}"
    sed -e "s/$/ slots=${GPUS_PER_NODE}/" -i "${hostfile_deepspeed}"
}

# +---------------------------------------+
# | 1. Git clone ezpz (if not found)    |
# | 2. Install ezpz (if not installed)  |
# +---------------------------------------+
function ezpz() {
    if [[ ! -d "${WORKING_DIR}/deps/ezpz" ]]; then
        mkdir -p "${WORKING_DIR}/deps"
        git clone https://github.com/saforem2/ezpz "${WORKING_DIR}/deps/ezpz"
    else
        echo "Found ezpz!"
    fi
    echo "Done with clone. Now, checking if ezpz is installed..."
    # if python3 -c 'import ezpz; print(ezpz.__file__)' 2> '/dev/null'; then
    if python3 -c "import sys; any(['ezpz' in s for s in sys.path])" 2> '/dev/null'; then
        echo "Has ezpz installed. Nothing to do."
    else
        echo "Does not have ezpz installed. Installing..."
        echo "Using $(which python3) to install ezpz:"
        python3 -m pip install -e "${WORKING_DIR}/deps/ezpz"  #  > ezpz-install.log 2>&1
    fi
    echo "Done with ezpz."
    source ${WORKING_DIR}/deps/ezpz/src/ezpz/bin/savejobenv  >  /dev/null 2>&1 #> /tmp/savejobenv.log 2>&1 || exit
    source ${WORKING_DIR}/deps/ezpz/src/ezpz/bin/getjobenv || exit
    make_ds_hostfile || exit
}

# +------------------------------------------------------------------------+
# | Save important environment variables to .deepspeed_env, which will be  |
# | forwarded to ALL ranks with DeepSpeed                                  |
# +------------------------------------------------------------------------+
function saveDSenv() {
    echo "Saving {PATH, LD_LIBRARY_PATH, htt{p,ps}_proxy, CFLAGS, PYTHONUSERBASE} to .deepspeed_env"
    {
        echo "PATH=${PATH}" ;
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" ;
        echo "http_proxy=${http_proxy}" ;
        echo "https_proxy=${https_proxy}" ;
        echo "CFLAGS=${CFLAGS}" ;
        echo "PYTHONUSERBASE=$PYTHONUSERBASE" ;
    } > .deepspeed_env
}

function setOutput() {
    # ---- Specify output location --------------------------------
    OUTPUT_PREFIX="ws${WORLD_SIZE}_ds_stage${ZERO_STAGE}_nl${NLAYERS}"
    OUTPUT_PREFIX="${OUTPUT_PREFIX}_hs${HIDDEN}_mb${MICRO_BATCH}"
    OUTPUT_PREFIX="${OUTPUT_PREFIX}_seq${SEQ}_gb${GLOBAL_BATCH}"
    OUTPUT_PREFIX="${OUTPUT_PREFIX}_pp${PP}_tp${TP}_${DTYPE}_opt${OPT}"
    OUTPUT_PREFIX="${OUTPUT_PREFIX}_lr${LR}_lwf${LR_WARMUP_FRAC}_ldi${LR_DECAY_ITERS}"
    if [[ -z "${NO_FLASH_ATTN:-}" ]]; then
        OUTPUT_PREFIX="${OUTPUT_PREFIX}_flash"
    fi
    export OUTPUT_PREFIX="${OUTPUT_PREFIX}"
    # OUTPUT_DIR="logs/${OUTPUT_PREFIX}/$(date +%m%d%H%M%S)_${HOSTNAME}"
    OUTPUT_DIR="logs/${OUTPUT_PREFIX}/$(date +%Y%m%d-%H%M%S)_${WORLD_SIZE}_${HOSTNAME}"
    export OUTPUT_DIR="${OUTPUT_DIR}"
    export OUTPUT_LOG="${OUTPUT_DIR}/output.log"
    export CKPT_DIR="checkpoints/${OUTPUT_PREFIX}"
    echo "${OUTPUT_LOG}" >> "logs/latest"
    mkdir -p "${OUTPUT_DIR}"
    printf "\n Please see logs at: %s\n" $(printGreen "${OUTPUT_DIR}")
    printf "Checkpoints will be saved to: %s\n" $(printYellow "${CKPT_DIR}")
}

function buildDSconfig() {
    # ---- Build DeepSpeed Config ---------------------------------
    export CPU_OPTIMIZER="${CPU_OPTIMIZER:-0}"
    export DS_CONFIG="${WORKING_DIR}/ds-configs/ds_stage${ZERO_STAGE}_mb${MICRO_BATCH}_gb${GLOBAL_BATCH}_pp${PP}_${DTYPE}.json"
    mkdir -p $(dirname "${DS_CONFIG}")
    echo "DS_CONFIG: ${DS_CONFIG}"
    printf "ZS: %s, , MB: %s, GB: %s, PP: %s, DTYPE: %s" "${ZERO_STAGE}" "${CPU_OPTIMIZER}" "${MICRO_BATCH}" "${GLOBAL_BATCH}" "${PP}" "${DTYPE}"
    # working_dir="${PBS_O_WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
    generateDSconfig "${DS_CONFIG}"
    # bash "${WORKING_DIR}/ALCF/generate_ds_config.sh" "${DS_CONFIG}"
    # -------------------------------------------------------------
}


function sumWeights() {
    local file_list=$1
    weights=$(cat "${file_list}" | awk '{print $1}' | tr '\n' '\ ,\ ' | sed 's/^/[/g' | sed 's/$/]/g' | tr '\ ' "\,\ ")
    python3 -c "import numpy as np; print(np.sum(${weights}))"
}

function sumFiles() {
    local rd=$1
    for f in $("${rd}/*.txt"); do
        ws=$(sumWeights "${rd}/${f}")
        echo "sum($f.weights)=${ws}"
    done
}

########################################################
# Setup / activate conda environment,
########################################################
setup_conda_sunspot() {
    ###### check if CONDA_PREFIX non-empty ################
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        eval "$(~/miniconda3/bin/conda shell.zsh hook)"
        conda activate anl_24_q2_release
    fi
    # ------------------------------------------------------------------------
    # XXX: Jerome's `frameworks_2024_5_v2` seems broken ??
    # - seems to be missing `python3 -c 'from mpi4py import MPI'` ???
    # - consequently, we leave the setup below commented out (for the time
    #   being):
    # if [[ -z "${CONDA_PREFIX-}" ]]; then
    #     module use -a /home/jmitche1/anl_release/2024/q2 ; module load frameworks_2024_5_v2
    # else
    #     echo "Caught CONDA_PREFIX=${CONDA_PREFIX}"
    # fi
    #
    # ------------------------------------------------------------------------
    ###### check if VIRTUAL_ENV non-empty ####################################
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        DEFAULT_VENV_PATH="${WORKING_DIR}/venvs/anl_24_q2_release"
        # venvs/anl_24_q2_release/bin/activate
        # if [[ -d "${DEFAULT_VENV_PATH}" ]]; then
        echo "Caught virtual env at ${DEFAULT_VENV_PATH}!"
        source "${DEFAULT_VENV_PATH}/bin/activate" || exit
        # fi
    else
        echo "Found existing python at: $(which python3)"
    fi
}

########################
# Setup conda on Sirius
########################
setup_conda_sirius() {
    if [[ -z "${CONDA_PREFIX-}" && -z "${VIRTUAL_ENV-}" ]]; then
        export MAMBA_ROOT_PREFIX=/lus/tegu/projects/PolarisAT/foremans/micromamba
        shell_name=$(echo "${SHELL}" | tr "\/" "\t" | awk '{print $NF}')
        eval "$("${MAMBA_ROOT_PREFIX}/bin/micromamba" shell hook --shell ${shell_name})"
        micromamba activate 2024-04-23
    else
        echo "Found existing python at: $(which python3)"
    fi
}

########################
# Setup conda on Polaris
########################
setup_conda_polaris() {
    # unset MPICH_GPU_SUPPORT_ENABLED
    ###### check if CONDA_PREFIX non-empty ################
    if [[ -z "${CONDA_PREFIX-}" ]]; then
        # if so, load the default conda/2024-04-29
        # module and activate base environment
        module use /soft/modulefiles
        module load conda/2024-04-29 ; conda activate base
    else
        echo "Caught CONDA_PREFIX=${CONDA_PREFIX}"
    fi
    ###### check if VIRTUAL_ENV non-empty #################
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        DEFAULT_VENV_PATH=${WORKING_DIR}/venvs/polaris/2024-04-29
        if [[ -d "${DEFAULT_VENV_PATH}" ]]; then
            echo "Caught virtual env at ${DEFAULT_VENV_PATH}!"
            source "${WORKING_DIR}/venvs/polaris/2024-04-29/bin/activate"
        fi
    else
        echo "Found existing python at: $(which python3)"
    fi
}


function setEnv() {
    local virtual_env="${VIRTUAL_ENV-}"
    local conda_prefix="${CONDA_PREFIX-}"
    if [[ -n "${conda_prefix}" && -z "${virtual_env}" ]]; then
        echo "No virtual environment found."
        echo "Using conda from: ${conda_prefix}"
    elif [[ -n "${virtual_env}" && -z "${conda_prefix}" ]]; then
        echo "No conda found."
        echo "Using virtual_env from: ${virtual_env}"
    elif [[ -n "${virtual_env}" && -n "${conda_prefix}" ]]; then
        echo "Using virtual_env: ${virtual_env} on top of conda from: ${conda_prefix}"
    elif [[ -z "${conda_prefix}" && -z "${virtual_env}" ]]; then
        echo "No conda_prefix or virtual_env found in environment..."
        echo "Setting up conda..."
        ######################## setup_conda ############################
        # ---- [SunSpot @ ALCF]  || [Aurora @ ALCF] ---------------------
        if [[ $(hostname) == x1* || $(hostname) == x4* ]]; then
            # ----- [Aurora] --------------------------------------------
            if [[ -z "${conda_prefix}" && -z "${virtual_env}" ]]; then
                if [[ $(hostname) == x4* ]]; then
                    # TODO: Update once Aurora back online
                    eval "$(conda shell.zsh hook)" && conda activate anl_release_q4v2
                # ----- [SunSpot] ---------------------------------------
                elif [[ $(hostname) == x1* ]]; then
                    echo "Running on SunSpot !!"
                    setup_conda_sunspot
                fi
            fi
            source "${WORKING_DIR}/ALCF/sunspot-env.sh" || exit
        # ----- [Polaris @ ALCF] --------------------------------------------
        elif [[ $(hostname) == x3* ]]; then
            if [[ "${PBS_O_HOST}" == sirius* ]]; then
                echo "Running on Sirius !!"
                setup_conda_sirius
            else
                echo "Running on Polaris !!"
                # ---- [load conda] -------------------------------------
                setup_conda_polaris
            fi
        # ----- [Perlmutter @ NERSC] ----------------------------------------
        elif [[ $(hostname) == login* || $(hostname) == nid* ]]; then
            echo "Running on Perlmutter !!"
            module load pytorch
            source "${SLURM_SUBMIT_DIR}/venvs/perlmutter/pytorch-2.1.0-cu12/bin/activate"
        else # ------------------------------------- [Unknown] -------------------
            echo "Unknown hostname $(hostname)"
            exit 1
        fi
    else
        echo "Unable to setup python environment. Exiting"
        exit 1
    fi
    #####################################################################
    pystr="Using: $(which python3)"
    printf "[python] %s" "$(printMagenta ${pystr})"
    printf "\n"
    export "PYTHON_EXEC=$(which python3)"
}


######################################################################
# `makeHostiles`:
#     Detect if `HOSTFILE` set in active environment.
#         - If so, use this.
#         - Otherwise, make default HOSTFILEs from "${PBS_NODEFILE}"
######################################################################
function makeHostfiles() {
    if [[ -n "${HOSTFILE}" ]]; then
        printf "!! USING CUSTOM HOSTFILE FROM: %s"  "${HOSTFILE}"
    else
        make_ds_hostfile
    fi
}

###############################################
# `setData`:
#     Ensure `DATA_FILE_LIST` is set,
#     fallback to default values if necessary.
###############################################
function setData() {  # ------------------------[dfl: abbrv. for DATA_FILE_LIST]
    # dfldir="${WORKING_DIR}/ALCF/data-lists"
    # =====[Set DATA_FILE_LIST_FALLBACK based on current machine]==============
    if [[ $(hostname) == x4* ]]; then    # -----------------------------[AURORA]
        dfl_fallback="/home/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/data_file_list_reweighted.txt"

    elif [[ $(hostname) == x1* ]]; then  # ----------------------------[SUNSPOT]
        # shellcheck: source ./data-lists/sunspot/books.txt
        dfl_fallback="${WORKING_DIR}/ALCF/data-lists/sunspot/books.txt"

    elif [[ $(hostname) == x3* ]]; then  # -------------------[POLARIS / SIRIUS]
        if [[ "${PBS_O_HOST}" == sirius* ]]; then  # -------------------[SIRIUS]
            # shellcheck: source ./data-lists/sirius/books.txt
            dfl_fallback="${WORKING_DIR}/ALCF/data-lists/sirius/books.txt"

        elif [[ "${PBS_O_HOST}" == polaris* ]]; then  # ---------------[POLARIS]
            # shellcheck: source ./data-lists/polaris/books.txt
            dfl_fallback="${WORKING_DIR}/ALCF/data-lists/polaris/dolma_v1_7_file_list.txt"
        fi

    elif [[ $(hostname) == login* || $(hostname) == nid* ]]; then # [PERLMUTTER]
        dfl_fallback="${SLURM_SUBMIT_DIR}/genslm-subsample.txt"

    else  # -----------------------------------------------------------[UNKNOWN]
        echo "Unknown hostname. Must manually specify DATA_FILE_LIST."
    fi
    # ==========================================================================
    # set `dfl` to `dfl_fallback` if not passed as an argument,
    # use this data file list to call `setData`
    dfl="${1:-${dfl_fallback}}"
    printf "Calling:  setData() with %s\n" "${dfl}"
    ndocs=$(wc -l < "${dfl}")
    ws=$(sumWeights "${dfl}")
    dfl_stem=$(echo "${dfl}" | tr "\/" "\t" | awk '{print $NF}' | sed "s/\.txt//g")
    dcp=".cache/${dfl_stem}/index-cache"
    export DATA_FILE_LIST="${dfl}"
    export NUM_DOCS="${ndocs}"
    export WEIGHT_SUM="${ws}"
    export DFL_STEM="${dfl_stem}"
    export DATA_CACHE_PATH="${dcp}"
    echo "--------------------"
    echo "Updated environment:"
    printf "DATA_FILE_LIST: %s\n" "${DATA_FILE_LIST}"
    printf "NUM_DOCS: %s\n " "${NUM_DOCS}"
    printf "WEIGHT_SUM: %s\n" "${WEIGHT_SUM}"
    printf "DFL_STEM: %s\n" "${DFL_STEM}"
    printf "DATA_CACHE_PATH: %s\n" "${DATA_CACHE_PATH}"
    echo "--------------------"
}

function generateDSconfig() {
    for v in "$GLOBAL_BATCH" "$MICRO_BATCH" "$GRAD_ACC_STEPS" "$ZERO_STAGE" \
             "$PP" "$DTYPE"
    do
      if [ -z $v ]; then
        echo "Please export required envs before execute $0"
        exit 1
      fi
    done
    if [ $# -ne 1 ]; then
      echo "Usage: $0 config_file"
      exit 1
    fi
    # \"optimizer\": {
    #   \"type\": \"AdamW\",
    #   \"params\": {
    #     \"lr\": ${LR},
    #     \"beta1\": 0.9,
    #     \"beta2\": 0.95,
    #     \"eps\": 1e-5,
    #     \"weight_decay\": 1e-1
    #   }
    # },
    # \"scheduler\": {
    #   \"type\": \"WarmupLR\",
    #   \"params\": {
    #       \"warmup_min_lr\": 0.00003,
    #       \"warmup_max_lr\": 0.0003,
    #       \"warmup_num_steps\": 5000
    #   }
    # },
    extra=""
    common="\
        \"train_batch_size\": $GLOBAL_BATCH,
        \"train_micro_batch_size_per_gpu\": $MICRO_BATCH,
        \"steps_per_print\": 1,
        \"gradient_accumulation_steps\": $GRAD_ACC_STEPS,
        \"zero_allow_untested_optimizer\": true,
        \"gradient_clipping\": 1.0,
        \"activation_checkpointing\": {
          \"partition_activations\": true,
          \"contiguous_memory_optimization\": true
        },
        \"wall_clock_breakdown\": false,"
    flops_profiler="\
        \"flops_profiler\": {
          \"enabled\": true,
          \"profile_step\": 2,
          \"module_depth\": -1,
          \"top_modules\": 1,
          \"detailed\": true,
          \"output_file\": null
        }"
    if [[ $DTYPE == "bf16" ]]; then
    dtype="\
        \"communication_data_type\": \"bf16\",
        \"fp16\": {
          \"enabled\": false,
          \"loss_scale\": 0,
          \"loss_scale_window\": 1000,
          \"hysteresis\": 2,
          \"min_loss_scale\": 1
        },
        \"bfloat16\": {
          \"enabled\": true,
          \"loss_scale\": 1.0
        },"
    elif [[ $DTYPE == "fp16" ]]; then
    dtype="\
        \"communication_data_type\": \"fp16\",
        \"fp16\": {
          \"enabled\": true,
          \"loss_scale\": 0,
          \"loss_scale_window\": 1000,
          \"hysteresis\": 2,
          \"min_loss_scale\": 1
        },
        \"bfloat16\": {
          \"enabled\": false,
          \"loss_scale\": 1.0
        },"
    else
      dtype="\"communication_data_type\": \"fp32\","
    fi
    if [ $ZERO_STAGE == 3 ]; then
    zero="\
        \"zero_optimization\": {
          \"stage\": 3,
          \"reduce_scatter\": false,
          \"mics_shard_size\": 4,
          \"mics_hierarchical_params_gather\": true,
          \"stage3_max_live_parameters\": 3e9,
          \"stage3_max_reuse_distance\": 3e9,
          \"stage3_param_persistence_threshold\": 1e5,
          \"stage3_prefetch_bucket_size\": 5e7,
          \"contiguous_gradients\": true,
          \"overlap_comm\": true,
          \"reduce_bucket_size\": 90000000,
          \"sub_group_size\": 1e9,
          \"offload_optimizer\": {
            \"device\": \"none\",
            \"buffer_count\": 4,
            \"pipeline_read\": false,
            \"pipeline_write\": false,
            \"pin_memory\": true
          }
        },"
    # elif [[ $ZERO_STAGE == 2 ]]; then
    elif [ "${ZERO_STAGE}" == 2 ] || [ "${ZERO_STAGE}" == 1 ]; then
    # if [[ -n "${CPU_OPTIMIZER}" ]]; then
    if [[ "${CPU_OPTIMIZER}" != 0 ]]; then
    echo "!!!! CAUGHT CPU_OPTIMIZER !!!!"
    zero="\
        \"zero_optimization\": {
            \"stage\": $ZERO_STAGE,
            \"offload_optimizer\": {
              \"device\": \"cpu\"
            }
        },"
    else
    zero="\
        \"zero_optimization\": {
          \"stage\": $ZERO_STAGE
        },"
    fi
    # elif [[ $ZERO_STAGE == 1 ]]; then
    if [[ $PP > 1 ]]; then
      extra="\
          \"data_types\": {
            \"grad_accum_dtype\": \"fp32\"
          },
          \"comms_logger\": {
            \"enabled\": true,
            \"verbose\": false,
            \"prof_all\": true,
            \"debug\": false
          },"
    else
      # echo 'please add the config for zero_stage 1 without pipeline-parallelism'
      extra="\
          \"comms_logger\": {
            \"enabled\": true,
            \"verbose\": false,
            \"prof_all\": true,
            \"debug\": false
          },"
    fi
    else
      echo 'Please add the correct config set!!!'
    fi
# flops_profiler must at the end because no ',' is allowed at the end
cat <<EOT > $1
{
$common
$zero
$dtype
$extra
$flops_profiler
}
EOT
}

function printBlack() {
    printf "\e[1;30m%s\e[0m\n" "$@"
}

function printRed() {
    printf "\e[1;31m%s\e[0m\n" "$@"
}

function printGreen() {
    printf "\e[1;32m%s\e[0m\n" "$@"
}

function printYellow() {
    printf "\e[1;33m%s\e[0m\n" "$@"
}

function printBlue() {
    printf "\e[1;34m%s\e[0m\n" "$@"
}

function printMagenta() {
    printf "\e[1;35m%s\e[0m\n" "$@"
}

function printCyan() {
    printf "\e[1;36m%s\e[0m\n" "$@"
}

function printWhite() {
    printf "\e[1;37m%s\e[0m\n" "$@"
}
