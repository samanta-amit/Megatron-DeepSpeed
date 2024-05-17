#!/bin/bash --login
#PBS -l walltime=06:00:00
#PBS -A argonne_tpc
#PBS -q prod
#PBS -l select=48
#PBS -l filesystems=eagle:home

#### Make it easy to track experiments by date ###################
YEAR="$(date "+%Y")"
MONTH="$(date "+%m")"
DAY="$(date "+%Y-%m-%d")"
TODAY="$(date "+%Y-%m-%d")"  # kept for backwards compatibility
STARTED_AT="$(date "+%Y-%m-%d-%H%M%S")"
##################################################################

if [[ -n "${DEBUG-}" ]]; then  # to use: `DEBUG=1 bash train_llama_alcf.sh`
    printf "\e[1;31m%s\e[0m\n" "!! RUNNING IN DEBUG MODE !!"
    set -euxo pipefail
fi

if [[ -v NOOP ]]; then         # to use: `NOOP=1 bash train_llama_alcf.sh`
  echo "Run NOOP mode"
  set -o noexec                # same as set -n
fi

sourceFile() {
    fp="$1"
    echo "source-ing ${fp}"
    if [[ -f "${fp}" ]]; then
        # shellcheck source="${fp}"
        source "${fp}"
    else
        echo "ERROR: UNABLE TO SOURCE ${fp}"
    fi
}

# ----[0. Navigate into `$PBS_O_WORKDIR`]--------------------------------------
cd "${PBS_O_WORKDIR}" || exit
HERE=$(python3 -c 'import os; print(os.getcwd())')
export HERE

# ----[1. Assert `./pretrain_gpt_alcf.py` exists:]-----------------------------
export EXEC="${HERE}/pretrain_gpt_alcf.py"
[ -f "${EXEC}" ] || exit

# ----[2. `source ./ALCF/helpers_alcf.sh`:]------------------------------------
sourceFile "${HERE}/ALCF/helpers.sh" || exit

# ----[3. Call fns from `./ALCF/helpers_alcf.sh`]------------------------------
get_machine || exit                   # 01. Identify machine we're on
setEnv || exit                        # 02. Load `conda` environment
# saveDSenv || exit                   # 03. Save env vars to `.deepspeed_env`
ezpz || exit                          # 04. Determine WORLD_SIZE, etc. from `PBS_*` vars
setParams || exit                     # 05. Set command line arguments to pass to `"${EXEC}"`
buildDSconfig || exit                 # 06. Create `deepspeed_config.json` from runtime params from ^
setOutput || exit                     # 07. Specify output directory for {logs, checkpoints, etc.}
setArgs || exit                       # 08. Specify additional `deepspeed` arguments
setData "${DATA_FILE_LIST-}" || exit  # 09. Specify `DATA_FILE_LIST` for dolma dataset
printJobInfo || exit                  # 11. Print job info
setupLauncher || exit                 # 12. set launcher to one of `MPICH` (default), or `deepspeed`
# -----------------------------------------------------------------------------

################################################
# Assert `$TBDIR` exists inside our `$CKPT_DIR`
# to ensure metrics are tied to checkpoint
################################################
TBDIR="${CKPT_DIR}/tensorboard"
mkdir -p "${TBDIR}"

data_cache_path="${CKPT_DIR}/${DATA_CACHE_PATH}" && mkdir -p "${data_cache_path}"

# Print info about loaded modules and runtime environment
module list
printenv |& tee "${CKPT_DIR}/.env"

# Take custom args
custom_args=" $@"

    # --log-num-zeros-in-grad \
    # --log-memory-to-tensorboard \
run_cmd="
    ${LAUNCH_CMD} \
    --${DTYPE} \
    --split 100,0,0 \
    --log-interval 1 \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-masked-softmax-fusion \
    --tokenizer-type Llama2Tokenizer \
    --no-gradient-accumulation-fusion \
    --accumulate-allreduce-grads-in-fp32 \
    --use-checkpoint-opt_param-scheduler \
    --log-timers-to-tensorboard \
    --log-optimizer-states-to-tensorboard \
    --optimizer ${OPT} \
    --save ${CKPT_DIR} \
    --load ${CKPT_DIR} \
    --seq-length ${SEQ} \
    --num-layers ${NLAYERS} \
    --hidden-size ${HIDDEN} \
    --train-iters ${TRAIN_ITER} \
    --tensorboard-dir ${TBDIR} \
    --eval-iters ${EVAL_ITERS} \
    --distributed-backend ${BE} \
    --num-attention-heads ${HEADS} \
    --save-interval ${SAVE_INTERVAL} \
    --eval-interval ${EVAL_INTERVAL} \
    --max-position-embeddings ${SEQ} \
    --micro-batch-size ${MICRO_BATCH} \
    --data-file-list ${DATA_FILE_LIST} \
    --tensor-model-parallel-size ${TP} \
    --global-batch-size ${GLOBAL_BATCH} \
    --pipeline-model-parallel-size ${PP} \
    --num-key-value-heads ${NUM_KV_HEAD} \
    --data-cache-path ${data_cache_path} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    ${LR_ARGS} \
    ${LLAMA_ARGS} \
    ${TIMING_STR} \
    $ds_args \
    ${gpt_args[*]} \
    $custom_args \
    |& tee ${OUTPUT_LOG}
    "

check_and_kill_if_running || exit
echo "${run_cmd}"
printf "[!! %s] View output at:\n %s\n" "$(printBlue "NOTE")" "$(printYellow ${OUTPUT_LOG})"
# printf "[!! \e[1;31m%s\e[0m] View output at:\n" "NOTE"
# printf "\e[1;34m%s\e[0m\n" "${OUTPUT_LOG}"
eval "${run_cmd}"
set +x
