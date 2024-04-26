#!/bin/bash --login
#PBS -l walltime=06:00:00
#PBS -A argonne_tpc
#PBS -q prod
#PBS -l select=48
#PBS -l filesystems=eagle:home

if [[ -n "${DEBUG-}" ]]; then
    printf "\e[1;31m%s\e[0m\n" "!! RUNNING IN DEBUG MODE !!"
    set -euxo pipefail
fi

function sourceFile() {
    fp="$1"
    echo "source-ing ${fp}"
    if [[ -f "${fp}" ]]; then
        # shellcheck source="${fp}"
        source "${fp}"
    else
        echo "ERROR: UNABLE TO SOURCE ${fp}"
    fi
}

# ----[0. Navigate into `$PBS_O_WORKDIR`]-------------------------------------
cd "${PBS_O_WORKDIR}" || exit
HERE=$(python3 -c 'import os; print(os.getcwd())')
export HERE

# ----[1. Assert `./pretrain_gpt_alcf.py` exists:]-----------------------------
export EXEC="${HERE}/pretrain_gpt_alcf.py"
[ -f "${EXEC}" ] || exit

# ----[2. `source ./ALCF/helpers_alcf.sh`:]------------------------------------
sourceFile "${HERE}/ALCF/helpers.sh" || exit

# ----[3. Call fns from `./ALCF/helpers_alcf.sh`]------------------------------
setEnv || exit                      # 1. load `conda` environment
# saveDSenv || exit                 # 2. save env vars to `.deepspeed_env`
ezpz || exit                        # 3. determine WORLD_SIZE, etc. from `PBS_*` vars
setParams || exit                   # 5. set command line arguments to pass to `"${EXEC}"`
buildDSconfig || exit               # 6. create `deepspeed_config.json` from runtime params from ^
setOutput || exit                   # 7. specify output directory for {logs, checkpoints, etc.}
setArgs || exit                     # 8. specify additional `deepspeed` arguments
setData "${DATA_FILE_LIST-}" || exit  # 9. specify `DATA_FILE_LIST` for dolma dataset
printJobInfo || exit                # 11. print job info
setupLauncher || exit
# -----------------------------------------------------------------------------

#### [DEPRECATED] ###########################################################
# if [[ -z "${HOSTFILE}" ]]; then
#     makeHostfiles || exit         # 4. create `deepspeed` hostfile from `$PBS_NODEFILE`
# else
#     echo "!! USING CUSTOM HOSTFILE FROM: ${HOSTFILE}"
# fi
# ----------------------------------------------------------------------------
# setDSlauncher "${HERE}" || exit   # 10. set `launcher` args for `deepspeed ${launcher} ${EXEC} ${args}`
# ----------------------------------------------------------------------------
# TORCH_DEVICE=$(python3 -c 'import ezpz as ez; print(ez.get_torch_device())')
# printf %s "Using TORCH_DEVICE=${TORCH_DEVICE}"
# if [[ "${TORCH_DEVICE}" == "cuda" ]]; then
#     printf %s "Setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
#     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# fi
# ----------------------------------------------------------------------------
# export MPICH_GPU_SUPPORT_ENABLED=1
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_DEBUG=INFO
#############################################################################

# Assert TBDIR exists inside our $CKPT_DIR
TBDIR="${CKPT_DIR}/tensorboard"
mkdir -p "${TBDIR}"

data_cache_path="${CKPT_DIR}/${DATA_CACHE_PATH}"
mkdir -p "${data_cache_path}"
module list

if [[ "${TIMING_LOG_LEVEL}" -ge 1 ]]; then
    TIMING_STR="\
        --timing-log-level ${TIMING_LOG_LEVEL} \
        --log-timers-to-tensorboard \
        --log-optimizer-states-to-tensorboard \
    "
else
    TIMING_STR=""
fi


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
    --lr-decay-style cosine \
    --no-bias-dropout-fusion \
    --no-masked-softmax-fusion \
    --tokenizer-type Llama2Tokenizer \
    --no-gradient-accumulation-fusion \
    --accumulate-allreduce-grads-in-fp32 \
    --use-checkpoint-opt_param-scheduler \
    --log-timers-to-tensorboard \
    --log-optimizer-states-to-tensorboard \
    --lr ${LR} \
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
    --lr-warmup-fraction ${LR_WARMUP_FRAC} \
    ${LLAMA_ARGS} \
    ${TIMING_STR} \
    $ds_args \
    ${gpt_args[*]} \
    $custom_args \
    |& tee ${OUTPUT_LOG}
    "

check_and_kill_if_running || exit
echo "${run_cmd}"
printf "[!! \e[1;31m%s\e[0m] View output at:\n" "NOTE"
printf "\e[1;34m%s\e[0m\n" "${OUTPUT_LOG}"
eval "${run_cmd}"
set +x
