#!/bin/bash --login
#PBS -l walltime=06:00:00
#PBS -A argonne_tpc
#PBS -q prod
#PBS -l select=48
#PBS -l filesystems=eagle:home

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

# +++++++++++++++ SCRIPT START +++++++++++++++++++++++
# ---- source ./helpers_alcf.sh ---------------------
HERE=$(python3 -c 'import os; print(os.getcwd())')
sourceFile "${HERE}/helpers_alcf.sh" || exit

# ---- load conda -----------------------------------
module load conda/2023-10-04; conda activate base
echo "Using $(which python3)"

# ---- fns from ./helpers_alcf.sh -------------------
ezpz
makeHostfiles
saveDSenv
# setupData "${DOLMA_CHUNK_IDX:-00}"
# export DOLMA_CHUNK_IDX="${DOLMA_CHUNK_IDX:-0}"
#
# ---- DATA SETUP ------------------------------------
DATA_FILE_LIST="./data_file_list_shuf_debug.txt" && export DATA_FILE_LIST="${DATA_FILE_LIST}"
NDOCS=$(wc -l < "${DATA_FILE_LIST}") && export NDOCS="${NDOCS}"
WEIGHT_SUM="$(sumWeights "${DATA_FILE_LIST}")" && export WEIGHT_SUM="${WEIGHT_SUM}"
DFL_STEM=$(echo "$DATA_FILE_LIST" | tr "\/" "\t" | awk '{print $NF}' | sed "s/\.txt//g") && export DFL_STEM="${DFL_STEM}"
dcp="${HERE}/.cache/${DFL_STEM}-index-cache"
DATA_CACHE_PATH="${DATA_CACHE_PATH:-${dcp}}" && export DATA_CACHE_PATH="${DATA_CACHE_PATH}"
mkdir -p "${DATA_CACHE_PATH}"
if [[ -n "${DOLMA_CHUNK_IDX}" ]]; then
    echo "Using DOLMA CHUNK ${DOLMA_CHUNK_IDX} from ${DATA_FILE_LIST} with ${NDOCS} documents..."
else
    echo "Using NDOCS=${NDOCS} documents from DATA_FILE_LIST=${DATA_FILE_LIST}"
fi
echo "DOCUMENT WEIGHT_SUM: ${WEIGHT_SUM}"
# ----------------------------------------------------

# ---- Parallelism Settings --------------------------
PP=${PP:-1}
TP=${TP:-2}
export PP="${PP}"
export TP="${TP}"
export HOSTFILE="${HOSTFILE:-${PBS_NODEFILE}}"
export WORLD_SIZE=${WORLD_SIZE:-$(wc -l < "${HOSTFILE}")}
# ----------------------------------------------------

# ---- Llama2 7B Config ------------------------------
export HEADS=${HEADS:-32}
export NLAYERS=${NLAYERS:-32}
export HIDDEN=${HIDDEN:-4096}
export NUM_KV_HEAD=${NUM_KV_HEAD:-8}
# ----------------------------------------------------

# ---- Run Settings ----------------------------------
export LR=${LR:-0.00015}
export SEQ=${SEQ:-4096}                       # SEQ_LEN: 4096
export DTYPE=${DTYPE:-fp16}                   # DTYPE: FP16
export ZERO_STAGE=${ZERO_STAGE:-2}
export MICRO_BATCH=${MICRO_BATCH:-8}
export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-"/eagle/datasets/dolma/utils/tokenizer.model"}"
export TRAIN_ITER=${TRAIN_ITER:-317892}
# export TRAIN_ITER="${TRAIN_ITER:-320000}"
export EVAL_ITERS="${EVAL_ITERS:-10}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-50000}"
export SAVE_INTERVAL=${SAVE_INTERVAL:-200}
export USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING:-1}
# export USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING:-0}
export GLOBAL_BATCH=$(( $WORLD_SIZE * $MICRO_BATCH * $GRAD_ACC_STEPS / $TP / $PP ))
export MODEL_TYPE="llama-seq${SEQ}-pp${PP}-tp${TP}-${NLAYERS}layers-${HEADS}heads-${HIDDEN}hidden"
export LLAMA_ARGS="--no-query-key-layer-scaling --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear"
# ----------------------------------------------------

echo "++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "- WORLD_SIZE:${WORLD_SIZE}"
echo "- NCCL: ${NCCL:-nccl}"
echo "- MODEL_TYPE: ${MODEL_TYPE}"
echo "- Using DATA_FILE_LIST: ${DATA_FILE_LIST}"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++"

# if [[ "${DOLMA_CHUNK_IDX}" == 0 ]]; then
#     TRAIN_ITER=78739
# elif [[ "${DOLMA_CHUNK_IDX}" == 1 ]]; then
#     TRAIN_ITER=81008
# elif [[ "${DOLMA_CHUNK_IDX}" == 2 ]]; then
#     TRAIN_ITER=79591
# elif [[ "${DOLMA_CHUNK_IDX}" == 3 ]]; then
#     TRAIN_ITER=78552
# else
#     echo "caught DOLMA_CHUNK_IDX=${DOLMA_CHUNK_IDX}"
#     TRAIN_ITER="${TRAIN_ITER:-320000}"
#     echo "Setting TRAIN_ITER=${TRAIN_ITER}"
#     # echo "Unknown DOLMA_CHUNK_IDX: ${DOLMA_CHUNK_IDX}"
# fi

# +++++NOTES ++++++++++++++++++++++++++++++++++++++++++++++++++
# XXX:
# - need to merge *.json files
# - Can we create indices on a per-dataset basis?
#   (i.e. one for common-crawl, one for stack-code, etc.)
# - Aggregate `stack-code/**/{*.bin,*.idx}`
#
# - Given: {f1.bin,f2.bin,...,fn.bin}
#   - tot_tokens = 0
#   - agg = []
#   - Start:
#     - read: f1.bin
#     - tot_tokens += sum(tokens(f1.bin))
#     - if tot_tokens < needed_tokens:
#         - agg.append(f1.bin)
#     - else:
#

# TODO:
# - StackExchange ~ 500B total, using 80% ~ 400B tokens
# - figure out how to deal with MANY small files (e.g. stack-code)
# - Add logic for determining `train_iters` dynamically from `data-file-list` 
#   (which specifies a single _chunk_)
# - get script from Varuni
# - should:
#   - take in a `data_file_list.txt`
#   - return number of training iterations
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ---- Build DeepSpeed Config ---------------------------------
DS_CONFIG="ds_stage${ZERO_STAGE}_mb${MICRO_BATCH}_gb${GLOBAL_BATCH}_pp${PP}_${DTYPE}.json"
bash "${HERE}/generate_config.sh" "${DS_CONFIG}" || exit 1
# -------------------------------------------------------------

# ---- Specify output location --------------------------------
OUTPUT_PREFIX="${HERE}/logs/ds_stage${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_mb${MICRO_BATCH}_seq${SEQ}_gb${GLOBAL_BATCH}_pp${PP}_tp${TP}_${DTYPE}"
# OUTPUT_DIR=logs/ds_stage${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_mb${MICRO_BATCH}_seq${SEQ}_gb${GLOBAL_BATCH}_pp${PP}_tp${TP}_${DTYPE}_`date +%m%d%H%M%S`_${HOSTNAME}
OUTPUT_DIR="${OUTPUT_PREFIX}/$(date +%m%d%H%M%S)_${HOSTNAME}"
mkdir -p "${OUTPUT_DIR}"
echo "!!!Please see logs at ${OUTPUT_DIR}"


# ---- Setup DeepSpeed arguments --------------------------------
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
# ---------------------------------------------------------------

gpt_args=()

# we are now using activation checkpoint provided by megatron, see below.
# ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
if [[ "$USE_ACTIVATION_CHECKPOINTING" == 1 ]]; then
    echo "!! Caught USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING} !!"
    gpt_args+=(
        "--checkpoint-activations"
        "--checkpoint-num-layers 1"
    )
fi

# take custom args
custom_args=" $@"

# launcher setting
hfds="${HERE}/hostfile_deepspeed"
hfmpi="${HERE}/hostfile_mpich"
[ -f "$hfds" ] || exit
[ -f "$hfmpi" ] || exit

LAUNCHER=${LAUNCHER:-MPICH}
if [[ $LAUNCHER == "deepspeed" ]]; then
    launcher=""
else
    launcher="--force_multi --hostfile $hfds --launcher=${LAUNCHER} --launcher_args='-hostfile ${hfmpi}'"
fi

NCCL=${NCCL:-nccl}
EXEC="pretrain_gpt_alcf.py"

# --vocab-file $VOCAB_FILE \
# --merge-file $MERGE_FILE \
# --lr-decay-iters 320000 \
    # --num-workers 0 \
    # --lr-warmup-iters 5000 \
    # --lr-decay-iters 10000 \
    # --num-workers 4 \
    # launch python3 ${EXEC} \
run_cmd="
    deepspeed $launcher ${EXEC} \
    --$DTYPE \
    --lr ${LR} \
    --log-interval 1 \
    --seq-length $SEQ \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --ffn-hidden-size 11008 \
    --train-iters $TRAIN_ITER \
    --eval-iters ${EVAL_ITERS} \
    --distributed-backend $NCCL \
    --num-attention-heads $HEADS \
    --max-position-embeddings $SEQ \
    --micro-batch-size $MICRO_BATCH \
    --save-interval ${SAVE_INTERVAL} \
    --eval-interval ${EVAL_INTERVAL} \
    --tensor-model-parallel-size $TP \
    --global-batch-size $GLOBAL_BATCH \
    --pipeline-model-parallel-size $PP \
    --data-file-list ${DATA_FILE_LIST} \
    --load checkpoints/${OUTPUT_PREFIX} \
    --save checkpoints/${OUTPUT_PREFIX} \
    --data-cache-path ${DATA_CACHE_PATH} \
    --num-key-value-heads ${NUM_KV_HEAD} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --split 90,5,5 \
    --data-impl mmap \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-gradient-accumulation-fusion \
    --use-flash-attn-v2 \
    --lr-decay-style cosine \
    --tokenizer-type Llama2Tokenizer \
    --use-checkpoint-opt_param-scheduler \
    --accumulate-allreduce-grads-in-fp32 \
    $ds_args \
    ${LLAMA_ARGS} \
    ${gpt_args[*]} \
    $custom_args \
    |& tee $OUTPUT_DIR/output.log
    "
    # ${EXTRA_ARGS} \

echo "Using $(which deepspeed)"
ds_report

echo ${run_cmd}
eval ${run_cmd}
set +x
