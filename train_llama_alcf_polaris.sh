#!/bin/bash --login
#PBS -l walltime=06:00:00
#PBS -A argonne_tpc
#PBS -q prod
#PBS -l select=48
#PBS -l filesystems=eagle:home

cd "${PBS_O_WORKDIR}" || exit
module load conda/2023-10-04; conda activate base
if [[ ! -d ezpz ]]; then
  git clone https://github.com/saforem2/ezpz
else
  echo "Found ezpz!"
fi
source ezpz/src/ezpz/bin/savejobenv > /tmp/savejobenv.log 2>&1 || exit
source ezpz/src/ezpz/bin/getjobenv || exit

# ---- Parallelism Settings ----
export PP=${PP:-1}
export TP=${TP:-2}
# ------------------------------

HOSTFILE="${HOSTFILE:-${PBS_NODEFILE}}"
export WORLD_SIZE=${WORLD_SIZE:-$(wc -l < "${HOSTFILE}")}

# ---- Llama2 7B Config -----------------------
export HEADS=${HEADS:-32}
export NLAYERS=${NLAYERS:-32}
export HIDDEN=${HIDDEN:-4096}
export NUM_KV_HEAD=${NUM_KV_HEAD:-8}
# ---------------------------------------------

# ---- Run Settings ------------------------------------------
export LR=${LR:-0.00015}
export SEQ=${SEQ:-4096}                       # SEQ_LEN: 4096
export DTYPE=${DTYPE:-fp16}                   # DTYPE: FP16
export ZERO_STAGE=${ZERO_STAGE:-2}
export MICRO_BATCH=${MICRO_BATCH:-8}
export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}
export GLOBAL_BATCH=$(( $WORLD_SIZE * $MICRO_BATCH * $GRAD_ACC_STEPS / $TP / $PP ))
# ---------------------------------------------

export EVAL_ITERS=${EVAL_ITERS:-20}
export TRAIN_ITER=${TRAIN_ITER:-317892}
export SAVE_INTERVAL=${SAVE_INTERVAL:-200}
export EVAL_INTERVAL=${EVAL_INTERVAL:-50000}
export USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING:-1}
# export USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING:-0}

# export DATA_FILE_LIST="/lus/eagle/projects/datasets/dolma/chunks/40/data_file_list_chunk_0_of_40.txt"
# export DATA_FILE_LIST="/lus/eagle/projects/datasets/dolma/chunks/10/data_file_list_chunk_0_of_10.txt"
export DATA_FILE_LIST="./dolma_data_file_list-00-of-04.txt"
# export DATA_FILE_LIST="./dolma-chunk-00-of-40.txt"

export MODEL_TYPE="llama-seq${SEQ}-pp${PP}-tp${TP}-${NLAYERS}layers-${HEADS}heads-${HIDDEN}hidden"

export LLAMA_ARGS="--no-query-key-layer-scaling --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear"
export EXTRA_ARGS="--use-flash-attn-v2 --num-key-value-heads ${NUM_KV_HEAD}"

# export DATA_CACHE_PATH="${DATA_CACHE_PATH}"
if [[ -n "$DATA_CACHE_PATH" ]]; then
    echo "Using DATA_CACHE_PATH: ${DATA_CACHE_PATH}"
    EXTRA_ARGS="${EXTRA_ARGS} --data-cache-path ${DATA_CACHE_PATH}"
else
    echo "Not using DATA_CACHE_PATH !!"
fi

echo "++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "- WORLD_SIZE:${WORLD_SIZE}"
echo "- NCCL: ${NCCL:-nccl}"
echo "- MODEL_TYPE: ${MODEL_TYPE}"
echo "- Using DATA_FILE_LIST: ${DATA_FILE_LIST}"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++"


# bash $LLM_DK_DIR/intel-extension-for-deepspeed/examples/gpt.sh $@

DS_CONFIG="ds_stage${ZERO_STAGE}_mb${MICRO_BATCH}_gb${GLOBAL_BATCH}_pp${PP}_${DTYPE}.json"
bash ./generate_config.sh "${DS_CONFIG}" || exit 1

OUTPUT_PREFIX="logs/ds_stage${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_mb${MICRO_BATCH}_seq${SEQ}_gb${GLOBAL_BATCH}_pp${PP}_tp${TP}_${DTYPE}"
# OUTPUT_DIR=logs/ds_stage${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_mb${MICRO_BATCH}_seq${SEQ}_gb${GLOBAL_BATCH}_pp${PP}_tp${TP}_${DTYPE}_`date +%m%d%H%M%S`_${HOSTNAME}
OUTPUT_DIR="${OUTPUT_PREFIX}/$(date +%m%d%H%M%S)_${HOSTNAME}"
mkdir -p "${OUTPUT_DIR}"
echo "!!!Please see logs at ${OUTPUT_DIR}"

# Hostfile path
hostfile_deepspeed=./hostfile_deepspeed
hostfile_mpich=./hostfile_mpich
cat "$PBS_NODEFILE" > hostfile_mpich
cat "$PBS_NODEFILE" > hostfile_deepspeed ; sed -e 's/$/ slots=4/' -i hostfile_deepspeed

ds_args=" "
ds_args=" --deepspeed ${ds_args}"
if [ "$PP" == 1 ]; then
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

gpt_args=()

if [[ "$USE_ACTIVATION_CHECKPOINTING" == 1 ]]; then
    echo "!! Caught USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING} !!"
    gpt_args+=(
        "--checkpoint-activations"
        "--checkpoint-num-layers 1"
    )
fi
# we are now using activation checkpoint provided by megatron, see below.
# ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
# NUM_KV_HEADS="${NUM_KV_HEADS:-0}"
# if [[ $NUM_KV_HEADS -]]

# take custom args
custom_args=" $@"

# launcher setting
LAUNCHER=${LAUNCHER:-MPICH}
if [[ $LAUNCHER == "deepspeed" ]]; then
    launcher=""
else
    launcher="--force_multi --hostfile $hostfile_deepspeed --launcher=${LAUNCHER} --launcher_args='-hostfile ${hostfile_mpich}'"
fi

NCCL=${NCCL:-nccl}
EXEC="./pretrain_gpt_alcf.py"

# MODEL=LLAMA_7B
# OUTPUT_PREFIX=${MODEL}_z${ZERO_STAGE}_seqlen_tp${TP}_pp${PP}_sp${SP}_nl${NUM_LAYERS}_hs${HIDDEN_SIZE}_gb${BS}_mb${MBS}

# --vocab-file $VOCAB_FILE \
# --merge-file $MERGE_FILE \
# --lr-decay-iters 320000 \
    # --num-workers 0 \
run_cmd="
    deepspeed $launcher ${EXEC} \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads $HEADS \
    --seq-length $SEQ \
    --max-position-embeddings $SEQ \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters $TRAIN_ITER \
    --lr ${LR} \
    --lr-warmup-iters 5000 \
    --lr-decay-iters 10000 \
    --ffn-hidden-size 11008 \
    --lr-decay-style cosine \
    --data-impl mmap \
    --log-interval 1 \
    --eval-iters ${EVAL_ITERS} \
    --eval-interval ${EVAL_INTERVAL} \
    --save-interval ${SAVE_INTERVAL} \
    --split 90,5,5 \
    --$DTYPE \
    $ds_args \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-gradient-accumulation-fusion \
    --distributed-backend $NCCL \
    --tokenizer-type Llama2Tokenizer \
    --save checkpoints/${OUTPUT_PREFIX} \
    --load checkpoints/${OUTPUT_PREFIX} \
    --use-checkpoint-opt_param-scheduler \
    --accumulate-allreduce-grads-in-fp32 \
    --tokenizer-model /eagle/datasets/dolma/utils/tokenizer.model \
    --data-file-list ${DATA_FILE_LIST} \
    --num-workers 4 \
    ${LLAMA_ARGS} \
    ${EXTRA_ARGS} \
    ${gpt_args[*]} \
    $custom_args \
    |& tee $OUTPUT_DIR/output.log
    "

echo "Using $(which deepspeed)"
ds_report

echo ${run_cmd}
eval ${run_cmd}
set +x
