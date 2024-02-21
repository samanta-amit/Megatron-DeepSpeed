#!/bin/bash login
# echo "!!!please use generate_hostfile.sh to set hostfile for 18 nodes before training"
export WORLD_SIZE=${WORLD_SIZE:-$(wc -l < "${PBS_NODEFILE}")}
export MICRO_BATCH=${MICRO_BATCH:-1}
export NLAYERS=${NLAYERS:-96}
export HIDDEN=${HIDDEN:-12288}
export HEADS=${HEADS:-96}
export LR=${LR:-0.0003}
export SEQ=${SEQ:-4096}
export TRAIN_ITER=${TRAIN_ITER:-300000}
export EVAL_ITERS=${EVAL_ITERS:-50}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
export EVAL_INTERVAL=${EVAL_INTERVAL:-50000}
export ZERO_STAGE=${ZERO_STAGE:-2}
export DTYPE=${DTYPE:-fp16}
export TP=${TP:-2}
export PP=${PP:-1}
export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}
export GLOBAL_BATCH=$(( $WORLD_SIZE * $MICRO_BATCH * $GRAD_ACC_STEPS / $TP / $PP ))
export USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING:-0}

# echo "USING DATA_FILE_LIST: ${DATA_FILE_LIST}" || exit


# bash $LLM_DK_DIR/intel-extension-for-deepspeed/examples/gpt.sh $@

# Disabling tensor/pipeline parallelism
TP=${TP:-1}
PP=${PP:-1}

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

# MODEL=LLAMA_7B
# OUTPUT_PREFIX=${MODEL}_z${ZERO_STAGE}_seqlen_tp${TP}_pp${PP}_sp${SP}_nl${NUM_LAYERS}_hs${HIDDEN_SIZE}_gb${BS}_mb${MBS}

# --vocab-file $VOCAB_FILE \
# --merge-file $MERGE_FILE \
# --lr-decay-iters 320000 \
run_cmd="
    deepspeed $launcher pretrain_gpt_alcf.py \
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
    --num-workers 0 \
    --tokenizer-type Llama2Tokenizer \
    --save checkpoints/${OUTPUT_PREFIX} \
    --load checkpoints/${OUTPUT_PREFIX} \
    --use-checkpoint-opt_param-scheduler \
    --accumulate-allreduce-grads-in-fp32 \
    --tokenizer-model /eagle/datasets/dolma/utils/tokenizer.model \
    --data-file-list ${DATA_FILE_LIST} \
    ${gpt_args[*]} \
    $custom_args \
    |& tee $OUTPUT_DIR/output.log
    "

echo "Using $(which deepspeed)"
ds_report

echo ${run_cmd}
eval ${run_cmd}
set +x
