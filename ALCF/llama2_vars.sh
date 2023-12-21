#!/bin/bash
# This example script is contributed by external user https://github.com/nrailgun
# set -ex

######################################
# Change the below configurations here
# wget https://the-eye.eu/public/AI/pile_neox/data/BookCorpusDataset_text_document.bin
# wget https://the-eye.eu/public/AI/pile_neox/data/BookCorpusDataset_text_document.idx
MEGATRON_DIR="/home/foremans/datascience/foremans/locations/thetaGPU/projects/argonne-lcf/Megatron-DeepSpeed"
DATA_DIR="~/datascience/foremans/locations/thetaGPU/projects/saforem2/Megatron-DeepSpeed/dataset/"
BASE_PATH="${MEGATRON_DIR}"
DS_CONFIG=${BASE_PATH}/deepspeed.json
DATASET_1="${DATA_DIR}/BookCorpusDataset_text_document"
# DATASET_1="./tmp/data/bookcorpus_train_1m_text_sentence"
DATASET="1 ${DATASET_1}"
CHECKPOINT_PATH=./tmp
TOKENIZER_PATH=./tmp/tokenizer.model # offical llama tokenizer.model

if [[ $(hostname) == nid* || $(hostname) == login* ]]; then
    DATA_PARENT="/global/homes/f/foremans/m3957/foremans/projects/saforem2/Megatron-DeepSpeed"
    DATA_TYPE="BookCorpusDataset_text_document"
elif [[ $(hostname) == theta* || $(hostname) == x3* ]]; then
    DATA_PARENT="/lus/grand/projects/fallwkshp23/datasets/GenSLMSubSample200k"
    DATA_TYPE="genslm_subsample_200k_sequence_document"
else
    echo "Unable to determine DATA_PARENT for $(hostname)."
    echo "Exiting!"
    exit 1
fi

DATA_DIR="${DATA_PARENT}/dataset"
DATA_PATH="${DATA_DIR}/${DATA_TYPE}"
VOCAB_FILE="${DATA_DIR}/gpt2-vocab.json"
MERGE_FILE="${DATA_DIR}/gpt2-merges.txt"

DATA_LOAD_ARGS=(
    "--data-path $DATA_PATH"
    "--vocab-file $VOCAB_FILE"
    "--merge-file $MERGE_FILE"
)

TP=2
PP=2
ZERO_STAGE=0

GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=$(wc -l < "${PBS_NODEFILE:-${COBALT_NODEFILE:-1}}")
NODE_RANK=0

HIDDEN_SIZE=2048 # e.g. llama-13b: 5120
FFN_HIDDEN_SIZE=5504 # e.g. llama-13b: 13824
NUM_LAYERS=24 # e.g. llama-13b: 40
NUM_HEADS=16 # e.g. llama-13b: 40
SEQ_LENGTH=2048
NUM_KV_HEADS=4 # llama2 70B uses GQA

MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=32 # e.g. llama: 4M tokens
TRAIN_STEPS=250000 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="true"
activation_checkpoint="false"

# Below configuration required for llama model as per llama paper
# --no-query-key-layer-scaling \
# --attention-dropout 0 \
# --hidden-dropout 0 \
# --use-rotary-position-embeddings \
# --untie-embeddings-and-output-weights \
# --swiglu \
# --normalization rmsnorm \
# --disable-bias-linear \
######################################



cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": true
  }
}
EOT

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"

if [ "${activation_checkpoint}" = "true" ]; then
  ds_args="--deepspeed-activation-checkpointing ${ds_args}"

  ## old argument for recomputing the transformer layer
  # ds_args="--checkpoint-activations ${ds_args}"

  ## new argument for recomputing the transformer layer
  ds_args="--recompute-granularity full --recompute-method uniform ${ds_args}"
  ## new argument for recomputing only the attention layer
  # ds_args="--recompute-granularity selective ${ds_args}"
fi


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# torchrun $DISTRIBUTED_ARGS \
#        pretrain_gpt.py \
#       
MEGATRON_ARGS=(
    "--tensor-model-parallel-size $TP"
    "--pipeline-model-parallel-size $PP"
    "--num-layers $NUM_LAYERS"
    "--hidden-size $HIDDEN_SIZE"
    "--ffn-hidden-size $FFN_HIDDEN_SIZE"
    "--num-attention-heads $NUM_HEADS"
    "--micro-batch-size $MICRO_BATCH_SIZE"
    "--global-batch-size $GLOBAL_BATCH_SIZE"
    "--seq-length $SEQ_LENGTH"
    "--max-position-embeddings $SEQ_LENGTH"
    "--train-iters $TRAIN_STEPS"
    "--save $CHECKPOINT_PATH"
    "--load $CHECKPOINT_PATH"
    "--data-path $DATASET"
    "--data-impl mmap"
    "--tokenizer-type GPTSentencePieceTokenizer"
    "--tokenizer-model $TOKENIZER_PATH"
    "--split 949,50,1"
    "--distributed-backend nccl"
    "--lr $LR"
    "--lr-decay-style cosine"
    "--min-lr $MIN_LR"
    "--weight-decay $WEIGHT_DECAY"
    "--clip-grad $GRAD_CLIP"
    "--lr-warmup-iters $LR_WARMUP_STEPS"
    "--optimizer adam"
    "--adam-beta1 0.9"
    "--adam-beta2 0.95"
    "--log-interval 1"
    "--save-interval 10000"
    "--eval-interval 1000"
    "--eval-iters 10"
    "--bf16"
    "--no-query-key-layer-scaling"
    "--attention-dropout 0"
    "--hidden-dropout 0"
    "--use-rotary-position-embeddings"
    "--untie-embeddings-and-output-weights"
    "--swiglu"
    "--normalization rmsnorm"
    "--disable-bias-linear"
    "--num-key-value-heads $NUM_KV_HEADS"
)

MEGATRON_ARGS+="${ds_args}"
MEGATRON_ARGS+="${DATA_LOAD_ARGS}"
