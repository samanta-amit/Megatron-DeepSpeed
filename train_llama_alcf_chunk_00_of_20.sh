#!/bin/bash --login
#PBS -l walltime=06:00:00
#PBS -A argonne_tpc
#PBS -q prod
#PBS -l select=48
#PBS -l filesystems=eagle:home

cd "${PBS_O_WORKDIR}" || exit
if [[ ! -d ezpz ]]; then
  git clone https://github.com/saforem2/ezpz
else
  echo "Found ezpz!"
fi
source ezpz/src/ezpz/bin/savejobenv > /tmp/savejobenv.log 2>&1 || exit
source ezpz/src/ezpz/bin/getjobenv || exit

module load conda/2023-10-04; conda activate base
echo "WORLD_SIZE:${WORLD_SIZE}"
echo "NCCL: ${NCCL:-nccl}"

export PP=1
export TP=2

export HEADS=32
export NLAYERS=32
export HIDDEN=4096
export NUM_KV_HEAD=8

export ZERO_STAGE=2
export MICRO_BATCH=8
export GRAD_ACC_STEPS=1
export SEQ=4096
export DTYPE=fp16

export EVAL_ITERS=20
export TRAIN_ITER=317892
export SAVE_INTERVAL=5000
export EVAL_INTERVAL=50000

export DATA_PATH="/eagle/datasets/dolma/data_Llama2Tokenizer/wiki-en-simple/"
export DATA_FILE_LIST="/eagle/datasets/dolma/chunks/data_file_list_chunk_0_of_20.txt"

echo "++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "Using DATA_FILE_LIST: ${DATA_FILE_LIST}"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++"

export MODEL_TYPE="llama-seq${SEQ}-pp${PP}-tp${TP}-${NLAYERS}layers-${HEADS}heads-${HIDDEN}hidden"
echo "MODEL_TYPE: ${MODEL_TYPE}"

export LLAMA_ARGS="--no-query-key-layer-scaling --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear"
export USE_ACTIVATION_CHECKPOINTING=1 ; [ -n "${MODEL_TYPE}" ]

bash set_params.sh "$LLAMA_ARGS" --use-flash-attn-v2 --num-key-value-heads $NUM_KV_HEAD |& tee "train-${MODEL_TYPE}-mbs-${MICRO_BATCH}-zs${ZERO_STAGE}-kvh${NUM_KV_HEAD}-$(tstamp).log"
