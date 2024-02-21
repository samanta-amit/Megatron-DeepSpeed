#!/bin/bash --login
#PBS -l walltime=06:00:00
#PBS -A argonne_tpc
#PBS -q prod
#PBS -l select=48
#PBS -l filesystems=eagle:home

eval "$(/home/foremans/miniconda3/bin/conda shell.zsh hook)" && conda activate anl_release_q4v2
cd ~/anl_24_release_q4/llm.devkit || exit
export PBS_O_WORKDIR="${HOME}/anl_24_release_q4/llm.devkit"
export LLM_DK_DIR="${PBS_O_WORKDIR}"
# && export PBS_O_WORKDIR=$(pwd)
# && export LLM_DK_DIR="${PBS_O_WORKDIR}"
echo "Setting PBS_O_WORKDIR = LLM_DK_DIR = ${PBS_O_WORKDIR} = ${LLM_DK_DIR}"

ezpz() {
  if [[ ! -d ezpz ]]; then
    git clone https://github.com/saforem2/ezpz
  else
    echo "Found ezpz!"
  fi
  source ezpz/src/ezpz/bin/savejobenv > /tmp/savejobenv.log 2>&1 || exit
  source ezpz/src/ezpz/bin/getjobenv || exit
}


setEnv() {
  SETENV_FILE="${HOME}/anl_24_release_q4/llm.devkit/setenv.sh"
  if [[ "${SETENV_FILE}" ]]; then
    # shellcheck source=/home/foremans/anl_24_release_q4/llm.devkit/setenv.sh
    source "${HOME}/anl_24_release_q4/llm.devkit/setenv.sh"
  else
    echo "Unable to source ${SETENV_FILE}, exiting!"
    exit
  fi
}

makeHostfiles() {
  # ---- OLD --------------------------------
  # cd ~/anl_24_release_q4/llm.devkit/intel-extension-for-deepspeed/examples && bash make_hostfiles.sh
  # -----------------------------------------
  # ---- Make MPICH hostfile ----------------
  export hostfile_mpich=hostfile_mpich
  cat "$PBS_NODEFILE" > "${hostfile_mpich}"
  # ---- Make DeepSpeed hostfile -------------------
  export hostfile_deepspeed=hostfile_deepspeed
  cat "$PBS_NODEFILE" > "${hostfile_deepspeed}"
  sed -e 's/$/ slots=12/' -i "${hostfile_deepspeed}"
  # -------------------------------------------------
}

makeDSenv() {
    echo "PATH=${PATH}" > .deepspeed_env
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> .deepspeed_env
    echo "http_proxy=${http_proxy}" >> .deepspeed_env
    echo "https_proxy=${https_proxy}" >> .deepspeed_env
    echo "CFLAGS=${CFLAGS}" >> .deepspeed_env
    echo "PYTHONUSERBASE=$PYTHONUSERBASE" >> .deepspeed_env
}

cd ~/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed || exit
ezpz
setEnv
makeDSenv
makeHostfiles


# Disabling tensor/pipeline parallelism
export PP=${PP:-1}
export TP=${TP:-1}
# export HEADS=32
# export HIDDEN=4096
# export NLAYERS=${NLAYERS:-96}

# ---- Llama2 7B Config -----------------------
export HEADS=${HEADS:-32}
export NLAYERS=${NLAYERS:-32}
export HIDDEN=${HIDDEN:-4096}
export NUM_KV_HEAD=${NUM_KV_HEAD:-8}
export MODEL_TYPE="llama-seq${SEQ}-pp${PP}-tp${TP}-${NLAYERS}layers-${HEADS}heads-${HIDDEN}hidden"
# ---------------------------------------------

# ---- Run Settings ---------------------------
export ZERO_STAGE=${ZERO_STAGE:-2}
export MICRO_BATCH=${MICRO_BATCH:-4}
export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}
export SEQ=${SEQ:-4096}
export DTYPE=${DTYPE:-bf16}
# ---------------------------------------------

# export EVAL_ITERS=${EVAL_ITERS:-20}
# export EVAL_INTERVAL=${EVAL_INTERVAL:-50000}
export TRAIN_ITER=${TRAIN_ITER:-317892}
export SAVE_INTERVAL=${SAVE_INTERVAL:-200}

export LR=${LR:-0.0003}
export WORLD_SIZE=${WORLD_SIZE:-$(wc -l < "${PBS_NODEFILE}")}
export GLOBAL_BATCH=$(( $WORLD_SIZE * $MICRO_BATCH * $GRAD_ACC_STEPS / $TP / $PP ))
export USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING:-0}

# export DATA_FILE_LIST="/lus/eagle/projects/datasets/dolma/chunks/40/data_file_list_chunk_0_of_40.txt"
# export DATA_FILE_LIST="/lus/eagle/projects/datasets/dolma/chunks/10/data_file_list_chunk_0_of_10.txt"
# export DATA_FILE_LIST="/lus/eagle/projects/datasets/dolma/data_file_list_select_15.txt"
export DATA_FILE_LIST="/home/foremans/dolma-chunks/40/data_file_list_chunk_0_of_40.txt"

export TOKENIZER_MODEL="/lus/gecko/projects/Aurora_deployment/AuroraGPT/datasets/dolma/utils/tokenizer.model"


export LLAMA_ARGS="--no-query-key-layer-scaling --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear"
export USE_ACTIVATION_CHECKPOINTING=1 ;
export EXTRA_ARGS="--use-flash-attn --num-key-value-heads ${NUM_KV_HEAD}"

# bash $LLM_DK_DIR/intel-extension-for-deepspeed/examples/gpt.sh $@

export DS_CONFIG="ds_stage${ZERO_STAGE}_mb${MICRO_BATCH}_gb${GLOBAL_BATCH}_pp${PP}_${DTYPE}.json"

HERE=$(python3 -c 'import os; print(os.getcwd())')
bash "${HERE}/generate_config.sh" "${DS_CONFIG}" || exit
# bash "$"/generate_config.sh "${DS_CONFIG}" || exit 1
#
data_file_list_stem=$(echo "$DATA_FILE_LIST" | tr "\/" "\t" | awk '{print $NF}' | sed "s/\.txt//g")
DATA_CACHE_PATH="${HERE}/.cache/${data_file_list_stem}-index-cache"
mkdir -p "${DATA_CACHE_PATH}"
EXTRA_ARGS="${EXTRA_ARGS} --data-cache-path ${DATA_CACHE_PATH}"
# if [[ -n "$DATA_CACHE_PATH" ]]; then
#     echo "Using DATA_CACHE_PATH: ${DATA_CACHE_PATH}"
#     EXTRA_ARGS="${EXTRA_ARGS} --data-cache-path ${DATA_CACHE_PATH}"
# else
#     echo "Not using DATA_CACHE_PATH !!"
# fi


OUTPUT_PREFIX="logs/ds_stage${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_mb${MICRO_BATCH}_seq${SEQ}_gb${GLOBAL_BATCH}_pp${PP}_tp${TP}_${DTYPE}"
# OUTPUT_DIR=logs/ds_stage${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_mb${MICRO_BATCH}_seq${SEQ}_gb${GLOBAL_BATCH}_pp${PP}_tp${TP}_${DTYPE}_`date +%m%d%H%M%S`_${HOSTNAME}
OUTPUT_DIR="${OUTPUT_PREFIX}/$(date +%m%d%H%M%S)_${HOSTNAME}"
mkdir -p "${OUTPUT_DIR}"

export OUTPUT_PREFIX="${OUTPUT_PREFIX}"
export OUTPUT_DIR="${OUTPUT_DIR}"
echo "!!!Please see logs at ${OUTPUT_DIR}"

gpt_args=()
ds_args=" "
ds_args=" --deepspeed ${ds_args}"
if [ "$PP" == 1 ]; then
   ds_args=" --no-pipeline-parallel ${ds_args}" 
fi
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"

# we are now using activation checkpoint provided by megatron, see below.
if [[ "$USE_ACTIVATION_CHECKPOINTING" == 1 ]]; then
    echo "!! Caught USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING} !!"
    ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
    gpt_args+=(
        "--checkpoint-activations"
    )
    # "--checkpoint-num-layers 1"
    # --checkpoint-activations \
    # --deepspeed-activation-checkpointing
fi

# take custom args
custom_args=" $@"

# launcher setting
LAUNCHER=${LAUNCHER:-MPICH}
if [[ $LAUNCHER == "deepspeed" ]]; then
    launcher=""
else
    launcher="--force_multi --hostfile $hostfile_deepspeed --launcher=${LAUNCHER} --launcher_args='-hostfile ${hostfile_mpich}'"
fi

CCL=${CCL:-ccl}
# NCCL=${NCCL:-nccl}
EXEC=pretrain_gpt.py

# MODEL=LLAMA_7B
# OUTPUT_PREFIX=${MODEL}_z${ZERO_STAGE}_seqlen_tp${TP}_pp${PP}_sp${SP}_nl${NUM_LAYERS}_hs${HIDDEN_SIZE}_gb${BS}_mb${MBS}
echo "++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "- WORLD_SIZE:${WORLD_SIZE}"
echo "- CCL: ${CCL:-ccl}"
echo "- MODEL_TYPE: ${MODEL_TYPE}"
echo "- Using DATA_FILE_LIST: ${DATA_FILE_LIST}"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++"


# --ffn-hidden-size 11008 \
# --vocab-file $VOCAB_FILE \
# --merge-file $MERGE_FILE \
# --lr-decay-iters 320000 \
# --num-workers 0 \
# --eval-iters ${EVAL_ITERS} \
# --eval-interval ${EVAL_INTERVAL} \
    # --lr-warmup-iters 5000 \
    # --lr-decay-iters 10000 \
#
# --accumulate-allreduce-grads-in-fp32 \
# --data-impl mmap \
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
    --lr-decay-style cosine \
    --log-interval 1 \
    --save-interval ${SAVE_INTERVAL} \
    --split 90,5,5 \
    --$DTYPE \
    $ds_args \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-gradient-accumulation-fusion \
    --distributed-backend $CCL \
    --tokenizer-type Llama2Tokenizer \
    --save checkpoints/${OUTPUT_PREFIX} \
    --load checkpoints/${OUTPUT_PREFIX} \
    --use-checkpoint-opt_param-scheduler \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --data-file-list ${DATA_FILE_LIST} \
    ${LLAMA_ARGS} \
    ${EXTRA_ARGS} \
    ${gpt_args[*]} \
    $custom_args \
    |& tee $OUTPUT_DIR/output.log
    "

run_cmd_intel="
    deepspeed $launcher pretrain_gpt.py \
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
    --lr 0.00015 \
    --lr-warmup-fraction .01 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-file-list ${DATA_FILE_LIST} \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --save-interval ${SAVE_INTERVAL} \
    --split 90,5,5 \
    --$DTYPE \
    --checkpoint-activations \
    --deepspeed-activation-checkpointing
    $ds_args \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-gradient-accumulation-fusion \
    --distributed-backend $CCL \
    --num-workers 0 \
    $custom_args \
    |& tee $OUTPUT_DIR/output.log
    "


# echo "Using $(which deepspeed)"
ds_report

echo "${run_cmd}"
eval "${run_cmd}"
set +x
