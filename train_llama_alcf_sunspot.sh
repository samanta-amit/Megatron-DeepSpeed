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

module () {
  if [ -z "${LMOD_SH_DBG_ON+x}" ]
  then
    case "$-" in
      (*v*x*) __lmod_sh_dbg='vx'  ;;
      (*v*) __lmod_sh_dbg='v'  ;;
      (*x*) __lmod_sh_dbg='x'  ;;
    esac
  fi
  if [ -n "${__lmod_sh_dbg:-}" ]
  then
    set +$__lmod_sh_dbg
    echo "Shell debugging temporarily silenced: export LMOD_SH_DBG_ON=1 for Lmod's output" >&2
  fi
  eval "$($LMOD_CMD $LMOD_SHELL_PRGM "$@")" && eval "$(${LMOD_SETTARG_CMD:-:} -s sh)"
  __lmod_my_status=$?
  if [ -n "${__lmod_sh_dbg:-}" ]
  then
    echo "Shell debugging restarted" >&2
    set -$__lmod_sh_dbg
  fi
  unset __lmod_sh_dbg
  return $__lmod_my_status
}

#
# eval "$(/home/foremans/miniconda3/bin/conda shell.zsh hook)"
# conda activate q4-drop



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ---- 0. Navigate into `$PBS_O_WORKDIR` -------------------------------------
cd "${PBS_O_WORKDIR}" || exit
HERE=$(python3 -c 'import os; print(os.getcwd())')
export HERE

# PARENT="$(dirname "${HERE}")"
# source "${PARENT}/setenv.sh" || exit
# ---- 1. Assert `./pretrain_gpt_alcf.py` exists: -----------------------------
export EXEC="${HERE}/pretrain_gpt_alcf.py"
[ -f "${EXEC}" ] || exit
# ---- 2. `source ./ALCF/helpers_alcf.sh`: ------------------------------------
sourceFile "${HERE}/ALCF/helpers.sh" || exit
# ---- 3. Call fns from `./ALCF/helpers_alcf.sh` ------------------------------
setEnv || exit                      # 1. load `conda` environment
saveDSenv || exit                   # 2. save env vars to `.deepspeed_env`
ezpz || exit                        # 3. determine WORLD_SIZE, etc. from `PBS_*` vars
makeHostfiles || exit               # 4. create `deepspeed` hostfile from `$PBS_NODEFILE`
setParams || exit                   # 5. set command line arguments to pass to `"${EXEC}"`
buildDSconfig || exit               # 6. create `deepspeed_config.json` from runtime params from ^
setOutput || exit                   # 7. specify output directory for {logs, checkpoints, etc.}
setArgs || exit                     # 8. specify additional `deepspeed` arguments
setData "${DATA_FILE_LIST}"|| exit  # 9. specify `DATA_FILE_LIST` for dolma dataset
setDSlauncher "${HERE}" || exit     # 10. set `launcher` args for `deepspeed ${launcher} ${EXEC} ${args}`
printJobInfo || exit                # 11. print job info
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Take custom args
custom_args=" $@"

# Assert `./hostfile_deepspeed` exists
export hfds="${HERE}/hostfile_deepspeed" && [ -f "${hfds}" ] || exit

# hf="${HOSTFILE:-${PBS_NODEFILE}}"
# nh=$(wc -l "${hf}")
# if [[ "${nh}" -gt 1 ]]; then
#     launch_cmd="deepspeed --hostfile $hfds --launcher MPICH ${EXEC}"
# else
#     launch_cmd="python3 ${EXEC}"
# fi
#
# echo "launch_cmd: ${launch_cmd}"

    # --use-flash-attn-v2 \
    # python3 ${EXEC} \
run_cmd="
    deepspeed --hostfile $hfds --launcher MPICH ${EXEC} \
    --$DTYPE \
    --num-workers 0 \
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
    --lr ${LR} \
    --seq-length $SEQ \
    --save ${CKPT_DIR} \
    --load ${CKPT_DIR} \
    --num-layers ${NLAYERS} \
    --hidden-size ${HIDDEN} \
    --train-iters ${TRAIN_ITER} \
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
    --data-cache-path ${DATA_CACHE_PATH} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    ${LLAMA_ARGS} \
    $ds_args \
    ${gpt_args[*]} \
    $custom_args \
    |& tee ${OUTPUT_LOG}
    "

    # ---------------------------------------------------
    # --vocab-file $VOCAB_FILE \
    # --merge-file $MERGE_FILE \
    # --lr-decay-iters 320000 \
    # --lr-warmup-iters 5000 \
    # --lr-decay-iters 10000 \
    # --num-workers 4 \
    # launch python3 ${EXEC} \
    # --data-impl mmap \
    # source ./ezpz/src/ezpz/bin/getjobenv || exit
    # ---------------------------------------------------
    # ${DIST_LAUNCH} ./local_rank.sh python3 ${EXEC} \
    # ${DIST_LAUNCH} python3 ${EXEC} \
    # deepspeed $launcher ${EXEC} \
    # >> ${OUTPUT_LOG} 2>&1 &
    # >> ${OUTPUT_LOG} 2>&1 &
    # |& tee $OUTPUT_DIR/output.log
    # ${EXTRA_ARGS} \

echo "All DeepSpeed(s): $(which -a deepspeed)"
echo "Using $(which deepspeed)"
ds_report

echo "${run_cmd}"

printf "[!! \e[1;31m%s\e[0m] View output at:\n" "NOTE"
printf "\e[1;34m%s\e[0m\n" "${OUTPUT_LOG}"
# echo "${OUTPUT_LOG}"
eval "${run_cmd}"
set +x
