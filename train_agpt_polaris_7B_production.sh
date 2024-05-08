#!/bin/bash --login

NOW="$(date "+%Y-%m-%d-%H%M%S")"
cd "${PBS_O_WORKDIR}" || exit

OUTDIR="${PBS_O_WORKDIR}/pbslogs"
mkdir -p "${OUTDIR}"
OUTFILE="${OUTDIR}/${PBS_JOBID}-${NOW}.log"
echo "${OUTFILE}" >> "${OUTDIR}/latest"
echo "Logging job output to: ${OUTFILE}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=6000
# export DEBUG=1
MICRO_BATCH=2 DATA_FILE_LIST="${PBS_O_WORKDIR}/ALCF/data-lists/polaris/dolma_v1_7_file_list.txt" bash "${PBS_O_WORKDIR}/train_llama_alcf.sh" |& tee "${OUTFILE}"
