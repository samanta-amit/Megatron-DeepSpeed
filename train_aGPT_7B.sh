#!/bin/bash --login
#

NOW="$(date "+%Y-%m-%d-%H%M%S")"
cd "${PBS_O_WORKDIR}" || exit

HOSTNAME=$(hostname)
if [[ "${HOSTNAME}" == x3* ]]; then
    MACHINE="polaris"
    # XXX:
    # - On Polaris, we see that:
    #       - on 1 or 2 nodes, only MICRO_BATCH=1 will fit in memory
    #       - on 8 nodes, MICRO_BATCH=2 will fit in memory
    #       - on 48 nodes, MICRO_BATCH=4 will fit in memory
    nhosts=$(wc -l < "${PBS_NODEFILE}")
    if [[ "${nhosts}" == 1 ]]; then
        export MBS=1
    elif [[ "${nhosts}" == 2 ]]; then
        export MBS=1
    elif [[ "${nhosts}" -ge 3 ]]; then
        export MBS=2
    elif [[ "${nhosts}" -ge 8 ]]; then
        export MBS=4
    fi
elif [[ "${HOSTNAME}" == x1* ]]; then
    MACHINE="sunspot"
elif [[ "${HOSTNAME}" == x4* ]]; then
    MACHINE="aurora"
fi

export nhosts
OUTDIR="${PBS_O_WORKDIR}/pbslogs"
mkdir -p "${OUTDIR}"
OUTFILE="${OUTDIR}/${PBS_JOBID}-${NOW}.log"

echo "+---------------------------------------------------------+"
echo "| Running on: ${MACHINE}"
echo "| Detected ${nhosts} hosts. Running with micro batch: ${MBS}"
echo "| Logging job output to: ${OUTFILE}"
echo "+---------------------------------------------------------+"

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=6000
echo "${OUTFILE}" >> "${OUTDIR}/latest"
# export DEBUG=1
export MICRO_BATCH="${MICRO_BATCH:-${MBS}}"
export DATA_FILE_LIST="${DATA_FILE_LIST:-${PBS_O_WORKDIR}/ALCF/data-lists/${MACHINE}/dolma_v1_7_file_list.txt}"
bash "${PBS_O_WORKDIR}/train_llama_alcf.sh" |& tee "${OUTFILE}"
