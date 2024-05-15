#!/bin/bash --login
#
HOSTNAME=$(hostname)
if [[ "${HOSTNAME}" == x3* ]]; then
    MACHINE="polaris"
elif [[ "${HOSTNAME}" == x1* ]]; then
    MACHINE="sunspot"
elif [[ "${HOSTNAME}" == x4* ]]; then
    MACHINE="aurora"
fi

NOW="$(date "+%Y-%m-%d-%H%M%S")"
cd "${PBS_O_WORKDIR}" || exit
nhosts=$(wc -l < "${HOSTFILE}")

if [[ "${nhosts}" == 1 || "${nhosts}" == 2 ]]; then
    MBS=1
elif [[ "${nhosts}" -ge 2 ]]; then
    MBS=2
elif [[ "${nhosts}" -ge 8 ]]; then
    MBS=4
fi

printf "Detected %s  hosts. Running with micro_batch:\n" ${nhosts} ${MBS}

OUTDIR="${PBS_O_WORKDIR}/pbslogs"
mkdir -p "${OUTDIR}"
OUTFILE="${OUTDIR}/${PBS_JOBID}-${NOW}.log"
echo "Running on: ${MACHINE}"
echo "${OUTFILE}" >> "${OUTDIR}/latest"
echo "Logging job output to: ${OUTFILE}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=6000
# export DEBUG=1
MICRO_BATCH="${MBS}" DATA_FILE_LIST="${PBS_O_WORKDIR}/ALCF/data-lists/${MACHINE}/dolma_v1_7_file_list.txt" bash "${PBS_O_WORKDIR}/train_llama_alcf.sh" |& tee "${OUTFILE}"
