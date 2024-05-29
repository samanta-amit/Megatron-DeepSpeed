#!/bin/bash --login
#PBS -l walltime=06:00:00
#PBS -A argonne_tpc
#PBS -q prod
#PBS -l select=48
#PBS -l filesystems=eagle:home

#############################################################################
# Check if running in `DEBUG=1` mode.
#   - If so, this will print each command before it is ran and exit if any of
#   them return a nonzero exit status.
#############################################################################
if [[ -n "${DEBUG-}" ]]; then  # to use: `DEBUG=1 bash train_llama_alcf.sh`
    printf "\e[1;31m%s\e[0m\n" "!! RUNNING IN DEBUG MODE !!"
    set -euxo pipefail
fi

if [[ -v NOOP ]]; then         # to use: `NOOP=1 bash train_llama_alcf.sh`
  echo "Run NOOP mode"
  set -o noexec                # same as set -n
fi

##################################################
# Helper function for `source`-ing another file
##################################################
sourceFile() {
    fp="$1"
    echo "source-ing ${fp}"
    if [[ -f "${fp}" ]]; then
        source "${fp}"
    else
        echo "ERROR: UNABLE TO SOURCE ${fp}"
    fi
}

#####################
# MAIN PROGRAM LOGIC
#####################
# ----[1. Navigate into `$PBS_O_WORKDIR`]--------------------------------------
cd "${PBS_O_WORKDIR}" || exit
HERE=$(python3 -c 'import os; print(os.getcwd())') && export HERE
sourceFile "${HERE}/ALCF/helpers.sh" || exit
setup || exit
###############################################################################

# Take custom args
export custom_args=" $@"
export run_cmd="${run_cmd} ${custom_args}"

echo "${run_cmd}" | tee -a "${OUTPUT_LOG}"
printf "[!! %s] View output at:\n %s\n" "$(printBlue "NOTE")" "$(printYellow "${OUTPUT_LOG}")" | tee -a "${OUTPUT_LOG}"
# eval "${run_cmd}" >> "${OUTPUT_LOG}" 2>&1  &
eval "${run_cmd}" |& tee -a "${OUTPUT_LOG}"
set +x
