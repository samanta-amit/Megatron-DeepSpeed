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

main() {
    #####################
    # MAIN PROGRAM LOGIC
    #####################
    #### 1. Navigate into `$PBS_O_WORKDIR`
    cd "${PBS_O_WORKDIR}" || exit
    #### 2. source `ALCF/helpers.sh`
    HERE=$(python3 -c 'import os; print(os.getcwd())') && export HERE
    source "${HERE}/ALCF/helpers.sh" || exit
    #### 3. call `setup` from `./ALCF/helpers.sh`
    setup || exit

    # Take custom args
    export custom_args=" $@"
    export run_cmd="${run_cmd} ${custom_args}"

    echo "${run_cmd}" | tee -a "${OUTPUT_LOG}"
    printf "[!! %s] View output at:\n %s\n" "$(printBlue "NOTE")" "$(printYellow "${OUTPUT_LOG}")" | tee -a "${OUTPUT_LOG}"
    eval "${run_cmd}" |& tee -a "${OUTPUT_LOG}"
    set +x
}

main
