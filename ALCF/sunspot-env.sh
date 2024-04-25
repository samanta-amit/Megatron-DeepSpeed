#!/bin/bash --login

export CCL_OP_SYNC=1                # Required by current oneCCL (HPCS-8067)
export CCL_PROCESS_LAUNCHER=pmix    # Required by Aurora mpich
export FI_PROVIDER=cxi              # Required by Aurora mpich
export PALS_PMI=pmix                # Required by Aurora mpich
export CCL_ATL_TRANSPORT=mpi        # Required by Aurora mpich
export FI_MR_CACHE_MONITOR=disabled # Required by Aurora mpich (HPCS-6501)
export CCL_SKIP_SCHEDULER=1         # Required by current oneCCL, will remove when set by default
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export CCL_TOPO_COLOR="card:{0,1},{2,3},{4,5},{6,7},{8,9},{10,11};plane:{0,3,4,6,8,11},{1,2,5,7,9,10}"
export UR_L0_IN_ORDER_BARRIER_BY_SIGNAL=0 # Required by current oneCCL


export LLM_DK_DIR=/home/$(whoami)/q4-drop_sunspot/llm.devkit

module load oneapi/release/2023.12.15.001
unset MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE
unset MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE
unset MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE


module use /home/ftartagl/graphics-compute-runtime/modulefiles
module load graphics-compute-runtime/agama-ci-devel-736.9
source /home/$(whoami)/q4-drop_sunspot/llm.devkit/torch-ccl/third_party/oneCCL/build/_install/env/vars.sh
module load gcc/12.1.0
module unload intel_compute_runtime/release/agama-devel-647
