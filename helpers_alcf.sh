#!/bin/bash --login

ezpz() {
    if [[ ! -d ezpz ]]; then
        git clone https://github.com/saforem2/ezpz
    else
        echo "Found ezpz!"
    fi
    if python3 -c 'import ezpz; print(ezpz.__file__)' 2> '/dev/null'; then
        echo "Has ezpz installed. Nothing to do."
    else
        echo "Does not have ezpz installed. Installing..."
        echo "Using $(which python3) to install \`ezpz\`:"
        python3 -m pip install -e ezpz > ezpz-install.log 2>&1
    fi
    source ezpz/src/ezpz/bin/savejobenv > /tmp/savejobenv.log 2>&1 || exit
    source ezpz/src/ezpz/bin/getjobenv || exit
}

saveDSenv() {
    echo "Saving {PATH, LD_LIBRARY_PATH, htt{p,ps}_proxy, CFLAGS, PYTHONUSERBASE} to .deepspeed_env"
    {
        echo "PATH=${PATH}" ;
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" ;
        echo "http_proxy=${http_proxy}" ;
        echo "https_proxy=${https_proxy}" ;
        echo "CFLAGS=${CFLAGS}" ;
        echo "PYTHONUSERBASE=$PYTHONUSERBASE" ;
    } > .deepspeed_env
}


sumWeights() {
    local file_list=$1
    weights=$(cat "${file_list}" | awk '{print $1}' | tr '\n' '\ ,\ ' | sed 's/^/[/g' | sed 's/$/]/g' | tr '\ ' "\,\ ")
    # weights=$(echo "$weights" | tr ",]" "]")
    # echo "weights: $weights"
    python3 -c "import numpy as np; print(np.sum(${weights}))"
}

sumFiles() {
    local rd=$1
    for f in $("${rd}/*.txt"); do
        ws=$(sumWeights "${rd}/${f}")
        echo "sum($f.weights)=${ws}"
    done
}

setupData() {
    cidx=$1
    echo "Caught DOLMA_CHUNK_IDX: ${cidx} !!"
    dfl="./chunks-reweighted/10/data_file_list_chunk_${cidx}_of_10.txt"
    if [[ -z "${DATA_FILE_LIST}" ]]; then
        DATA_FILE_LIST="${dfl}"
    else
        echo "Caught DATA_FILE_LIST: ${DATA_FILE_LIST} from ENV!!"
    fi
    NDOCS=$(wc -l < "${DATA_FILE_LIST}") && export NDOCS="${NDOCS}"
    WEIGHT_SUM="$(sumWeights "${DATA_FILE_LIST}")"
    export WEIGHT_SUM="${WEIGHT_SUM}"
    export NDOCS="${NDOCS}"
    echo "Using DATA_FILE_LIST: ${DATA_FILE_LIST} with ${NDOCS} documents"
    echo "WEIGHT SUM: ${WEIGHT_SUM}"
    data_file_list_stem=$(echo "$DATA_FILE_LIST" | tr "\/" "\t" | awk '{print $NF}' | sed "s/\.txt//g")
    export DOLMA_CHUNK_IDX="${cidx}"
    export DATA_FILE_LIST_STEM="${data_file_list_stem}"
    export DATA_CACHE_PATH=".cache/${data_file_list_stem}/index-cache"
    mkdir -p "${DATA_CACHE_PATH}"
}



setEnv() {
    if [[ $(hostname) == x4* ]]; then
        SETENV_FILE="${HOME}/anl_24_release_q4/llm.devkit/setenv.sh"
        if [[ "${SETENV_FILE}" ]]; then
            # shellcheck source=/home/foremans/anl_24_release_q4/llm.devkit/setenv.sh
            source "${HOME}/anl_24_release_q4/llm.devkit/setenv.sh"
        else
            echo "Unable to source ${SETENV_FILE}, exiting!"
            exit
        fi
    elif [[ $(hostname) == x3* ]]; then
        # ---- load conda -----------------------------------
        module load conda/2023-10-04; conda activate base
        if [[ "${VIRTUAL_ENV}" ]]; then
            echo "Caught VIRTUAL_ENV = ${VIRTUAL_ENV} from environment!!"
        else
            echo "Not using VIRTUAL_ENV"
            # sourceFile "${HERE}/venvs/polaris/2023-10-04/bin/activate" || exit
        fi
    else
        echo "Unknown hostname $(hostname)"
        exit 1
    fi
}

makeHostfiles() {
    GPUS_PER_NODE=$(python3 -Wignore -c 'import ezpz; print(ezpz.get_gpus_per_node())')
    export GPUS_PER_NODE="${GPUS_PER_NODE}"
    # ---- Make MPICH hostfile ----------------
    export hostfile_mpich=hostfile_mpich
    cat "$PBS_NODEFILE" > "${hostfile_mpich}"
    # ---- Make DeepSpeed hostfile -------------------
    export hostfile_deepspeed=hostfile_deepspeed
    cat "$PBS_NODEFILE" > "${hostfile_deepspeed}"
    sed -e "s/$/ slots=${GPUS_PER_NODE}/" -i "${hostfile_deepspeed}"
}


makeDSenv() {
    echo "PATH=${PATH}" > .deepspeed_env
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> .deepspeed_env
    echo "http_proxy=${http_proxy}" >> .deepspeed_env
    echo "https_proxy=${https_proxy}" >> .deepspeed_env
    echo "CFLAGS=${CFLAGS}" >> .deepspeed_env
    echo "PYTHONUSERBASE=$PYTHONUSERBASE" >> .deepspeed_env
}
