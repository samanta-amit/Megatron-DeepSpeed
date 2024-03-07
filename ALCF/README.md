# Megatron-DeepSpeed @ ALCF

## Polaris

### Install

1. Clone [`argonne-lcf/Megatron-DeepSpeed`](https://github.com/argonne-lcf/Megatron-DeepSpeed)

    ```bash
    [#](#.md) ---- 0. Clone + navigate into `Megatron-DeepSpeed`:
    $ git clone https://github.com/argonne-lcf/Megatron-DeepSpeed
    $ cd Megatron-DeepSpeed
    ```

2. Create `conda` env:

    ```bash
    $ module load conda/2023-10-04 #; conda activate /lus/eagle/projects/datascience/foremans/miniconda3/envs/polaris/2024-03-06
    $ MPICC="cc -shared -taret-accel=nvidia80"
    $ DAY=$(date "+%Y-^m-%d")
    $ conda create --solver libmamba -c pytorch -c nvidia --name "${DAY}" "python==3.10"
    $ export PYTHONUSERBASE="${HOME}/.local/polaris/conda/${DAY}"
    ```

3. Install dependencies:

    ```bash
    $ conda install -c pytorch -c nvidia --solver libmamba mpi4py pytorch-cuda=11.8 ninja torchvision torchaudio pytorch-cuda=11.8 transformers xformers triton
    $ python3 -m pip install --upgrade pip pybind11 toolong appdirs wandb sentencepiece ipython setuptools wheel ninja
    $ python3 -m pip install --upgrade deepspeed wandb
    ```

    1. Install `apex`:

        ```bash
        $ git clone https://github.com/NVIDIA/apex
        $ cd apex
        $ module swap gcc gcc/10.3.0
        $ python3 -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
        ```

    2. Install `ezpz`:

        ```bash
        $ git clone https://github.com/saforem2/ezpz
        $ python3 -m pip install -e "ezpz[dev]"
        ```

### Running

- The (shell) script used to launch pre-training is:
    - Polaris:
      [`train_llama_alcf_polaris.sh`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/train_llama_alcf_polaris.sh)
    - Aurora:
      [`train_llama_alcf_aurora.sh`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/train_llama_alcf_aurora.sh)

- These shell script(s) will set the appropriate environment variables, load the correct conda
modules and launch
[`pretrain_gpt_alcf.py`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/pretrain_gpt_alcf.py) using `deepspeed`


- Explicitly, to launch:

    ```bash
    # 1. Launch interactive job
    $ qsub -A <your-project> -q debug -l select=2 -l walltime=01:00:00,filesystems=eagle:home -I
    # 2. Load conda environment
    $ module load conda/2023-10-04 ; conda activate /lus/eagle/projects/datascience/foremans/miniconda3/envs/polaris/2024-03-06
    $ export PYTHONUSERBASE=/home/foremans/.local/polaris/conda/2024-03-06
    # 3. Navigate into `Megatron-DeepSpeed` directory
    $ cd Megatron-DeepSpeed
    # 4. Launch:
    $ bash train_llama_alcf_polaris.sh
    ```

<details closed><summary><b>[Output]</b></summary>

```bash
source-ing /lus/eagle/projects/datascience/foremans/tmp/Megatron-DeepSpeed/ALCF/helpers_alcf.sh

CommandNotFoundError: Your shell has not been properly configured to use 'conda deactivate'.
To initialize your shell, run

    $ conda init <SHELL_NAME>

Currently supported shells are:
  - bash
  - fish
  - tcsh
  - xonsh
  - zsh
  - powershell

See 'conda init --help' for more information and options.

IMPORTANT: You may need to close and restart your shell after running 'conda init'.


Saving {PATH, LD_LIBRARY_PATH, htt{p,ps}_proxy, CFLAGS, PYTHONUSERBASE} to .deepspeed_env
Found ezpz!
/lus/eagle/projects/datascience/foremans/tmp/Megatron-DeepSpeed/ezpz/src/ezpz/__init__.py
Has ezpz installed. Nothing to do.
┌──────────────────────────────────────────────────────────────────
│ [Hosts]:
│     • [host:0] - x3005c0s37b0n0.hsn.cm.polaris.alcf.anl.gov
│     • [host:1] - x3005c0s37b1n0.hsn.cm.polaris.alcf.anl.gov
└──────────────────────────────────────────────────────────────────
┌──────────────────────────────────────────────────────────────────
│ [DIST INFO]:
│     • Loading job env from: /home/foremans/.pbsenv
│     • HOSTFILE: /var/spool/pbs/aux/1777928.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
│     • NHOSTS: 2
│     • NGPU_PER_HOST: 4
│     • NGPUS (NHOSTS x NGPU_PER_HOST): 8
│     • WORLD_SIZE: 8
│     • DIST_LAUNCH: mpiexec --verbose --envall -n 8 -ppn 4 --hostfile /var/spool/pbs/aux/1777928.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
└──────────────────────────────────────────────────────────────────
┌──────────────────────────────────────────────────────────────────
│ [Launch]:
│     • Use: 'launch' (=mpiexec --verbose --envall -n 8 -ppn 4 --hostfile /var/spool/pbs/aux/1777928.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov)
│       to launch job
└──────────────────────────────────────────────────────────────────

# [...]
```

</details>
