# Megatron-DeepSpeed @ ALCF

## Polaris

<details closed><summary>⬜ <code>TODOs</code></summary>

- Convergence:
    - [ ] Use `bf16` on both systems
    - [ ] Will need to track (for each layer):
        - [ ] inputs / outputs
        - [ ] weights, gradients
    - [ ] Start thread in Intel SC23 channel to discuss convergence issues
        - [ ] Add hooks to track additional data

- [ ] Ensure / double check that optimizer settings from `ds_config.json` aren't being overwritten by some defaults in `megatron/arguments.py`
    - [ ] specifically, `momentum, beta{1, 2}, etc`
    
<details closed><summary><b>✅ <code>Completed</code></b></summary>

- Continue runs on Polaris @
    - [x] 48 Nodes
    - [x] 32 Nodes
    - [x] 16 Nodes
    - [x] 8 Nodes
    - [x] 4 Nodes

- [x] Then, try re-creating ( / fixing) conda with `cuda==12.1`
    - 😔, failed.
     
- ~~‼️  Unable to save checkpoints with `torch==2.1` + `cuda==11.8`~~:
    - Fixed in [a57a21f](https://github.com/argonne-lcf/Megatron-DeepSpeed/commit/a57a21f6b2a8abf847f5ef599e1b1edcb5a5e1b5)

    <details closed><summary><code>🐛 Bug</code></summary>
        
    - Training progresses OK:

        ```bash
        [2024-03-07 15:27:02,646] [INFO] [timer.py:260:stop] epoch=0/micro_step=199/global_step=199, RunningAvgSamplesPerSec=58.730622229657506, CurrSamplesPerSec=61.35304005128382, MemAllocated=6.01GB, MaxMemAllocated=19.52GB
        iteration      199/  317892 | consumed samples:       152832 | consumed tokens:    625999872 | elapsed time per iteration (ms): 14287.5 | learning rate: 2.407E-04 | global batch size:   768 | lm loss: 5.905366E+00 | loss scale: 8192.0 | actual seqlen:  4096 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 53.753 | tokens per gpu per second (tgs): 1146.733 | TFLOPs: 69.85 |
        [2024-03-07 15:27:15,063] [INFO] [logging.py:96:log_dist] [Rank 0] step=200, skipped=4, lr=[0.000240653265864008, 0.000240653265864008], mom=[(0.9, 0.999), (0.9, 0.999)]
        [2024-03-07 15:27:17,188] [INFO] [timer.py:260:stop] epoch=0/micro_step=200/global_step=200, RunningAvgSamplesPerSec=58.730745476291396, CurrSamplesPerSec=58.75503515561452, MemAllocated=6.01GB, MaxMemAllocated=19.52GB
        iteration      200/  317892 | consumed samples:       153600 | consumed tokens:    629145600 | elapsed time per iteration (ms): 14541.4 | learning rate: 2.407E-04 | global batch size:   768 | lm loss: 5.897035E+00 | loss scale: 8192.0 | actual seqlen:  4096 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 52.815 | tokens per gpu per second (tgs): 1126.713 | TFLOPs: 68.63 |
        saving checkpoint at iteration     200 to checkpoints/ds_stage2_nl32_hs4096_mb8_seq4096_gb768_pp1_tp2_fp16
        # ...
        ```

    - Then crashes with:

      ```python
      Traceback (most recent call last):
      Traceback (most recent call last):
        File "/lus/eagle/projects/datascience/foremans/tmp/Megatron-DeepSpeed/pretrain_gpt_alcf.py", line 575, in <module>
          model = main()
        File "/lus/eagle/projects/datascience/foremans/tmp/Megatron-DeepSpeed/pretrain_gpt_alcf.py", line 554, in main
          model = pretrain(
        File "/lus/eagle/projects/datascience/foremans/tmp/Megatron-DeepSpeed/megatron/training.py", line 226, in pretrain
          iteration = train(forward_step_func,
        File "/lus/eagle/projects/datascience/foremans/tmp/Megatron-DeepSpeed/megatron/training.py", line 1290, in train
          save_checkpoint_and_time(iteration, model, optimizer,
        File "/lus/eagle/projects/datascience/foremans/tmp/Megatron-DeepSpeed/megatron/training.py", line 1151, in save_checkpoint_and_time
          save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
        File "/lus/eagle/projects/datascience/foremans/tmp/Megatron-DeepSpeed/megatron/checkpointing.py", line 259, in save_checkpoint
          state_dict[UNIVERSAL_CHECKPOINT_INFO] = _universal_checkpoint_info(model)
        File "/lus/eagle/projects/datascience/foremans/tmp/Megatron-DeepSpeed/megatron/checkpointing.py", line 783, in _universal_checkpoint_info
          info.update(model[0].universal_checkpoint_info())
        File "/lus/eagle/projects/datascience/foremans/tmp/Megatron-DeepSpeed/megatron/model/gpt_model.py", line 203, in universal_checkpoint_info
          info[TP_REPLICATED_PARAMETER_PATTERNS] = self._get_tp_replicated_param_patterns()
        File "/lus/eagle/projects/datascience/foremans/miniconda3/envs/polaris/2024-03-06/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1695, in __getattr__
          raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
      AttributeError: 'GPTModel' object has no attribute '_get_tp_replicated_param_patterns'
      ```

      🤔
</details>

</details>

</details>

</details>

### Install

1. Clone [`argonne-lcf/Megatron-DeepSpeed`](https://github.com/argonne-lcf/Megatron-DeepSpeed)

    ```bash
    $ git clone https://github.com/argonne-lcf/Megatron-DeepSpeed
    $ cd Megatron-DeepSpeed
    ```

2. Create `conda` env:

    ```bash
    $ module load conda/2023-10-04
    $ export MPICC="cc -shared -taret-accel=nvidia80"
    $ export DAY=$(date "+%Y-%m-%d")
    $ export PYTHONUSERBASE="${HOME}/.local/polaris/conda/${DAY}"
    $ conda create --solver libmamba -c pytorch -c nvidia --name "${DAY}" "python==3.10"
    ```

    > [!NOTE]
    > In the `conda create` command above,
    > you can replace `--name "${DAY}"` with
    > `--prefix /path/to/your/conda/envs`, if you prefer:

3. Install dependencies:

    ```bash
    $ conda activate "${DAY}"  # e.g. 2024-03-07
    $ conda install -c pytorch -c nvidia --solver libmamba mpi4py ninja transformers xformers triton pytorch torchvision torchaudio pytorch-cuda=11.8
    $ python3 -m pip install --upgrade pip pybind11 toolong appdirs wandb sentencepiece ipython setuptools wheel ninja
    $ python3 -m pip install --upgrade deepspeed wandb
    ```

    - [`NVIDIA/apex`](https://github.com/NVIDIA/apex):

        ```bash
        $ git clone https://github.com/NVIDIA/apex
        $ cd apex
        # NOTE: need GCC < 11 for APEX ¯\_(ツ)_/¯ ??
        $ module swap gcc gcc/10.3.0
        $ python3 -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
        ```

    - [`ezpz`](https://github.com/saforem2/ezpz):

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
    # $ export PYTHONUSERBASE=/home/foremans/.local/polaris/conda/2024-03-06
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
