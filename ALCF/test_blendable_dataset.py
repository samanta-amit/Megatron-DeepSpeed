#!/usr/bin/env python
from mpi4py import MPI
import os
from megatron.data.gpt_dataset import build_train_valid_test_datasets
import numpy as np
from megatron.global_vars import set_args, set_global_variables, get_args
from megatron.arguments import parse_args 
from megatron.initialize import initialize_megatron
from megatron.data.data_samplers import build_pretraining_data_loader
import time
from megatron.core import mpu
comm = MPI.COMM_WORLD
from megatron.utils import PerfTrace, Profile

dlp = Profile("TEST_BLENDABLEDATASET")
import datetime
def print_rank_0(msg):
    if comm.rank==0:
        print(f" [INFO][{datetime.datetime.now()}] {msg}", flush=True)
        

initialize_megatron(allow_no_cuda=True)
args = get_args()
if os.getenv('DLIO_PROFILER_DATASET_DIR') is not None:
    extra_trace_path = os.environ['DLIO_PROFILER_DATASET_DIR']
else:
    extra_trace_path=''

os.makedirs(args.trace_dir, exist_ok=True)
PerfTrace.initialize_log(f"{args.trace_dir}/trace-{comm.rank}-of-{comm.size}.pfw",  f"{args.data_cache_path}:{extra_trace_path}:{args.data_path}:{args.save}:{args.load}", process_id=comm.rank)

data_file_list = args.data_file_list
print_rank_0(f"Reading data from {args.data_file_list}")
files = []
weights = []
flist = []
with open(data_file_list, 'r') as fin:
    for f in fin.readlines():
        w, fname, c = f.split()
        weights.append(float(w))
        flist.append(fname)
        files.append(float(w))
        files.append(fname)
        files.append(c)
splits_string="100,0,0"

weights = np.array(weights)
weights = weights/np.sum(weights)

num_samples = args.global_batch_size*args.train_iters
num_datasets = len(weights)
print_rank_0(f"Number of datasets: {num_datasets}")
print_rank_0(f"Global batch size: {args.global_batch_size}")
print_rank_0(f"Training iterations: {args.train_iters}")
train_valid_test_num_samples = [num_samples, 0, 0]
seed=args.seed
data_impl = args.data_impl
skip_warmup = not args.mmap_warmup
seq_length = args.seq_length
splits_string = "1,0,0"

# Build datasets
start_build_dataset = time.time()
print_rank_0(f"Starting to build the blendable dataset")
train_ds, valid_ds, test_ds = build_train_valid_test_datasets(files, data_impl, splits_string,
                            train_valid_test_num_samples,
                            seq_length, seed, skip_warmup, data_cache_path=args.data_cache_path)


end_build_dataset = time.time()
print_rank_0(f"Finished building the blendable dataset in {end_build_dataset - start_build_dataset} second")
print_rank_0(f"Total number of samples: {len(train_ds)}")
print_rank_0(f"Weights set: {weights[:min(8, num_datasets)]}")

start_build_dataloader = time.time()
print_rank_0(f"Starting to build the data loader")
rank_in_parallel_group = mpu.get_sequence_parallel_rank()
train_dataloader = build_pretraining_data_loader(
    train_ds, args.consumed_train_samples)
valid_dataloader = build_pretraining_data_loader(
        valid_ds, args.consumed_valid_samples)
test_dataloader = build_pretraining_data_loader(test_ds, 0)
end_build_dataloader = time.time()
print_rank_0(f"Finished building the data loader in {end_build_dataloader - start_build_dataloader} second")

print_rank_0(f"Starting loading the data")
start_loading_time = time.time()
NUM_ITEMS=100
n=0
for i in dlp.iter(train_dataloader):
    print(f"[{comm.rank}] DATA {i}")
    n+=1
    if (n%NUM_ITEMS==0):
        print_rank_0(f"Loaded {n} batches in {time.time() - start_loading_time}")
    if n>=5000:
        break
end_loading_time = time.time()
print_rank_0(f"Finished loading the data ({n} batches) in {end_loading_time - start_loading_time}")



dataset_idx = [train_ds.dataset_index[i] for i in range(num_samples)]
ratio_select=np.zeros(num_datasets)

for e in range(min(100, args.train_iters)):
    ratio_select=np.zeros(num_datasets)
    for i in range(num_datasets):
        ratio_select[i] = np.sum([i==d for d in dataset_idx[e*args.global_batch_size:(e+1)*args.global_batch_size]])/args.global_batch_size
    print_rank_0(f"iter-{e}: {ratio_select[:min(8, num_datasets)]}")

