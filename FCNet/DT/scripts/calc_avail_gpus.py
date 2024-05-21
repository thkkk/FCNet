from FCNet.DT.utils.common import get_availble_gpus
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--max_gpu_num', '-mg', type=int, default=10)
parse.add_argument('--max_used_memory', '-mm', type=int, default=1000)
args = parse.parse_args()

availble_gpus = get_availble_gpus(max_used_memory=args.max_used_memory)
if args.max_gpu_num > 0:
    availble_gpus = availble_gpus[:args.max_gpu_num]
print(len(availble_gpus), end='')