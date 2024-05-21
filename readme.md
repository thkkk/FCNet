## installation

at least: 
- `pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113`
- gym, d4rl
- isaacgym, legged_gym

- `pip install -e .` 

## training example

under `FCNet/DT/` path

### DT
```
screen -X -S train quit
screen -dm -S train bash -c \
'OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=`python ./scripts/calc_avail_gpus.py -mg 8` train.py \
d_m=128 n_layer=4 n_head=8 batch_size=128 data_mode=nonmdp epochs=50 lr=0.005 seq_len=64 dt_mode=as_a model_name=transformer load_data_mode=episode2chunk max_used_memory=2000 \
tasks=[unitree_general] task_config_name=expert \
train=True \
save_test_best=True use_wandb=False use_tensorboard=False \
> train.log 2>&1'
```

### FCNet

```
screen -X -S train quit
screen -dm -S train bash -c \
'OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=`python ./scripts/calc_avail_gpus.py -mg 8` train.py \
n_layer=4 batch_size=128 data_mode=nonmdp epochs=50 lr=0.005 seq_len=64 n_modes=10 ctx_dim=16 width=128 fno_hidden_size=256 final_hidden_size=128 dt_mode=as_a model_name=fourier_controller load_data_mode=episode2chunk \
tasks=[unitree_general] task_config_name=expert max_used_memory=2000 \
train=True \
save_test_best=True use_wandb=False use_tensorboard=True \
> train1.log 2>&1'
```



## inference example

under `FCNet/DT/data/data_collect` path

replace model_id

unitree aliengo:
```
screen -dm -S play bash -c \
'python play.py -m hydra/launcher=joblib \
    task=unitree_general \
    aligo_auto_complete_chks=False \
    dt_policy_name=2024-01-20_22-56-27 kv_cache=True \
    headless=True \
    record=False \
    max_episode_length=1500 \
    play_ep_cnt=3 \
    calc_dt_mlp_loss=False \
    num_envs=2048 \
    simplify_print_info=False \
    multi_gpu=True \
    dummy=False \
    resume=True \
    add_noise=True \
    push_robots=True > play.log 2>&1'
```
set `print_inference_action_time=True` for evaluating inference time.

Or `python d4rl_parallel_test_speed.py`  under `FCNet/DT/d4rl` path

d4rl:
```
# medium-expert
screen -X -S play_d4rl quit
screen -dm -S play_d4rl bash -c \
'python d4rl_eval.py \
	--train_task hopper \
    --eval_task hopper-medium-replay-v2 \
    --dt_policy_name 2024-01-26_13-55-05 --kv_cache > play_d4rl.log 2>&1'
```



## dataset

The part of dataset is in `FCNet/DT/data/data/unitree_general_expert_240000_255_r-100_partial/`. Running the above training program will load it automatically.


## Acknowledgement

- Decision Transformer
- 