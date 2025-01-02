# Fourier Controller Networks for Real-Time Decision-Making in Embodied Learning, ICML 2024

- 📝[Paper](https://arxiv.org/pdf/2405.19885)
- 🌍[Project Page](https://thkkk.github.io/fcnet)
- 🛢️[Dataset](https://ml.cs.tsinghua.edu.cn/~hengkai/unitree_general_expert_10000_191_r-100.zip)
- 👏Contributors: [Hengkai Tan](https://github.com/thkkk), [Kai Ma](https://github.com/mk2001233), [Songming Liu](https://github.com/csuastt)


## Installation

at least: 
- we recommend virtual env with `python=3.8`.
- `pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113`
- `pip install -e .` 

## Dataset

The 2M small version dataset is in [Download link](https://ml.cs.tsinghua.edu.cn/~hengkai/unitree_general_expert_10000_191_r-100.zip).
<!-- The part of dataset is in `FCNet/DT/data/data/unitree_general_expert_240000_255_r-100_partial/`.  -->

Our dataset is processed as mentioned in the FCNet paper, concatenating state and action together.

## Training example

The core code of FCNet is in `FCNet/DT/models/fourier_controller.py`.

Copy the data folder like `unitree_general_expert_10000_191_r-100` into the `FCNet/DT/data/data` directory so that the `FCNet/DT/data/data/unitree_general_expert_10000_191_r-100` directory contains three .dat files.

### DT
```bash
cd FCNet/DT/  # Need to run in the FCNet/DT/ directory
screen -X -S train quit
screen -dm -S train bash -c \
'OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=`python ./scripts/calc_avail_gpus.py -mg 8` train.py \
d_m=128 n_layer=4 n_head=8 batch_size=128 data_mode=nonmdp epochs=50 lr=0.005 seq_len=64 dt_mode=as_a model_name=transformer load_data_mode=episode2chunk max_used_memory=2000 \
tasks=[unitree_general] task_config_name=expert \
train=True \
save_test_best=True use_wandb=False use_tensorboard=False \
> train.log 2>&1'
```

And then you will see a `train.log` log file in `FCNet/DT/` path.

### FCNet

```bash
# training unitree aliengo
cd FCNet/DT/  # Need to run in the FCNet/DT/ directory
screen -X -S train quit
screen -dm -S train bash -c \
'OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=`python ./scripts/calc_avail_gpus.py -mg 8` train.py \
n_layer=4 batch_size=128 data_mode=nonmdp epochs=50 lr=0.005 seq_len=64 n_modes=10 ctx_dim=16 width=128 fno_hidden_size=256 final_hidden_size=128 dt_mode=as_a model_name=fourier_controller load_data_mode=episode2chunk \
tasks=[unitree_general] task_config_name=expert max_used_memory=2000 \
train=True \
save_test_best=True use_wandb=False use_tensorboard=True \
> train.log 2>&1'
```

Generally, the loss of FCNet will be much smaller than that of Decision Transformer, especially when the amount of data is small.

If you want to train on d4rl dataset, you can use the script under `FCNet/DT/d4rl`. First, use `download_d4rl.py` to download the d4rl data set to obtain the .pkl data set. Then run `convert_pkl_to_memmap.py` to convert it to our format, which will generate a dataset folder like `halfcheetah_expert_1000_1000` in the `FCNet/DT/data/data/` directory.

```bash
# training d4rl hopper-medium
screen -X -S d4rl_train quit
screen -dm -S d4rl_train bash -c \
'OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=`python ./scripts/calc_avail_gpus.py -mg 8` train.py \
d_m=128 n_layer=4 n_head=12 batch_size=16 data_mode=nonmdp epochs=50 lr=0.005 weight_decay=0.0001 seq_len=100 n_modes=10 ctx_dim=1 width=128 fno_hidden_size=512 final_hidden_size=128 dt_mode=as_a model_name=fourier_controller load_data_mode=episode2chunk \
tasks=[hopper] task_config_name=medium max_used_memory=2000 \
train=True \
save_test_best=True use_wandb=False use_tensorboard=False \
> train.log 2>&1'
```

After training, in `FCNet/FCNet/DT/log/unitree_general` path, you will see a timestamp string like `2024-01-20_22-56-27` which means model_id.

## Inference example

Replace the model_id `dt_policy_name=2024-01-20_22-56-27` in the following run command with the actual model you need to infer.

unitree aliengo: (NOTE: If you need to view the effect in the simulator during inference, please install `Isaac Gym Preview 3` (Preview 2 will not work!) from https://developer.nvidia.com/isaac-gym
and `legged_gym`.)

```bash
cd FCNet/DT/data/data_collectcd 
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

d4rl evaluation:
```
# medium-expert
screen -X -S play_d4rl quit
screen -dm -S play_d4rl bash -c \
'python d4rl_eval.py \
	--train_task hopper \
    --eval_task hopper-medium-replay-v2 \
    --dt_policy_name 2024-01-26_13-55-05 --kv_cache > play_d4rl.log 2>&1'
```

## Acknowledgement

- [Decision Transformer](https://github.com/kzl/decision-transformer)
- [nerfies](https://github.com/nerfies/nerfies.github.io)

## BibTeX
If you find our work useful for your project, please consider citing the following paper.

```
@inproceedings{tanfourier,
  title={Fourier Controller Networks for Real-Time Decision-Making in Embodied Learning},
  author={Tan, Hengkai and Liu, Songming and Ma, Kai and Ying, Chengyang and Zhang, Xingxing and Su, Hang and Zhu, Jun},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
