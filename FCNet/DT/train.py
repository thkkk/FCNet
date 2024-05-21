class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

import sys
sys.stdout = Unbuffered(sys.stdout)
sys.path.append('..')

import os
os.environ['HYDRA_FULL_ERROR'] = '1'
from os.path import exists, join

import yaml
import pprint
import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf
from lion_pytorch import Lion
from transformers import get_cosine_schedule_with_warmup

from data.load_data import get_dataloaders
from utils.runner import Runner
from utils.hydra_utils import *
from utils import fourier_utils
from utils.common import omegaconf_to_dict, get_availble_gpus, \
    get_parameter_number, sync_mgpu
from utils.type_utils import load2input_func

import numpy as np

import torch
# torch.set_printoptions(
#     precision=2,
#     # threshold=torch.inf,
#     edgeitems=3,
#     linewidth=300_000,
#     profile='full',
#     sci_mode=False,)
from torch.cuda.amp import autocast, GradScaler


@hydra.main(version_base=None, config_path="./cfg", config_name="config")
def main(cfg: DictConfig):
    # yaml config to dict
    cfg_dict = omegaconf_to_dict(cfg)
    
    # parse task configs
    task_config_name = cfg_dict['task_config_name']
    if task_config_name is None:
        raise ValueError('please specify task_config_name in python args!')
    task_config_dir = './cfg/train_task_config'
    with open(join(task_config_dir, f'{task_config_name}.yaml')) as f:
        task_config = yaml.safe_load(f)
    
    # set test training mode: ban wandb, tensorboard and 
    # saving best test metrics model. Scaling the data 
    # to smaller.
    test_train_mode = cfg_dict['test_train_mode']
    if test_train_mode:
        cfg_dict['use_wandb'] = False
        cfg_dict['use_tensorboard'] = False
        cfg_dict['save_test_best'] = False
        cfg_dict['data_scale'] = True

    # yaml 参数
    d_m = cfg_dict['d_m']
    n_layer = cfg_dict['n_layer']
    n_head = cfg_dict['n_head']
    ffn_coef = cfg_dict['ffn_coef']
    epochs = cfg_dict['epochs']
    lr = cfg_dict['lr']
    warmup_ratio = cfg_dict['warmup_ratio']
    weight_decay = cfg_dict['weight_decay']
    optimizer_use_triton = cfg_dict['optimizer_use_triton']
    train_ratio = cfg_dict['train_ratio']
    batch_size = cfg_dict['batch_size']
    data_read_from = cfg_dict['data_read_from']
    load_data_mode = cfg_dict['load_data_mode'] # chunk, episode, episode2chunk
    add_last_action = cfg_dict['add_last_action'] # valid only when load_data_mode=episode, add last action to the input
    tasks = cfg_dict['tasks']
    distributed = cfg_dict['distributed']
    data_scale = cfg_dict['data_scale']
    max_sample_number = cfg_dict['max_sample_number']
    if max_sample_number is None:
        max_sample_number = np.inf
    load_data_statistics = cfg_dict['load_data_statistics']
    data_mode = cfg_dict['data_mode_checked']
    model_name = cfg_dict['model_name_checked']
    # for fourier controller
    fno_hidden_size = cfg_dict['fno_hidden_size']
    final_hidden_size = cfg_dict['final_hidden_size']
    width = cfg_dict['width']
    ctx_dim = cfg_dict['ctx_dim']
    n_modes = cfg_dict['n_modes']
    is_chunk_wise = cfg_dict['is_chunk_wise']
    
    if data_mode == 'mdp':
        cfg_dict['seq_len'] = 'mdp'
    seq_len = cfg_dict['seq_len']
    episode_first_no_backward = cfg_dict['episode_first_no_backward'] # episode train first max episode length no backward

    dt_mode = cfg_dict['dt_mode_name']
    aligo_enable = cfg_dict['aligo_enable']

    use_fp16 = cfg_dict['use_fp16']
    use_flash_attn = cfg_dict['use_flash_attn']
    is_causal = cfg_dict['is_causal']

    aligo_name = cfg_dict['aligo_name']
    aligo_data_root_dir = cfg_dict['aligo_data_root_dir']

    assert tasks is not None and isinstance(tasks, omegaconf.listconfig.ListConfig), \
        f'{tasks}, {type(tasks)}'

    if not distributed or int(os.environ['LOCAL_RANK']) == 0:
        pp = pprint.PrettyPrinter(indent=4)
        print("cfg_dict: ")
        pp.pprint(cfg_dict)

    # dtype = torch.float16
    dtype = torch.float32

    availble_gpus = get_availble_gpus(max_used_memory=cfg_dict['max_used_memory'])
    print(f"availble_gpus: {availble_gpus}")
    assert len(availble_gpus) > 0
    local_rank = 0
    world_size = 1
    local_device_id = availble_gpus[local_rank]

    if distributed:
        print('distributed: {}'.format(distributed))
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_device_id = availble_gpus[local_rank]

        print(f'local_rank: {local_rank}, gpu_device_id: {local_device_id}, \
world_size: {world_size}, local_device_id: {local_device_id}')
        torch.cuda.set_device(local_device_id)
        # torch.distributed.init_process_group(backend='nccl', init_method='env://')
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23450',
                                             rank=local_rank, world_size=world_size)
        print('distribute inited!')
    # world_size = torch.distributed.get_world_size()
    assert len(availble_gpus) >= world_size, \
        f"availble_gpus: {availble_gpus}, world_size: {world_size}"
    device = torch.device('cuda', local_device_id)
    # print("task_config", task_config)
    train_loader, test_loader, src_dim, tgt_dim, dataset_info, srcs, tgts, src_episodes_list, tgt_episodes_list = \
        get_dataloaders(
            data_read_from=data_read_from,
            load_data_mode=load_data_mode,
            tasks=tasks,
            seq_len=seq_len, 
            dt_mode=dt_mode, 
            aligo_name=aligo_name, 
            aligo_data_root_dir=aligo_data_root_dir,
            local_rank=local_rank, 
            world_size=world_size, 
            train_ratio=train_ratio, 
            dtype=dtype, 
            aligo_enable=aligo_enable, 
            batch_size=batch_size,
            data_scale=data_scale, 
            distributed=distributed, 
            data_mode=data_mode, 
            max_sample_number=max_sample_number,
            load_data_statistics=load_data_statistics,
            task_config=task_config, 
            device=device
        )
    
    # convert load_data_mode to input_data_mode
    input_data_mode = load2input_func(load_data_mode) # chunk, episode
    
    cfg_dict['src_dim'] = src_dim; cfg_dict['tgt_dim'] = tgt_dim
    train_loader_len, test_loader_len = len(train_loader), len(test_loader)
    print(f'src_dim: {src_dim}, tgt_dim: {tgt_dim}')
    print(f'train_loader_len: {train_loader_len}, test_loader_len: {test_loader_len}')

    model_config = dict(
        src_dim=src_dim,
        tgt_dim=tgt_dim,
        hidden_size=d_m,
        ffn_coef=ffn_coef,
        n_layer=n_layer, 
        n_head=n_head, 
        dt_mode=dt_mode,
        seq_len=seq_len, 
        data_mode=data_mode,
        input_data_mode=input_data_mode, # chunk, episode
        # max_episode_length=max_episode_length,
        tasks=list(tasks),
        use_fp16=use_fp16,
        export_model_as_jit=cfg_dict['export_model_as_jit'],
        use_flash_attn=use_flash_attn, 
        is_causal=is_causal,
        dataset_info=dataset_info, 
        model_name=model_name, 
        double_v_dim=cfg_dict['double_v_dim'],
        add_last_action=add_last_action,
        episode_first_no_backward=episode_first_no_backward,
        ctx_dim=ctx_dim, 
        fno_hidden_size=fno_hidden_size, 
        final_hidden_size=final_hidden_size, 
        n_modes=n_modes, width=width, 
        is_chunk_wise=is_chunk_wise, 
        batch_size=batch_size
    )
    print("model_name", model_name)
    if not distributed or int(os.environ['LOCAL_RANK']) == 0:
        print("model_config:")
        pp.pprint(model_config)
    if model_name == 'transformer':
        from models import DecisionTransformer
        model = DecisionTransformer(model_config).to(device)
    elif model_name == 'retnet':
        from models import DecisionRetNet
        model = DecisionRetNet(model_config).to(device)
    elif model_name == "fourier_controller":
        from models.fourier_controller import FourierController
        state_dim = src_dim - ctx_dim # ctx_dim=16 is the number of commands
        # NOTICE: state : (state, context), context is the command, in the last 16 dimensions
        model = FourierController(model_config).to(device)
    elif model_name == "mlp_imitation":
        from models.mlp_imitation import MLPImitationModel
        model = MLPImitationModel(model_config).to(device)
    elif model_name == "rnn_imitation":
        from models.rnn_imitation import RNNImitationModel
        model = RNNImitationModel(model_config).to(device)
    if local_rank == 0:
        get_parameter_number(model)
    if model_name in ["fourier_controller"]:
        # if tasks == ["unitree_general"]:
        #     weight_decay = 0.0
        optimizer = fourier_utils.Adam(
            model.parameters(),
            lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay,
        )  # for complex computation
        # weight_decay=weight_decay,
    else:
        optimizer = Lion(model.parameters(), lr=lr, weight_decay=weight_decay, 
                        use_triton=optimizer_use_triton)
    # 估计train_steps
    num_training_steps = int(epochs * train_loader_len)
    if episode_first_no_backward:
        raise NotImplementedError
        num_training_steps -= int(epochs * train_loader.max_episode_length)
    num_training_steps = sync_mgpu(num_training_steps, device, 'max')
    print(f"num_training_steps: {num_training_steps}")
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(num_training_steps * warmup_ratio), 
        num_training_steps)
    # scheduler = LambdaLR(
    #     optimizer=optimizer,
    #     lr_lambda=lambda step: rate(
    #         step, d_m, factor=1, warmup=warmup_steps,),)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_device_id],
            output_device=local_device_id,
            find_unused_parameters=False
        )

    runner = Runner(
        model=model,
        scheduler=scheduler,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        local_rank=local_rank,
        args=cfg_dict,
        device=device,
        model_config=model_config
    )
    runner.train()
    return

if __name__=='__main__':
    main()