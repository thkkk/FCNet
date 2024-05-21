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

import hydra
from omegaconf import DictConfig, OmegaConf
from FCNet.DT.utils.hydra_utils import *
from FCNet.DT.utils.common import print_dict, omegaconf_to_dict, get_availble_gpus

import os
os.environ['HYDRA_FULL_ERROR'] = '1'

availble_gpus = get_availble_gpus()
print(f'availble_gpus: {availble_gpus}')

# @hydra.main(version_base=None, config_path="../../cfg/train_mlp_cfg", config_name="config")
@hydra.main(version_base=None, config_path="../../cfg", config_name="config")
def main(cfg: DictConfig):
    global availble_gpus
    # yaml config 转为 dict
    cfg_dict = omegaconf_to_dict(cfg)
    # yaml 参数
    task = cfg_dict['task_name']
    headless = cfg_dict['headless']
    local_rank = cfg_dict['local_rank']
    print(f'local_rank: {local_rank}')
    assert len(availble_gpus) > local_rank
    
    # 运行
    cmd = f'CUDA_VISIBLE_DEVICES={availble_gpus[local_rank]} python ./sub/train_sub.py --task {task}'
    if headless: cmd += ' --headless'
    os.system(cmd)
    return

if __name__ == '__main__':
    main()