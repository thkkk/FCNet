import os
# os.environ["HYDRA_FULL_ERROR"] = "1"

from omegaconf import DictConfig, OmegaConf

from mkmn_stra.utils.hydra_utils import *
from mkmn_stra.utils.common import print_dict, omegaconf_to_dict

import hydra
import time

# @hydra.main(version_base=None, config_path="../../cfg/play_save_data_cfg", config_name="config")
@hydra.main(version_base=None, config_path="../../cfg", config_name="config")
def my_app(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)
    return

if __name__ == "__main__":
    my_app()
    print('------------------- end -----------------------')
