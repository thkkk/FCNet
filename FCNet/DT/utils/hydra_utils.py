import hydra
from omegaconf import DictConfig, OmegaConf

## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if bool(pred) else b)
# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
# num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

OmegaConf.register_new_resolver('or', lambda x, y: bool(x) or bool(y))
OmegaConf.register_new_resolver('and', lambda x, y: bool(x) and bool(y))
OmegaConf.register_new_resolver('not', lambda x: not x)
def check_choices(val:str, flag:bool, lis:list) -> str:
    # assert isinstance(flag, bool), f"flag: {flag}, {type(flag)}"
    if bool(flag):
        assert val in lis, f"{val} is not in {lis}"
    return val
OmegaConf.register_new_resolver('check_choices', check_choices)

