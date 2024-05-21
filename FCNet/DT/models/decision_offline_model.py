import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecisionOfflineModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        
        if config['add_last_action']:
            self.src_dim = config['src_dim'] + config['tgt_dim']
        else:
            self.src_dim = config['src_dim']
        self.tgt_dim = config['tgt_dim']
        self.hidden_size = config['hidden_size']
        self.n_layer = config['n_layer']
        self.n_head = config['n_head']
        self.seq_len = config['seq_len']
        # self.dropout = config['dropout']
        self.dropout = 0.1
        self.ffn_coef = int(4)
        if 'ffn_coef' in config and \
                config['ffn_coef'] is not None:
            self.ffn_coef = config['ffn_coef']
            assert self.ffn_coef >= 1, self.ffn_coef
        self.data_mode = config.get('data_mode', 'nonmdp')
        self.input_data_mode = config.get('input_data_mode', 'chunk') # chunk or episode
    
    def _complete_input_tensor_feature(self, x: torch.Tensor):
        '''
            @param x: (bz, <=seq_len, <=src_dim)
        '''
        assert x.size(-1) <= self.src_dim, f"x size: {x.size(-1)}, src_dim: {self.src_dim}"
        if x.size(-1) == self.src_dim:
            y = x
        else:
            blank_tensor_shape = tuple(list(x.shape)[:-1] + [self.src_dim - x.size(-1)])
            blank_tensor = torch.zeros(blank_tensor_shape, dtype=x.dtype, device=x.device)
            y = torch.cat([x, blank_tensor], dim=-1)
        return y