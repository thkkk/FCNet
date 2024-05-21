import numpy as np
import torch

FLOAT_TYPE = (
    float,
    np.float16, np.float32, np.float64, np.float128,
    torch.FloatTensor, torch.DoubleTensor,
)

INT_TYPE = (
    int,
    np.int8, np.int16, np.int32, np.int64,
    torch.LongTensor, torch.IntTensor,
)

# disk data mode -> memory data mode
CHUNK_LOAD_MODES = (
    'chunk2chunk',
    'episode2chunk',
)
EPISODE_LOAD_MODES = (
    'episode2episode',
)

# load_data_mode to input_data_mode
LOAD2INPUT = {
    CHUNK_LOAD_MODES: 'chunk',
    EPISODE_LOAD_MODES: 'episode',
}
def load2input_func(load_data_mode: str):
    for k, v in LOAD2INPUT.items():
        if load_data_mode in k:
            return v
    return None

