import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierController(nn.Module):
    """
    Fourier Controller for next-state prediction:

    [s1, s2, ..., sT] ----> [a1, a2, ..., aT]\n
    [ctx1, ..., ctxT] --|

    Only predict next action based on current states and contexts. In fact, the contexts can be merged
    into the states, but we keep them separate for better understanding.

    Support both recurrent and parallel computation.
    Default: parallel computation.
    """
    def __init__(self, config:dict):
        super(FourierController, self).__init__()
        
        self.s_dim = config["src_dim"] - config["ctx_dim"]
        self.a_dim = config["tgt_dim"]
        self.ctx_dim = config["ctx_dim"]
        self.hidden_size = config["fno_hidden_size"]
        self.final_hidden_size = config["final_hidden_size"]  
            # hidden_size for the last layer
        self.n_layer = config["n_layer"]
        self.n_modes = config["n_modes"]
        if "is_chunk_wise" in config and config["is_chunk_wise"] == True:
            self.T = int(config["seq_len"] / self.n_layer)
        else:
            self.T = config["seq_len"]

        self.act = nn.GELU()
        
        self.fc0 = nn.Linear(self.s_dim + self.ctx_dim, self.hidden_size)
        
        if "precomp_dft_mat" in config:
            precomp_dft_mat = config["precomp_dft_mat"]
        else:
            precomp_dft_mat = True

        # Precompute the DFT matrix
        if precomp_dft_mat:
            W = get_dft_matrix(self.n_modes, self.T)
            # Reshape W to (1, n_modes, 1, T+1)
            W = W.reshape(1, self.n_modes, 1, -1)
            # Initialize W as a non-trainable parameter
            self.W = nn.Parameter(torch.view_as_real(W), requires_grad=False)
        else:
            self.W = None
        # Precompute the exponential terms for IDFT
        idft_exps = get_idft_exps(self.n_modes, self.T)
        self.idft_exps = nn.Parameter(
            torch.view_as_real(idft_exps), requires_grad=False)

        # Fourier Layers
        self.fourier_layers = nn.ModuleList()
        for _ in range(self.n_layer):
            self.fourier_layers.append(
                FourierLayer(self.hidden_size, self.n_modes, self.T, 
                             self.W, self.idft_exps, self.act))

        self.fc1 = nn.Linear(self.hidden_size, self.final_hidden_size)
        self.fc2 = nn.Linear(self.final_hidden_size, self.a_dim)
    
    def reset_recur(self, batch, device):
        """
        reset the computation model to initial recurrent state. Including set_recur, clear_recur_cache, init_recur_cache.
        """
        self.set_recur()
        self.clear_recur_cache()
        self.init_recur_cache(batch, device)
    
    def set_recur(self):
        '''
        Set the computation model to recurrent.
        
        Warning: 
        1. **DO NOT** use this mode in training.\n
        2. In this mode, the batch size should not
        be variable during testing.
        '''
        for layer in self.fourier_layers:
            layer.csc.is_recurrent = True
    
    def set_parall(self):
        '''
        Set the computation model to parallel.
        '''
        for layer in self.fourier_layers:
            layer.csc.is_recurrent = False
    
    def clear_recur_cache(self):
        '''
        Clear the cache for recurrent computation.
        '''
        for layer in self.fourier_layers:
            layer.csc.clear_recur_cache()
    
    def init_recur_cache(self, batch, device):
        '''
        Initialize the cache for recurrent computation.

        Parameters
        ----------
        batch: batch size
        device: device to store the cache
        '''
        for layer in self.fourier_layers:
            layer.csc.init_recur_cache(batch, device)

    def forward(self, s, ctx, prev_x_layers=None, prev_x_ft_layers=None):
        '''
        Parameters
        ---------
        s: states, (batch, T, s_dim) [parallel] or
            (batch, 1, s_dim) [recurrent]
        ctx: context, (batch, T, ctx_dim) [parallel] or
            (batch, 1, ctx_dim) [recurrent]
        prev_x_layers: the embeddings for each layer in the previous chunk, 
            (n_layers, batch, T, hidden_size)
        prev_x_ft_layers: the Fourier modes of prev_x_layers,
            (n_layers, batch, n_modes, hidden_size)
        
        Note: prev_x_layers and prev_x_ft_layers should be 
        both `None` or not `None`. If they are not `None`,
        this means that the training trajectory is chunk-wise.
        In recurrent computation, prev_x_layers and prev_x_ft_layers
        should be both `None` (i.e., no chunk-wise training), and 
        the batch size should not be variable during training.

        Return
        ------
        s_pred if prev_x_layers is None

        [s_pred, x_layers, x_ft_layers] else

        s_pred: the predicted next-window states, (batch, T, s_dim) [parallel] or
            (batch, 1, s_dim) [recurrent]
        x_layers: the embeddings for each layer in the current chunk, 
            (n_layers, batch, T, hidden_size)
        x_ft_layers: the Fourier modes of x_layers,
            (n_layers, batch, n_modes, hidden_size)

        Note: x_layers and x_ft_layers are on the same device as inputs.
        '''
        # Prepare for chunk-wise training if needed
        is_chunk_wise = prev_x_layers is not None
        x_layers, x_ft_layers = prev_x_layers, prev_x_ft_layers

        x = torch.cat((s, ctx), dim=-1)
            # (batch, T, s_dim + ctx_dim)
        
        # Apply linear transform
        x = self.fc0(x)
        
        # Apply Fourier Layers
        for i in range(self.n_layer):
            if is_chunk_wise:
                x, x_layers[i], x_ft_layers[i] = self.fourier_layers[i](
                    x, prev_x=x_layers[i], prev_x_ft=x_ft_layers[i])
            else:
                x = self.fourier_layers[i](x)

        # Apply linear transform
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        
        if is_chunk_wise:
            return x, x_layers, x_ft_layers
        else:
            return x


class FourierLayer(nn.Module):
    """
    -------> CausalSpecConv ----+---> GeLU ---> Residual Connection\n
       |---> FFNBlock ----------| 
    """
    def __init__(self, hidden_size, n_modes, T, W, idft_exps, act):
        super(FourierLayer, self).__init__()
        self.n_modes = n_modes
        self.T = T
        self.act = act

        self.csc = CausalSpecConv(
            hidden_size, n_modes, T, W, idft_exps)
        self.ffn = FFNBlock(hidden_size, hidden_size, act)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x, prev_x=None, prev_x_ft=None):
        '''
        Parameters
        ----------
        x: embedding, (batch, T, hidden_size) [parallel] or
            (batch, 1, hidden_size) [recurrent]
        prev_x: embedding in the previous chunk, (batch, T, hidden_size)
        prev_x_ft: modes of prev_x, (batch, n_modes, hidden_size)

        Note: prev_x and prev_x_ft should be both `None` or not `None`.
        In recurrent computation, prev_x and prev_x_ft should be both `None`.

        Return
        ------
        [out, x, x_ft] if prev_x is not None

        out if prev_x is None

        out: the output embedding, (batch, T, hidden_size) [parallel] or
            (batch, 1, hidden_size) [recurrent]
        x_ft: modes of x, (batch, n_modes, hidden_size)
        '''
        is_chunk_wise = prev_x is not None
        resid_x = x
        x = self.ln1(x)
        if is_chunk_wise:
            x, _x, _x_ft = self.csc(x, prev_x=prev_x, prev_x_ft=prev_x_ft)
        else:
            x = self.csc(x)
        x = self.act(x) + resid_x

        resid_x = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = x + resid_x

        if is_chunk_wise:
            return x, _x, _x_ft
        return x


class FFNBlock(nn.Module):
    """
    Linear -> GeLU -> Linear
    """
    def __init__(self, hidden_size, intermediate_size, act):
        super(FFNBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.act = act

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class CausalSpecConv(nn.Module):
    """
    STFT (window_len=T) -> linear transform -> Inverse STFT  
    """
    def __init__(self, hidden_size, n_modes, T, W, idft_exps):
        super(CausalSpecConv, self).__init__()
        self.hidden_size = hidden_size
        self.n_modes = n_modes 
            # number of Fourier modes to be retained, at most (T + 1) // 2
        if n_modes > (T + 1) // 2:
            raise ValueError('''
                n_modes should be less than or equal to (T + 1) // 2''')
        self.T = T  # sequence length
        # The precomputed DFT matrix
        self.W = W  # (1, n_modes, 1, T+1)
        # The precomputed exponential terms for IDFT
        self.idft_exps = idft_exps  # (n_modes,)
        # Recurrent or parallel computation
        # Default: parallel computation
        self.is_recurrent = False

        weights = 1 / n_modes * torch.randn(
            n_modes, n_modes, hidden_size, dtype=torch.cfloat)
        self.weights = nn.Parameter(torch.view_as_real(weights))

        # Cache for recurrent computation
        self.recur_cache = {}

    def init_recur_cache(self, batch, device):
        '''
        Initialize the cache for recurrent computation.

        x_window: the embedding in the window [s0, s1, ..., sT-1],
            where sT shoulde be the latest state and sT+1 should be
            the next state to be predicted, [(batch, 1, hidden_size)] x T
        x_window_ptr: the pointer to the oldest state s0 in x_window, int
        x_ft: the Fourier modes of x_window, (batch, n_modes, hidden_size)
        exp_term: the exponential term in the sliding DFT,
            (1, n_modes, 1)
        '''
        mode_idx = torch.arange(0, self.n_modes, 
            dtype=torch.float32, device=device)
        exp_term = torch.exp((1j * 2 * torch.pi * mode_idx / self.T))
        # Circular buffer
        x_window_ptr = 0
        x_window = [torch.zeros(batch, 1, self.hidden_size, 
                                dtype=torch.float32, device=device)
                    for _ in range(self.T)]
        # Since the weights are fixed during testing,
        # we can change the order of the computation,
        # and precompute the weights @ idft_exps
        weights = torch.view_as_complex(self.weights).permute(2, 0, 1)  
        # (hidden_size, n_modes, n_modes)
        idft_exps = torch.view_as_complex(self.idft_exps)  # (n_modes,)
        idft_weights = torch.matmul(weights, idft_exps).permute(1, 0)
        # (n_modes, hidden_size)
        self.recur_cache = {
            'x_window': x_window,
            'x_window_ptr': x_window_ptr,
            'x_ft': torch.zeros(batch, self.n_modes, self.hidden_size,
                                    dtype=torch.cfloat, device=device),
            'exp_term': exp_term.reshape(1, self.n_modes, 1),
            'idft_weights': idft_weights
        }
    
    def clear_recur_cache(self):
        self.recur_cache = {}
    
    def update_recur_cache(self, x, x_ft):
        '''
        Update the cache for recurrent computation.

        Parameters
        ----------
        x: the embedding of the latest state sT, (batch, 1, hidden_size)
        x_ft: the Fourier modes of [s1, s2, ..., sT], 
            (batch, n_modes, hidden_size)
        '''
        ptr = self.recur_cache['x_window_ptr']
        self.recur_cache['x_window'][ptr] = x
        self.recur_cache['x_window_ptr'] = (ptr + 1) % self.T
        self.recur_cache['x_ft'] = x_ft

    def get_state_from_cache(self):
        '''
        Return the oldeset state s0 in the cache.
        '''
        ptr = self.recur_cache['x_window_ptr']
        return self.recur_cache['x_window'][ptr]

    def _forward_recur(self, x, prev_x=None, prev_x_ft=None):
        '''
        Forward by recurrent computation.

        Do not use this function directly.
        '''
        # Recurrent computation should not be chunk-wise
        if prev_x is not None or prev_x_ft is not None:
            raise ValueError('''
                prev_x and prev_x_ft should be 
                both None for recurrent computation''')
        
        # Sliding DFT
        x_ft = self.recur_cache['x_ft']
        exp_term = self.recur_cache['exp_term']
        x0 = self.get_state_from_cache()
        x_ft = exp_term * (x_ft + (x - x0))
            # (batch, n_modes, hidden_size)

        # Update the cache
        self.update_recur_cache(x, x_ft)

        # Linear transform & IDFT
        idft_weights = self.recur_cache['idft_weights']
        out = torch.einsum("btmd,md->btd", 
                            x_ft.unsqueeze(1), idft_weights).real
        # x_ft.unsqueeze(1): (batch, 1, n_modes, hidden_size)
        # idft_weights: (n_modes, hidden_size)
        # complexity: O(batch * n_modes * hidden_size^2)
        # out: (batch, 1, hidden_size)

        return out

    def forward(self, x, prev_x=None, prev_x_ft=None):
        '''
        Parameters
        ----------
        x: embedding, (batch, T, hidden_size) [parallel] or 
            (batch, 1, hidden_size) [recurrent]
        prev_x: embedding in the previous chunk, (batch, T, hidden_size)
        prev_x_ft: modes of prev_x, (batch, n_modes, hidden_size)

        Note: prev_x and prev_x_ft should be both `None` or not `None`.
        In recurrent computation, prev_x and prev_x_ft should be both `None`.

        Return
        ------
        [out, x, x_ft] if prev_x is not None

        out if prev_x is None
            
        out: the output embedding, (batch, T, hidden_size) [parallel] or
            (batch, 1, hidden_size) [recurrent]
        x_ft: modes of x, (batch, n_modes, hidden_size)
        '''
        # Recurrent computation
        if self.is_recurrent:
            return self._forward_recur(x, prev_x, prev_x_ft)
        
        # Parallel computation
        batch = x.shape[0]  # batch size
        dim = x.shape[-1]   # hidden_size
        T = self.T

        # The results in the previous chunk
        if prev_x is not None and prev_x_ft is None or \
            prev_x is None and prev_x_ft is not None:
            raise ValueError('''prev_x and prev_x_ft 
                             should be both None or not None''')
        is_chunk_wise = prev_x is not None
        if not is_chunk_wise:
            # prev_x is the embedding in the previous chunk
            # prev_x: (batch, T, dim)
            prev_x = torch.zeros_like(x)
            # prev_x_ft are the modes of prev_x[:, 0:T, :]
            # prev_x_ft: (batch, n_modes, dim)
            prev_x_ft = torch.zeros(
                batch, self.n_modes, dim, 
                dtype=torch.cfloat, device=x.device)

        # Concatenate the previous chunk and the current chunk
        x_cat = torch.cat((prev_x, x), dim=-2)
            # (batch, 2T, dim)

        # Apply Short-Time Fourier Transform
        # x_stft[:, i, ...] is the result of sliding DFT
        # in x_cat[:, i+1:i+1+T, ...]
        # x_stft: (batch, T, n_modes, dim),
        # Get the DFT matrix
        if self.W is not None:
            w = torch.view_as_complex(self.W)
        else:
            w = get_dft_matrix(self.n_modes, T, device=x_cat.device)
            w = w.reshape(1, self.n_modes, 1, -1)
            w = w.to(x_cat.device)
                # (1, n_modes, 1, T+1)

        # Parallelly compute the convolution
        # f is the first sequence in the convolution
        # f: (batch, n_modes, dim, T+1)
        # f[..., 0] = prev_x_ft
        # f[..., i] = -x_cat[:, i-1:i, :] + x_cat[:, i-1+T:i+T, :], i=1~T
        f = (-x_cat[:, :T, :] + x_cat[:, T:2*T, :])\
            .permute(0, 2, 1).unsqueeze(1)
            # (batch, 1, dim, T)
        f = f.expand(-1, self.n_modes, -1, -1)
            # (batch, n_modes, dim, T)
        f = torch.concat((prev_x_ft.unsqueeze(-1), f), dim=-1)
            # (batch, n_modes, dim, T+1)

        # Let's start the convolution
        # f:            (batch, n_modes, dim, T+1)
        # w:            (1,     n_modes, 1,   T+1)
        # conv(f, w):   (batch, n_modes, dim, T+1)
        f = torch.fft.fft(f, n=2*(T+1)-1, dim=-1)
        w = torch.fft.fft(w, n=2*(T+1)-1, dim=-1)
        f = f * w
        x_stft = torch.fft.ifft(f, n=2*(T+1)-1, dim=-1)[..., 1:T+1]  
            # (batch, n_modes, dim, T)
        x_stft = x_stft.permute(0, 3, 1, 2)
            # (batch, T, n_modes, dim)
        
        # Prepare x, x_ft for the this chunk
        if is_chunk_wise:
            x_ft = x_stft[:, -1, :, :]

        # Apply linear transform
        weights = torch.view_as_complex(self.weights)
        x_stft = torch.einsum("btmd,mnd->btnd", x_stft, weights)

        # Apply IDFT
        idft_exps = torch.view_as_complex(self.idft_exps)
        out = torch.einsum("btmd,m->btd", x_stft, idft_exps).real
            # (batch, T, dim)

        if is_chunk_wise:
            return out, x, x_ft
        return out


def get_dft_matrix(n_modes, T, device=None):
    """
    Return the truncated DFT matrix W of size (n_modes, T+1):

    W[i1, i2] = exp(1j * 2 * pi * i1 * (i2+1) / T), 
    i1=0~n_modes-1, i2=0~T-1

    W[:, T] = W[:, T-1] for boundary condition in convolution
    """
    time_idx = torch.arange(1, T + 2, 
        dtype=torch.float32, device=device)
    mode_idx = torch.arange(0, n_modes, 
        dtype=torch.float32, device=device)
    full_idx = torch.outer(mode_idx, time_idx)
    W = torch.exp((1j * 2 * torch.pi * full_idx / T))
    W[:, -1] = W[:, -2]

    return W


def get_idft_exps(n_modes, T, device=None):
    """
    Return the exponential terms for IDFT:

    idft_exps[0] = 1 / T,
    idft_exps[i] = 2 * exp(1j * 2 * pi * i / T * (T-1)) / T, 
    i=(1~n_modes-1)

    Note: this term is used for computing the last element in
    the IDFT sequence.
    """
    mode_idx = torch.arange(0, n_modes, 
            dtype=torch.float32, device=device)
    idft_exps = torch.exp((
        1j * 2 * torch.pi * mode_idx / T * (T - 1))) / T
    idft_exps[1:] = idft_exps[1:] * 2
    
    return idft_exps
