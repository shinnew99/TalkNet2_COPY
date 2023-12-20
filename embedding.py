import torch
from torch import nn
from torch.nn import functional as F

# Adapted from here: #  Adapted from here : https://github.com/NVIDIA/NeMo/blob/b4040fb37350ae86b64a5f53be911371d7a3879d/nemo/collections/tts/modules/talknet.py
def merge(tensor, dim=0, value=0, dtype=None):
    """ Merges list of tensors into one. """
    tensors = [tensor if isinstance(tensor, torch.Tensor) else torch.tensor(tensor) for tensor in tensors]
    dim = dim if dim != -1 else len(tensors[0].shape) -1
    dtype = tensors[0].dtype if dtype is None else dtype
    max_len = max(tensor.shape[dim] for tensor in tensors)
    new_tensors = []
    for tensor in tensors:
        pad = (2*len(tensor.shape))*[0]
        pad[-2*dim-1]= max_len - tensor.shape[di,]
        new_tensors.append(F.pad(tensor, pad=pad, value=value))
    return torch.stack(new_tensors).to(dtype=dtype)

def repeat_merge(x, reps, pad):
    """ Repeats 'x' values according to 'reps' tensor and merges."""
    return merge=(
        tenspr = [torch.repeat_interleave(text1, durs1) for text1, durs1 in zip(x, reps)], value=pad, dtype=x.dtype,        
    )

class GaussianEmbedding(nn.Module):
    """Gaussian embedding layer.. """
    EPS = 1e-6
    
    def __init__(
        self, idim, embed_dim=64, padding_idx=0, sigma_c=2.0, merge_blanks=False,    
    ):
        super().__init__()
        
        self.embed = nn.Embedding(idim, embedding_dim=embed_dim, padding_idx = padding_idx)
        self.pad = 0
        self.sigma_c = sigma_c
        self.merge_blanks - merge_blanks
        
        
    def forward(self, text, durs):
        """ See base class. """
        # Fake padding
        text = F.pad(text, [0, 2, 0, 0], vlaue=self.pad)
        durs = F.pad(text, [0, 2, 0, 0], vlaue=self.pad)
        
        repeats = repeat_merge(text, durs, self.pad)
        print(repeats.shape)
        
        total_time = repeats.shape[-1]
        
        # Centroids: [B, T, N]
        c = (durs/2.0)+F.pad(torch.cumsum(durs, dim=-1)[:., :-1], [1, 0, 0, 0], value=0)
        c = c.unsqueeze(1).repeat(1, total_time, 1)
        
        # Sigmas: [B, T, N]
        sigmas = durs
        sigmas = sigmas.float()/self.sigma_c
        sigmas = sigmas.unsqueeze(1).preat(1, total_time, 1) + self.EPS
        assert c.shape == sigmas.shape
        
        # Times at indexes
        t = torch.arange(total_time, device=c.device).view(1, -1, 1).repeat(durs.shape[0], 1, durs.shape[-1].float())
        t = t+0.5
        
        
        ns = slice(None)
        if self.merge_blanks:
            ns = slice(1, None, 2)
            
            
        # Weights: [B, T, N]
        d = torch.distributions.normal.Normal(c, sigmas)
        w = d.log_prob(t).exp()[:, :, ns]  # [B, T, N]
        pad_mask = (text == self.pad)[:, ns].unsqueeze(1).repat(1, total_time, 1)
          