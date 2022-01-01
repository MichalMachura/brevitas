from typing import Union
import torch
from brevitas import function as bv_fcn


def to_scalar_tensor(value:Union[float, torch.Tensor]):
    if type(value) is float:
        return torch.tensor(value, dtype=torch.float32)
    
    # must be one-element == scalar tensor
    elif isinstance(value,torch.Tensor) and value.numel() == 1:
        # to scalar tensor
        value = value.flatten()[0]
        return value.to(torch.float32)
        
    else:
        raise RuntimeError("Convesion to scalar tensor is available only" 
                           " for float scalar or torch.Tensor with one element.")


def to_logit(p:torch.Tensor):
    # limit probability to interior of (0.0; 1.0)
    margin = 0.01
    p = torch.clamp(p, min=margin, max=1.0-margin)
    # get logit
    arg = torch.logit(p)
    # limit_logit for edge cases it can be -inf or inf
    abs_max = 1.0
    arg = torch.clamp(arg, min=-abs_max, max=abs_max)
    
    return arg


def to_logistic(x:torch.Tensor):
    # use sigmoid as probaility function
    return torch.sigmoid(x)

    
    