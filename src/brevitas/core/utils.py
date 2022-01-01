from typing import Optional, Union
import torch

import brevitas

VALUE_ATTR_NAME = 'value'


@torch.jit.ignore
def inplace_tensor_add(tensor: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    tensor.mul_(value)
    return tensor


@torch.jit.ignore
def inplace_tensor_mul(tensor: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    tensor.mul_(value)
    return tensor


@torch.jit.ignore
def inplace_momentum_update(
        tensor: torch.Tensor,
        update: torch.Tensor,
        momentum: Optional[float],
        counter: int,
        new_counter: int) -> torch.Tensor:
    if momentum is None:
        tensor.mul_(counter / new_counter)
        tensor.add_(update / new_counter)
    else:
        tensor.mul_(1 - momentum)
        tensor.add_(momentum * update)
    return tensor


class StatelessBuffer(brevitas.jit.ScriptModule):

    def __init__(self, value: torch.Tensor):
        super(StatelessBuffer, self).__init__()
        self.register_buffer(VALUE_ATTR_NAME, value)
    
    @brevitas.jit.script_method
    def forward(self):
        return self.value.detach()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(StatelessBuffer, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        value_key = prefix + VALUE_ATTR_NAME
        if value_key in missing_keys:
            missing_keys.remove(value_key)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        output_dict = super(StatelessBuffer, self).state_dict(destination, prefix, keep_vars)
        del output_dict[prefix + VALUE_ATTR_NAME]
        return output_dict


@brevitas.jit.script
def to_scalar_tensor(value:Union[float, torch.Tensor])->torch.Tensor:
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


@brevitas.jit.script
def to_logit(p:torch.Tensor)->torch.Tensor:
    # limit probability to interior of (0.0; 1.0)
    p = torch.clamp(p, min=0.1, max=0.9)
    # get logit
    arg = torch.logit(p)
    # limit_logit for edge cases it can be -inf or inf
    abs_max = 1.0
    arg = torch.clamp(arg, min=-abs_max, max=abs_max)
    
    return arg


@brevitas.jit.script
def to_logistic(x:torch.Tensor)->torch.Tensor:
    # use sigmoid as probaility function
    return torch.sigmoid(x)
    