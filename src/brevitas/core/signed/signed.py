from typing import Union
import torch
import brevitas
from brevitas.core.signed.utils import to_logistic, to_logit, to_scalar_tensor
from brevitas.function import round_ste


class ConstSign(brevitas.jit.ScriptModule):

    def __init__(self, 
                 signed:bool=True) -> None:
        super().__init__()
        self.signed = signed
        self.signed_tensor = to_scalar_tensor(float(signed))
    
    @brevitas.jit.script_method
    def forward(self, x:torch.Tensor=None):
        return self.signed_tensor
    

class ConstSigned(ConstSign):

    def __init__(self) -> None:
        super().__init__(True)


class ConstUnsigned(ConstSign):

    def __init__(self) -> None:
        super().__init__(False)


class ParameterSigned(brevitas.jit.ScriptModule):

    def __init__(self, 
                 init_signed_probability:Union[float,torch.Tensor]=0.6, 
                 trainable:bool=True) -> None:
        super().__init__()
        # convert to scalar tensor or raise Exception if is wrong (more than one element) 
        p = to_scalar_tensor(init_signed_probability)        
        # get probability argument
        arg = to_logit(p)
        # and treat it as initial value for parameter
        self.signed_param = torch.nn.Parameter(arg, requires_grad=trainable)
    
    @brevitas.jit.script_method
    def forward(self, x:torch.Tensor):
        # passed arg can be used for statistics calculations
        # self.signed belong to the whole real number domain
        # -> apply logistic transformation 
        p = to_logistic(self.signed_param)
        # round value to get 0.0 (unsigned) or 1.0 (signed)
        signed = round_ste(p)
        
        return signed
    
    @property
    def signed(self):
        with torch.no_grad():
            return (self(None) > 0.5).item()