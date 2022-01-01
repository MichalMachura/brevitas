from typing import Union
import torch
import brevitas
from brevitas.core.utils import to_logistic, to_logit, to_scalar_tensor
from brevitas.function import round_ste


class SignedImpl(brevitas.jit.ScriptModule):
    """
    Abstract class of signed - quantization property - implementation.
    Derrived classes 
    - should implement:
        `signed` - property that returns bool of quantization type: signed (True) 
                    or not (False)
        `forward_get` - method return current `signed` value as float (main feature)
    
    - can implement:
        `analyse` - method for tensor analysis (before quantization) 
    """
    def __init__(self) -> None:
        super().__init__()
    
    @property
    def signed(self)->bool:
        raise NotImplementedError("Method `signed` is not implemented!")
    
    @brevitas.jit.script_method
    def analyse(self, x:torch.Tensor):
        # override this method to analyse quantized tensor e.g.
        # for inference of 'signed' from stats
        pass
        
    @brevitas.jit.script_method
    def forward_get(self)->torch.Tensor:
        # method should return float value of 'signed' 
        raise NotImplementedError("Method `forward_get` is not implemented!")
    
    @brevitas.jit.script_method
    def forward(self, x:torch.Tensor=None)->torch.Tensor:
        # when arg is passed
        if x is not None:
            # do some analysis of it
            self.analyse(x)
        
        # get current state of signed
        signed_float = self.forward_get()
        
        return signed_float


class ConstSigned(SignedImpl):

    def __init__(self, 
                 signed:Union[bool,SignedImpl]=True) -> None:
        super().__init__()
        # convert to bool
        signed = signed.signed if issubclass(signed, SignedImpl) else signed
        
        self.signed:bool = signed
        self.signed_tensor:torch.Tensor = to_scalar_tensor(float(signed))
    
    @brevitas.jit.script_method
    def forward_get(self)->torch.Tensor:
        return self.signed_tensor
    

class Signed(ConstSigned):

    def __init__(self) -> None:
        super().__init__(True)


class Unsigned(ConstSigned):

    def __init__(self) -> None:
        super().__init__(False)


class ParameterSigned(SignedImpl):
    """
    Args:
        signed (bool,float,torch.Tensor,SignImpl): initial propability value of signed.
                Should have value in range of 0.0 to 1.0.
                Bool is converted to float 0.0 (False) or 1.0 (True)
        trainable (bool): Flag which determines if parameter is trainable or not
    
    Returns:
        torch.Tensor: Value of signed 0.0 or 1.0
    """

    def __init__(self, 
                 signed:Union[bool,float,torch.Tensor,SignedImpl]=0.6, 
                 trainable:bool=True) -> None:
        super().__init__()
        # types conversion
        signed = signed.signed if issubclass(signed, SignedImpl) else signed
        init_signed_probability = float(signed) if type(signed) is bool else signed
        # convert to scalar tensor or raise Exception if is wrong (more than one element) 
        p = to_scalar_tensor(init_signed_probability)        
        # get probability argument
        arg = to_logit(p)
        # and treat it as initial value for parameter
        self.signed_param = torch.nn.Parameter(arg, requires_grad=trainable)
    
    @brevitas.jit.script_method
    def forward_get(self)->torch.Tensor:
        # self.signed belong to the whole real number domain
        # -> apply logistic transformation 
        p = to_logistic(self.signed_param)
        # round value to get 0.0 (unsigned) or 1.0 (signed)
        signed = round_ste(p)
        
        return signed
    
    @property
    def signed(self)->bool:
        with torch.no_grad():
            # call forward without arg
            return (self.forward_get() > 0.5).item()
    