from typing import Union
import torch
import brevitas
from brevitas.core.utils import to_logistic, to_logit, to_scalar_tensor
from brevitas.function import round_ste
from tests.brevitas.core.int_quant_fixture import narrow_range


class NarrowRangeImpl(brevitas.jit.ScriptModule):
    """
    Abstract class of narrow range - quantization property - implementation.
    Derrived classes 
    - should implement:
        `narrow_range` - property that returns bool of quantization type: narrow range (True) 
                    or not (False)
        `forward_get` - method return current `narrow_range` value as float (main feature)
    
    - can implement:
        'analyse' - method for tensor analysis (before quantization) 
    """
    def __init__(self) -> None:
        super().__init__()
    
    @property
    def narrow_range(self)->bool:
        raise NotImplementedError("Method `narrow_range` is not implemented!")
    
    @brevitas.jit.script_method
    def analyse(self, x:torch.Tensor):
        # override this method to analyse quantized tensor e.g.
        # for inference of 'signed' from stats
        pass
    
    @brevitas.jit.script_method
    def forward_get(self)->torch.Tensor:
        # method should return float value of 'signed' 
        raise NotImplementedError("Method `narrow_range` is not implemented!")
    
    @brevitas.jit.script_method
    def forward(self, x:torch.Tensor=None)->torch.Tensor:
        # when arg is passed
        if x is not None:
            # do some analysis of it
            self.analyse(x)
        
        # get current state of narrow range
        narrow_range_float = self.forward_get()
        
        return narrow_range_float


class ConstNarrowRange(NarrowRangeImpl):

    def __init__(self, 
                 narrow_range:Union[bool,NarrowRangeImpl]=True) -> None:
        super().__init__()
        # convert to bool
        narrow_range = narrow_range.narrow_range if issubclass(narrow_range, NarrowRangeImpl) else narrow_range
        
        self.narrow_range:bool = narrow_range
        self.narrow_range_tensor:torch.Tensor = to_scalar_tensor(float(narrow_range))
    
    @brevitas.jit.script_method
    def forward_get(self)->torch.Tensor:
        return self.narrow_range_tensor
    

class NarrowRange(ConstNarrowRange):

    def __init__(self) -> None:
        super().__init__(True)


class BasicRange(ConstNarrowRange):

    def __init__(self) -> None:
        super().__init__(False)


class ParameterNarrowRange(NarrowRangeImpl):
    """
    Args:
        narrow_range (bool,float,torch.Tensor,SignImpl): initial propability value of narrow range.
                Should have value in range of 0.0 to 1.0.
                Bool is converted to float 0.0 (False) or 1.0 (True)
        trainable (bool): Flag which determines if parameter is trainable or not
    
    Returns:
        torch.Tensor: Value of narrow range 0.0 or 1.0
    """

    def __init__(self, 
                 narrow_range:Union[bool,float,torch.Tensor,NarrowRangeImpl]=0.4, 
                 trainable:bool=True) -> None:
        super().__init__()
        # types conversion
        narrow_range = narrow_range.signed if issubclass(narrow_range, NarrowRangeImpl)\
                                           else narrow_range
        init_narrow_range_probability = float(narrow_range) if type(narrow_range) is bool\
                                                            else narrow_range
        # convert to scalar tensor or raise Exception if is wrong (more than one element) 
        p = to_scalar_tensor(init_narrow_range_probability)        
        # get probability argument
        arg = to_logit(p)
        # and treat it as initial value for parameter
        self.narrow_range_param = torch.nn.Parameter(arg, requires_grad=trainable)
    
    @brevitas.jit.script_method
    def forward_get(self)->torch.Tensor:
        # self.signed belong to the whole real number domain
        # -> apply logistic transformation 
        p = to_logistic(self.narrow_range_param)
        # round value to get 0.0 (narrowed range) or 1.0 (basic range)
        narrow_range = round_ste(p)
        
        return narrow_range
    
    @property
    def narrow_range(self)->bool:
        with torch.no_grad():
            # call forward 
            return (self.forward_get() > 0.5).item()