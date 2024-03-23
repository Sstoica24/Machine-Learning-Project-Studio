import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    # import pdb; pdb.set_trace()
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    # always output fan_in by fan_out tensors. rand pulls from
    # uniform dist
    #because shape is passed in as *shape, you can't pass
    # in tuple value for shape and instead just ints. 
    return rand(fan_in, fan_out, low=-a, high=a)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    # import pdb; pdb.set_trace()
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(fan_in, fan_out) * std
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    bound = math.sqrt(2) * math.sqrt(3 / (fan_in))
    return rand(fan_in, fan_out, low=-bound, high=bound)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    std = math.sqrt(2) / math.sqrt(fan_in)
    return randn(fan_in, fan_out) * std
    ### END YOUR SOLUTION
