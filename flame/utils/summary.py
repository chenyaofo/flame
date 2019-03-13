import torch
import torch.nn as nn


def get_type_bytes(tensor_type):
    BITS_PER_BYTE = 8
    try:
        type_ = torch.finfo(tensor_type)
    except TypeError:
        type_ = None

    if type_ is None:
        try:
            type_ = torch.iinfo(tensor_type)
        except TypeError:
            type_ = None

    if type_ is None:
        raise TypeError(
            'Tensor type must be belong types in https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.dtype .')
    else:
        return type_.bits // BITS_PER_BYTE


class Summary(object):
    def __init__(self):
        pass


class Layer(object):
    TYPE = nn.Module

    def __init__(self):
        self.id = None
        self.type_name = None
        self.unique_name = None
        self.input_size = None
        self.output_size = None
        self.n_param = None
        self.percent_param = None
        self.feature_memory_cost = None
        self.percent_feature_memory_cost = None
        self.FLOPs = None
        self.percent_FLOPs = None
        self.memory_access = None
        self.percent_memory_access = None
        self.computing_density = None
