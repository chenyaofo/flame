import torch
import torch.nn.modules as modules

import inspect


def get_primitive_nn_modules():
    modules = []
    for name, obj in inspect.getmembers(modules):
        if inspect.isclass(obj):
            if not issubclass(obj,torch.nn.Container):
                modules.append(obj)
    return modules

primitive_nn_modules = get_primitive_nn_modules()


register_type_func = dict()


def register_layer(layer_type: torch.nn.Module):
    if not issubclass(layer_type, torch.nn.Module):
        raise TypeError("The layer type to register should be the subclass of torch.nn.Module .")

    def decorator(f):
        if not callable(f):
            raise TypeError("The register function should a callable object.")
        register_type_func[layer_type] = f
        return f

    return decorator


@register_layer(layer_type=torch.nn.Conv2d)
def conv2d_flops(module: torch.nn.Conv2d, input: torch.Tensor, output: torch.Tensor):
    in_channels, out_channels, kernel_h, kernel_w = module.weight.size()
    batch_size, _, output_h, output_w = output.size()
    if module.bias is None:
        flops = out_channels * output_h * output_w * (2 * in_channels * kernel_h * kernel_w - 1)
    else:
        flops = 2 * in_channels * out_channels * output_h * output_w * kernel_h * kernel_w
    return flops


@register_layer(layer_type=torch.nn.BatchNorm2d)
def batchnorm2d_flops(module: torch.nn.Conv2d, input: torch.Tensor, output: torch.Tensor):
    batch_size, out_channels, output_h, output_w = output.size()
    flops = (batch_size * out_channels * output_h * output_w) * 4
    return flops


@register_layer(layer_type=torch.nn.ReLU)
def relu_flops(module: torch.nn.Conv2d, input: torch.Tensor, output: torch.Tensor):
    return 0


@register_layer(layer_type=torch.nn.MaxPool2d)
def maxpool2d_flops(module: torch.nn.Conv2d, input: torch.Tensor, output: torch.Tensor):
    return 0


@register_layer(layer_type=torch.nn.AvgPool2d)
def avgpool2d_flops(module: torch.nn.Conv2d, input: torch.Tensor, output: torch.Tensor):
    return 0


@register_layer(layer_type=torch.nn.Linear)
def linear_flops(module: torch.nn.Conv2d, input: torch.Tensor, output: torch.Tensor):
    input = input[0]
    batch_size, in_features = input.size()
    _, output_features = output.size()
    if module.bias is None:
        flops = batch_size * (2 * in_features - 1) * output_features
    else:
        flops = 2 * batch_size * in_features * output_features
    return flops
