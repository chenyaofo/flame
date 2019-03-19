import time
import typing
import functools

import torch
import torch.nn as nn

import flame

from .layers import register_type_func


def leaf_named_modules(module, memo=None, prefix=''):
    if memo is None:
        memo = set()
    if module not in memo:
        memo.add(module)
        if len(module._modules.items()) == 0:
            yield prefix, module
        for name, module in module._modules.items():
            if module is None:
                continue
            submodule_prefix = prefix + ('.' if prefix else '') + name
            for m in leaf_named_modules(module, memo, submodule_prefix):
                yield m


def replace_batch_size(size: list, batch_size: int):
    _, *remains = size
    return [batch_size] + remains


def to_human_format(value: int):
    units = ["", "K", "M", "G"]

    for unit in units:
        if value < 1000.0:
            return "{:.2f}{}".format(value, unit)
        else:
            value /= 1000.0
    return "{:.2f}P".format(value)


def get_type_bytes(tensor):
    tensor_type = tensor.dtype
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


class LayerSummary(object):
    TYPE = nn.Module

    def __init__(self):
        self.id = None
        self.type_name = None
        self.unique_name = None

        self.input_size = None

        self.n_output = None
        self.output_size = None
        self.output_memory_cost = None
        self.output_percent = None

        self.n_param = 0
        self.param_memory_cost = 0
        self.param_percent = None

        self.FLOPs = 0
        self.FLOPs_percent = 0.

        self.memory_access = None
        self.memory_access_percent = None

        self.computing_density = None


def to_tensor_list(obj):
    if torch.is_tensor(obj):
        return [obj]
    else:
        return list(filter(torch.is_tensor, obj))


def summary_hook(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor, batch_size: int,
                 summary: LayerSummary):
    # preprocess
    inputs = to_tensor_list(input)
    outputs = to_tensor_list(output)

    summary.type_name = str(module.__class__).split(".")[-1].split("'")[0]

    n_input = sum([inp.numel() for inp in inputs]) * batch_size
    summary.input_size = [tuple(replace_batch_size(list(inp.size()), batch_size)) for inp in inputs]
    input_memory_cost = sum([inp.numel() * get_type_bytes(inp) for inp in inputs]) * batch_size

    summary.n_output = sum([outp.numel() for outp in outputs]) * batch_size
    summary.output_size = [tuple(replace_batch_size(list(outp.size()), batch_size)) for outp in outputs]
    summary.output_memory_cost = sum([outp.numel() * get_type_bytes(outp) for outp in outputs]) * batch_size

    if hasattr(module, "weight") and torch.is_tensor(module.weight):
        n_weight = module.weight.numel()
        summary.n_param += n_weight
        summary.param_memory_cost += n_weight * get_type_bytes(module.weight)
    elif hasattr(module, "bias") and torch.is_tensor(module.bias):
        n_bias = module.weight.numel()
        summary.n_param += n_bias
        summary.param_memory_cost += n_bias * get_type_bytes(module.weight)

    try:
        FLOPs_func = register_type_func[module.__class__]
    except KeyError:
        flame.logger.error("The module type {} is not register, set the FLOPs to 0.".format(module.__class__.__name__))
    else:
        summary.FLOPs = FLOPs_func(module, input, output) * batch_size

    summary.memory_access = input_memory_cost + summary.param_memory_cost + summary.output_memory_cost

    summary.computing_density = summary.FLOPs / summary.memory_access


def register_summary_hook(module: torch.nn.Module, model, hooks):
    if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList, torch.nn.ModuleDict)) or module == model:
        pass
    else:
        hooks.append(module.register_forward_hook(summary_hook))


class Summary(object):
    def __init__(self, name):
        self.name = name
        self.layers: typing.List[LayerSummary] = []
        self.total = LayerSummary()

    def gather(self):
        self.total.n_output = sum(map(lambda item: item.n_output, self.layers))
        self.total.output_memory_cost = sum(map(lambda item: item.output_memory_cost, self.layers))
        self.total.n_param = sum(map(lambda item: item.n_param, self.layers))
        self.total.param_memory_cost = sum(map(lambda item: item.param_memory_cost, self.layers))
        self.total.FLOPs = sum(map(lambda item: item.FLOPs, self.layers))
        self.total.memory_access = sum(map(lambda item: item.memory_access, self.layers))
        self.total.computing_density = self.total.FLOPs / self.total.memory_access
        for layer_summary in self.layers:
            layer_summary.output_percent = layer_summary.n_output / self.total.n_output
            layer_summary.param_percent = layer_summary.n_param / self.total.n_param
            layer_summary.FLOPs_percent = layer_summary.FLOPs / self.total.FLOPs
            layer_summary.memory_access_percent = layer_summary.memory_access / self.total.memory_access

    def to_pretty_table(self):
        try:
            from prettytable import PrettyTable
        except ImportError:
            raise ImportError("Please install package prettytable to use 'to_pretty_table' func.")
        table = PrettyTable()
        table.field_names = ["#", "type", "name", "in. size", "out. size", "out. mem", "out. %",
                             "# param", "param mem", "param %", "FLOPs", "FLOPs %", "MAC", "MAC %", "computing density"]
        for s in self.layers:
            table.add_row([s.id, s.type_name, s.unique_name, "\n".join(map(str, s.input_size)),
                           "\n".join(map(str, s.output_size)),
                           to_human_format(s.output_memory_cost), to_human_format(s.output_percent * 100),
                           to_human_format(s.n_param), to_human_format(s.param_memory_cost),
                           to_human_format(s.param_percent * 100),
                           to_human_format(s.FLOPs), to_human_format(s.FLOPs_percent * 100),
                           to_human_format(s.memory_access), to_human_format(s.memory_access_percent * 00),
                           to_human_format(s.computing_density)])
        table.add_row(["total", "", "", "",
                       to_human_format(self.total.n_output), "",
                       to_human_format(self.total.output_memory_cost), "",
                       to_human_format(self.total.n_param), to_human_format(self.total.param_memory_cost), "",
                       to_human_format(self.total.FLOPs), "",
                       to_human_format(self.total.memory_access), "",
                       to_human_format(self.total.computing_density)])
        return table

    def to_html(self, path=None):
        html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Table</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/semantic-ui/2.4.1/semantic.min.css">
</head>
<body>
'''
        html += '''<h1 align="center">{}</h1>
<p align="center">model summary</p>
<p align="center">generate time: {}</p>
'''.format(self.name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        html += '''<table class="ui sortable striped table">
    <thead class="table__head">
    <tr>
        <th>#</th>
        <th>type</th>
        <th>name</th>
        <th>in. size</th>

  
        <th>out. size</th>
        <th>out. mem</th>
        <th>out. %</th>

        <th># param</th>
        <th>param mem</th>
        <th>param %</th>

        <th>FLOPs</th>
        <th>FLOPs %</th>

        <th>MAC</th>
        <th>MAC %</th>

        <th>computing density</th>
    </tr>
    </thead>
    <tbody>
'''
        for s in self.layers:
            html += '''    <tr>
        <td>{}</td>
        <td>{}</td>
        <td>{}</td>
        <td>{}</td>

        <td>{}</td>
        <td>{}</td>
        <td>{}</td>

        <td>{}</td>
        <td>{}</td>
        <td>{}</td>

        <td>{}</td>
        <td>{}</td>

        <td>{}</td>
        <td>{}</td>

        <td>{}</td>

    </tr>
'''.format(s.id, s.type_name, s.unique_name, "<br/>".join(map(str, s.input_size)),
           "<br/>".join(map(str, s.output_size)),
           to_human_format(s.output_memory_cost), to_human_format(s.output_percent * 100),
           to_human_format(s.n_param), to_human_format(s.param_memory_cost),
           to_human_format(s.param_percent * 100),
           to_human_format(s.FLOPs), to_human_format(s.FLOPs_percent * 100),
           to_human_format(s.memory_access), to_human_format(s.memory_access_percent * 00),
           to_human_format(s.computing_density))

        html += '''    <tr>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>

                <td>{}</td>
                <td>{}</td>
                <td>{}</td>

                <td>{}</td>
                <td>{}</td>
                <td>{}</td>

                <td>{}</td>
                <td>{}</td>

                <td>{}</td>
                <td>{}</td>

                <td>{}</td>

            </tr>
        '''.format("total", "", "", "",
                   "",
                   to_human_format(self.total.output_memory_cost), "",
                   to_human_format(self.total.n_param), to_human_format(self.total.param_memory_cost), "",
                   to_human_format(self.total.FLOPs), "",
                   to_human_format(self.total.memory_access), "",
                   to_human_format(self.total.computing_density))

        html += '''</tbody>
</table>
<script src="https://cdnjs.cloudflare.com/ajax/libs/semantic-ui/2.4.1/semantic.min.js"></script>
</body>
</html>
'''
        if path is None:
            return html
        else:
            with open(path, "w") as f:
                f.write(html)

    def to_csv(self, path=None):
        pass


def is_container(m: torch.nn.Module):
    return issubclass(m.__class__, torch.nn.Container)


class SummaryTester(object):
    def __init__(self, input_size, device="cpu"):
        self.input_size = list(input_size)
        self.batch_size, *self.feature_size = self.input_size
        self.trying_input_size = [1] + self.feature_size
        self.device = device

    def __call__(self, module: torch.nn.Module, name="net"):
        summary = Summary(name)
        hooks = []
        x = torch.empty(self.trying_input_size, device=self.device)

        for i, (name, m) in enumerate(filter(
                lambda item: not is_container(item[1]) and not item[1] == module,
                leaf_named_modules(module))):
            layer_summary = LayerSummary()
            layer_summary.id = i
            layer_summary.unique_name = name
            hooks.append(m.register_forward_hook(
                functools.partial(summary_hook, batch_size=self.batch_size, summary=layer_summary)))
            summary.layers.append(layer_summary)

        module(x)

        for hook in hooks:
            hook.remove()

        summary.gather()

        return summary
