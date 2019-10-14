import os
import random
import glob
import pathlib
import hashlib
import subprocess
import zipfile

import typing

import numpy
import torch
import torch.nn as nn
import flame
import pyhocon

from flame._utils import create_code_snapshot as create_code_snapshot_impl


def compute_file_hash(file, hash_algorithm="sha256"):
    hasher = getattr(hashlib, hash_algorithm)
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_model_with_hash_suffix(state_dict, path: str, hash_algorithm="sha256", suffix_length=8) -> None:
    '''Save the state_dict with hash suffix.

    Example:
        >> save_model_with_hash_suffix(state_dict,"resnet50.pth")
        # the saved filename is like 'resnet50-19c8e357.pth'

    :param state_dict: The state_dict of torch.nn.Module, it should be on device('cpu') for distribution.
    :param path: The store path of serialization file, such as /tmp/resnet.pth .
    :param hash_algorithm: All available hash algorithms can be seen using hashlib.algorithms_available,
    following the rule of pytorch(https://pytorch.org/docs/stable/model_zoo.html#torch.utils.model_zoo.load_url),
    it should be sha256 to be compatible withtorch.utils.model_zoo.load_url. Default: 'sha256'.
    :param suffix_length: The length of hash string of filename, following the style of pytorch(
    https://pytorch.org/docs/stable/model_zoo.html#torch.utils.model_zoo.load_url), it should be 8. Default: 8.
    :return: None
    '''
    dst = pathlib.Path(path)
    tmp_pt = dst.with_name("temporary").with_suffix(".pth")
    torch.save(state_dict, tmp_pt)
    signature = compute_file_hash(tmp_pt, hash_algorithm)[:suffix_length]
    purename, extension = dst.stem, dst.suffix
    dst = dst.with_name("{}-{}".format(purename, signature)).with_suffix(extension)
    tmp_pt.rename(dst)


def replace_layer_by_unique_name(module: nn.Module, unique_name: str, layer: nn.Module) -> None:
    if unique_name == "":
        return
    unique_names = unique_name.split(".")
    if len(unique_names) == 1:
        module._modules[unique_names[0]] = layer
    else:
        replace_layer_by_unique_name(
            module._modules[unique_names[0]],
            ".".join(unique_names[1:]),
            layer
        )


def get_layer_by_unique_name(module: nn.Module, unique_name: str) -> nn.Module:
    if unique_name == "":
        return module
    unique_names = unique_name.split(".")
    if len(unique_names) == 1:
        return module._modules[unique_names[0]]
    else:
        return get_layer_by_unique_name(
            module._modules[unique_names[0]],
            ".".join(unique_names[1:]),
        )


def create_code_snapshot(name: str = "code-snapshot",
                         include_suffix: typing.List[str] = (".py",),
                         source_directory: str = os.getcwd(),
                         store_directory: str = flame.output_directory,
                         hocon: pyhocon.ConfigTree = flame.hocon) -> None:
    create_code_snapshot_impl(name, include_suffix, source_directory, store_directory, hocon)


def get_last_commit_id():
    try:
        output = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8")
        output = output.strip()
        return output
    except subprocess.CalledProcessError as e:
        out_bytes = e.output
        code = e.returncode
        return None


def generate_random_from_system():
    return int.from_bytes(os.urandom(2), byteorder="little", signed=False)


def set_reproducible(seed=0):
    '''
    To ensure the reproducibility, refer to https://pytorch.org/docs/stable/notes/randomness.html.
    Note that completely reproducible results are not guaranteed.
    '''
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_cudnn_auto_tune():
    torch.backends.cudnn.benchmark = True


def compute_nparam(module: nn.Module, skip_pattern):
    return sum(map(lambda p: p.numel(), module.parameters()))
