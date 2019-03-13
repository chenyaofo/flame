import pathlib
import hashlib

import torch


class FileHasher(object):
    def __init__(self, algorithm="md5"):
        if not hasattr(hashlib, algorithm):
            raise Exception("Not support such algorithm().".format(algorithm))
        self.algorithm = getattr(hashlib, algorithm)

    def __call__(self, filename):
        hasher = self.algorithm()
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


# follow the statement on
# https://pytorch.org/docs/stable/model_zoo.html#torch.utils.model_zoo.load_url
def save_model_with_hash_suffix(state_dict, path, hash_algorithm="sha256", suffix_length=8):
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
    sha1_hash = FileHasher(hash_algorithm)
    signature = sha1_hash(tmp_pt)[:suffix_length]
    purename, extension = dst.stem, dst.suffix
    dst = dst.with_name("{}-{}".format(purename, signature)).with_suffix(extension)
    tmp_pt.rename(dst)


def replace_layer_by_unique_name(module, unique_name, layer):
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


def get_layer_by_unique_name(module, unique_name):
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