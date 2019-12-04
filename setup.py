import io
import os
import re
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")


VERSION = find_version('flame', '__init__.py')


def find_requirements(file_path):
    with open(file_path) as f:
        return f.read().splitlines()


requirements = [
    'pyhocon',
    'prettytable',
    'torch>=0.4.0',
]

setup(
    name="torch-flame",
    version=VERSION,
    description="A deep learning utility library for PyTorch",
    url="https://github.com/chenyaofo/flame",
    author="chenyaofo",
    author_email="chenyaofo@gmail.com",
    packages=find_packages(exclude=['test']),
    package_data={'': ['requirements.txt']},
    install_requires=requirements,
)
