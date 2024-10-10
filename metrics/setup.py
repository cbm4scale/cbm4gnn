# To build the extension, follow the instructions below:
# Set the MKLROOT environment variables to the path of your MKL installation.
# This can be done by running `source /path/to/setvars.sh` in your terminal.
# Run `python setup.py build_ext --inplace develop` in the `cbm` directory.

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# wrong result with ILP64. Why is this happening?
extra_compile_args = [
    '-march=native',
    '-O2',
    '-std=c++20',
    '-fopenmp',
]

install_requires = [
    "torch>=2.0.0",
]

setup(
    name='cbm_metrics_cpp',
    packages=find_packages(),
    install_requires=install_requires,
    version="0.1",
    ext_modules=[
        CppExtension(
            'cbm_metrics_cpp',
            ['cbm_metrics.cpp'],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
