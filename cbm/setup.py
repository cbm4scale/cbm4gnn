# To build the extension, follow the instructions below:
# Set the MKLROOT environment variables to the path of your MKL installation.
# This can be done by running `source /path/to/setvars.sh` in your terminal.
# Run `python setup.py build_ext --inplace` in the `cbm` directory.

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# Assuming MKLROOT is set in your environment variables
mkl_root = os.getenv('MKLROOT')  # Replace '/path/to/your/mkl' with a default path if MKLROOT is not set


# wrong result with ILP64. Why is this happening?
extra_compile_args = [
    #'-DMKL_ILP64',
    '-m64',
    '-I"{}/include"'.format(mkl_root),
    '-fopenmp',
    '-march=native',
    '-O2',
    '-std=c++20',
    f'-I{os.path.dirname(os.path.abspath(__file__))}/../arbok/source/arbok/include',
    #--------
    ##'-DMKL_ILP64',  # Use ILP64 integer size
    #'-m64',  # Target 64-bit architecture
    #'-I{}/include'.format(mkl_root),  # Include MKL header files
    #'-O3',
    #'-march=native',
    #'-std=c++17',
    #'-fopenmp', #fopenmp must be on top!

]

extra_link_args = [
    '-m64',  
    '-L{}/lib'.format(mkl_root),
    '-Wl,--no-as-needed', 
    '-lmkl_intel_ilp64', 
    '-lmkl_gnu_thread', 
    '-lmkl_core',
    '-lgomp', 
    '-lpthread',
    '-lm',
    '-ldl',
    f'-L{os.path.dirname(os.path.abspath(__file__))}/../arbok/build/',
    '-larbok',
    #----
    #'-m64',
    #'-Wl,--start-group',
    #'{}/lib/libmkl_intel_ilp64.a'.format(mkl_root),  # Static linking MKL ILP64 interface
    #'{}/lib/libmkl_sequential.a'.format(mkl_root),  # Static linking MKL sequential library
    #'{}/lib/libmkl_core.a'.format(mkl_root),  # Static linking MKL core library
    #'-Wl,--end-group',
    #'-lpthread',  # Link pthread for parallelism in MKL
    #'-lm',  # Link the math library
    #'-ldl',  # Link the dynamic loading library
    

]

install_requires = [
    "torch>=2.0.0",
]

setup(
    name='cbm_mkl_cpp',
    packages=find_packages(),
    install_requires=install_requires,
    version="0.1",
    ext_modules=[
        CppExtension(
            'cbm_mkl_cpp',
            ['cbm_extensions.cpp'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
