
from setuptools import setup
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


setup(
    name='cbm_mkl_cpp',
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
