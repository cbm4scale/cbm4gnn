# python setup.py build_ext --inplace
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# Assuming MKLROOT is set in your environment variables
mkl_root = os.getenv("MKLROOT")  # Replace "/path/to/your/mkl" with a default path if MKLROOT is not set


# wrong result with ILP64. Why is this happening?
extra_compile_args = [
    # "-DMKL_ILP64",  # Use ILP64 integer size
    "-m64",  # Target 64-bit architecture
    "-I{}/include".format(mkl_root),  # Include MKL header files
    "-O3",
    "-march=native",
]

extra_link_args = [
    "-m64",
    "-Wl,--start-group",
    "{}/lib/libmkl_intel_ilp64.a".format(mkl_root),  # Static linking MKL ILP64 interface
    "{}/lib/libmkl_gnu_thread.a".format(mkl_root),  # Static linking MKL sequential library
    "{}/lib/libmkl_core.a".format(mkl_root),  # Static linking MKL core library
    "-Wl,--end-group",
    "-lpthread",  # Link pthread for parallelism in MKL
    "-lm",  # Link the math library
    "-ldl",  # Link the dynamic loading library
]


setup(
    name="mkl_cpp",
    ext_modules=[
        CppExtension(
            "mkl_cpp",
            ["spmm_mkl.cpp"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)