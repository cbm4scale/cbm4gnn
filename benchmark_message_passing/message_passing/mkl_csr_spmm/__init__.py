try:
    from .mkl_cpp import s_csr_spmm_ as mkl_single_csr_spmm, d_csr_spmm_ as mkl_double_csr_spmm
except ImportError:
    try:
        import os
        pwd = os.path.dirname(os.path.abspath(__file__))
        command = f"cd {pwd} && python setup.py build_ext --inplace"
        return_code = os.system(command)
        if return_code != 0:
            raise Exception("Error building mkl_cpp extension.")
    except:
        raise RuntimeError("Error building mkl_cpp extension. Please check the setup.py file in mkl_spmm folder.")
    from .mkl_cpp import s_csr_spmm_ as mkl_single_csr_spmm, d_csr_spmm_ as mkl_double_csr_spmm

all = [
    "mkl_single_csr_spmm",
    "mkl_double_csr_spmm",
]
