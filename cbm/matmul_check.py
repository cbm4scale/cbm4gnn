import time
import torch
import argparse

import scipy.sparse as sp
from scipy.io import loadmat

import cbm_mkl_cpp as cbm_
from cbm import cbm_matrix

def seq_mkl_csr_spmm(a, x, y):
        row_ptr_s = a.crow_indices()[:-1].to(torch.int32)
        row_ptr_e = a.crow_indices()[1: ].to(torch.int32)
        col_ptr = a.col_indices().to(torch.int32)
        val_ptr = a.values().to(torch.float32)

        cbm_.seq_s_spmm_csr_int32(
            row_ptr_s, row_ptr_e, col_ptr, val_ptr, x, y)

def omp_mkl_csr_spmm(a, x, y):
        row_ptr_s = a.crow_indices()[:-1].to(torch.int32)
        row_ptr_e = a.crow_indices()[1: ].to(torch.int32)
        col_ptr = a.col_indices().to(torch.int32)
        val_ptr = a.values().to(torch.float32)

        cbm_.omp_s_spmm_csr_int32(
            row_ptr_s, row_ptr_e, col_ptr, val_ptr, x, y)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequential", action="store_true")

    args = parser.parse_args()
    if (args.sequential) : 
        torch.set_num_threads(1)

    # load dataset for test
    group, name = ('SNAP', 'ca-HepPh')
    mat = loadmat(f'{name}.mat')['Problem'][0][0][2].tocsr()

    # use edge_index and values to create CSR and CBM format
    row = torch.from_numpy(mat.tocoo().row).to(torch.int32)#(args.device, torch.int64)
    col = torch.from_numpy(mat.tocoo().col).to(torch.int32)#(args.device, torch.int64)
    edge_index = torch.stack([row, col])
    values = torch.ones(row.size(0), dtype=torch.float32) #,device=args.device)

    # init csr format
    a = torch.sparse_coo_tensor(edge_index.to(torch.int32), values, mat.shape).to_sparse_csr()

    # init cbm format
    c = cbm_matrix(edge_index, values, rows=mat.shape[0], cols=mat.shape[1], alpha=2)

    # init embedding
    x = torch.randn((mat.shape[1], 512), dtype=torch.float32)

    if (args.sequential):
        print("Sequential execution...")

        # validate torch matmul
        t1 = time.time()
    
        with torch.no_grad():
            for _ in range(10):
                y1 = a @ x
                
        t2 = time.time()
        print("seq-torch-csr: ", t2-t1, " seconds")
 
        t1 = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                y2 = c.seq_torch_csr_matmul(x)
    
        t2 = time.time()
        print("seq-torch-cbm: ", t2-t1, " seconds")

        # validate results
        torch.testing.assert_close(y1, y2, atol=1e-4, rtol=1e-4)
        print("seq-torch-matmul: PASSED")


        # validate mkl matmul

        t1 = time.time()

        with torch.no_grad():
            y3 = torch.empty((mat.shape[0], x.shape[1]), dtype=torch.float32)

            for _ in range(10):
                seq_mkl_csr_spmm(a,x,y3)

        t2 = time.time()
        print("seq-mkl-csr: ", t2-t1, " seconds")

        t1 = time.time()


        with torch.no_grad():
            y4 = torch.empty((mat.shape[0], x.shape[1]), dtype=torch.float32)

            for _ in range(10):
                c.seq_mkl_csr_spmm_update(x,y4)

        t2 = time.time()
        print("seq-mkl-cbm: ", t2-t1, " seconds")

        # validate results
        torch.testing.assert_close(y3, y4, atol=1e-4, rtol=1e-4)
        print("seq-mkl-matmul: PASSED")

    else:
        print("Parallel execution...")

        # validate torch matmul
        t1 = time.time()

        with torch.no_grad():
            for _ in range(10):
                y1 = a @ x

        t2 = time.time()
        print("par-torch-csr: ", t2-t1, " seconds")
 
        t1 = time.time()

        with torch.no_grad():
            for _ in range(10):
                y2 = c.omp_torch_csr_matmul(x)

        t2 = time.time()
        print("par-torch-cbm: ", t2-t1, " seconds")

        # validate results
        torch.testing.assert_close(y1, y2, atol=1e-4, rtol=1e-4)
        print("par-torch-matmul: PASSED")


        # validate mkl matmul
        t1 = time.time()

        with torch.no_grad():
            y3 = torch.empty((mat.shape[0], x.shape[1]), dtype=torch.float32)

            for _ in range(10):
                omp_mkl_csr_spmm(a,x,y3)

        t2 = time.time()
        print("par-mkl-csr: ", t2-t1, " seconds")

        t1 = time.time()

        with torch.no_grad():
            y4 = torch.empty((mat.shape[0], x.shape[1]), dtype=torch.float32)

            for _ in range(10):
                c.omp_mkl_csr_spmm_update(x,y4)

        t2 = time.time()
        print("par-mkl-cbm: ", t2-t1, " seconds")

        # validate results
        torch.testing.assert_close(y3, y4, atol=1e-4, rtol=1e-4)
        print("par-mkl-matmul: PASSED")

        torch.testing.assert_close(y1, y2, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(y1, y3, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(y1, y4, atol=1e-4, rtol=1e-4)
        print("all tests PASSED!")
