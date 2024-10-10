import time
import argparse
from os.path import exists
import warnings
import wget
from scipy.io import loadmat
import torch

# intel mkl wrappers that were used in cbm4mm
from cbm import cbm_mkl_cpp as mkl

# import our class
from cbm.cbm4mm import cbm4mm
from cbm.cbm4gcn import cbm4gcn

# avoid annoying warnings...
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state", category=UserWarning)

def mkl_matmul(a, x, y):
        row_ptr_s = a.crow_indices()[:-1].to(torch.int32)
        row_ptr_e = a.crow_indices()[1: ].to(torch.int32)
        col_ptr = a.col_indices().to(torch.int32)
        val_ptr = a.values().to(torch.float32)

        mkl.s_spmm_csr_int32(
            row_ptr_s, row_ptr_e, col_ptr, val_ptr, x, y)


def download_texas_A_and_M_university(dataset):
    url = "https://sparse.tamu.edu/mat/{}/{}.mat"
    group, name = dataset
    if not exists(f"{name}.mat"):
        print(f"Downloading {group}/{name}:")
        wget.download(url.format(group, name))
        print("")

def read_texas_A_and_M_university(dataset):
    group, name = dataset
    return loadmat(f"{name}.mat")["Problem"][0][0][2].tocsr()

def matmul_example(cbm: cbm4mm, csr: torch.sparse_csr_tensor) -> None: 
    print("number of features: " + str(n_features))
    print("number of iterations: " + str(n_iterations))

    # init embedding
    x = torch.randn((csr.shape[1], n_features), dtype=torch.float32)
 
    # matmul with csr
    y1 = torch.empty((csr.shape[0], x.shape[1]), dtype=torch.float32)

    t1 = time.time()
    with torch.no_grad():

        for _ in range(n_iterations):
            mkl_matmul(csr, x, y1)

    t2 = time.time()
    print("mkl-csr: ", t2-t1, " seconds")

    # matmul with cbm
    y2 = torch.empty((csr.shape[0], x.shape[1]), dtype=torch.float32)

    t1 = time.time()
    with torch.no_grad():

        for _ in range(n_iterations):
            cbm.matmul(x,y2)

    t2 = time.time()
    print("mkl-cbm: ", t2-t1, " seconds")

    # compare results
    torch.testing.assert_close(y1, y2, atol=1e-4, rtol=1e-4)
    print("PASSED!")


# example code to measure delta_spmm and update time
def measure_delta_spmm_and_update(cbm: cbm4mm, n: int, m: int) -> None:
    print("number of features: " + str(n_features))
    print("number of iterations: " + str(n_iterations))

    # init embedding
    x = torch.randn((n, n_features), dtype=torch.float32)
 
    # init result matrix
    y = torch.empty((m, x.shape[1]), dtype=torch.float32)
    
    # measure deltas_spmm
    t1 = time.time()
    with torch.no_grad():

        for _ in range(n_iterations):
            cbm.deltas_spmm(x,y)

    t2 = time.time()
    print("delta_spmm: ", (t2 - t1) / n_iterations, " seconds")

    # measure deltas_spmm
    t1 = time.time()
    with torch.no_grad():

        for _ in range(n_iterations):
            cbm.update(y)

    t2 = time.time()
    print("update: ", (t2 - t1) / n_iterations, " seconds")


n_features = 0
n_iterations = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--features", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=50)

    args = parser.parse_args()
    n_features = args.features
    n_iterations = args.iterations

    # load dataset for test
    group, name = ('SNAP', 'ca-HepPh')
    download_texas_A_and_M_university((group, name))
    mat = read_texas_A_and_M_university((group, name))

    # use edge_index and values to create CSR and CBM format
    row = torch.from_numpy(mat.tocoo().row).to(torch.int32)#(args.device, torch.int64)
    col = torch.from_numpy(mat.tocoo().col).to(torch.int32)#(args.device, torch.int64)
    values = torch.ones(row.size(0), dtype=torch.float32) #,device=args.device)
    edge_index = torch.stack([row, col])
    

    # init csr format
    a = torch.sparse_coo_tensor(edge_index.to(torch.int32), values, mat.shape).to_sparse_csr()

    # init cbm format
    c = cbm4mm(edge_index, values, alpha=16)

    matmul_example(c,a)
    measure_delta_spmm_and_update(c, a.shape[1], a.shape[0])



    
   
