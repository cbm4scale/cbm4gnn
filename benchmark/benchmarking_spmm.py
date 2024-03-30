# This script is based on the original script from torch_scatter library by rusty1s:
# https://github.com/rusty1s/pytorch_scatter/blob/c095c62e4334fcd05e4ac3c4bb09d285960d6be6/benchmark/scatter_segment.py

import time
import os.path as osp
import itertools

import argparse
import wget
import torch
from scipy.io import loadmat
from torch import zeros

from torch_scatter import scatter, segment_coo, segment_csr

from benchmark.mkl_csr_spmm import mkl_single_csr_spmm

short_rows = [
    ('DIMACS10', 'citationCiteseer'),
    ('SNAP', 'web-Stanford'),
]
long_rows = [
    ('Janna', 'StocF-1465'),
    ('GHS_psdef', 'ldoor'),
]


def download(dataset):
    url = 'https://sparse.tamu.edu/mat/{}/{}.mat'
    group, name = dataset
    if not osp.exists(f'{name}.mat'):
        print(f'Downloading {group}/{name}:')
        wget.download(url.format(group, name))
        print('')


def underline(text, flag=True):
    return f'\033[4m{text}\033[0m' if flag else text


def bold(text, flag=True):
    return f'\033[1m{text}\033[0m' if flag else text

@torch.no_grad()
def correctness(dataset):
    group, name = dataset
    mat = loadmat(f'{name}.mat')['Problem'][0][0][2].tocsr()
    rowptr = torch.from_numpy(mat.indptr).to(args.device, torch.int64)
    row = torch.from_numpy(mat.tocoo().row).to(args.device, torch.int64)
    col = torch.from_numpy(mat.tocoo().col).to(args.device, torch.int64)
    edge_index = torch.stack([row, col])
    values = torch.ones(row.size(0), dtype=torch.float32, device=args.device)
    a = torch.sparse_coo_tensor(edge_index.to(torch.int32), values, mat.shape).to_sparse_csr()

    dim_size = rowptr.size(0) - 1

    for size in sizes:
        x = torch.randn((mat.shape[0], size), device=args.device)
        x = x.squeeze(-1) if size == 1 else x

        x_j = x.gather(0, col.view(-1, 1).expand(-1, x.size(1)))

        out0 = a @ x
        out1 = scatter(x_j, row, dim=0, dim_size=dim_size, reduce='add')
        out2 = segment_coo(x_j, row, dim_size=dim_size, reduce='add')
        out3 = segment_csr(x_j, rowptr, reduce='add')

        out4 = x.new_zeros(dim_size, *x.size()[1:])
        row_tmp = row.view(-1, 1).expand_as(x_j) if x.dim() > 1 else row
        out4.scatter_reduce_(0, row_tmp, x_j, 'sum', include_self=False)
        out5 = torch.empty(dim_size, *x.size()[1:], dtype=x.dtype, device=x.device)
        mkl_single_csr_spmm(edge_index.to(torch.int32), values, x, out5)

        torch.testing.assert_close(out0, out1, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out1, out2, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out1, out3, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out1, out4, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out1, out5, atol=1e-2, rtol=1e-2)


def time_func(func, x):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()

    if not args.with_backward:
        with torch.no_grad():
            for _ in range(iters):
                func(x)
    else:
        x = x.requires_grad_()
        for _ in range(iters):
            out = func(x)
            out = out[0] if isinstance(out, tuple) else out
            torch.autograd.grad(out, x, out, only_inputs=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t


def timing(dataset):
    group, name = dataset
    mat = loadmat(f'{name}.mat')['Problem'][0][0][2].tocsr()
    rowptr = torch.from_numpy(mat.indptr).to(args.device, torch.long)
    row = torch.from_numpy(mat.tocoo().row).to(args.device, torch.long)
    col = torch.from_numpy(mat.tocoo().col).to(args.device, torch.long)
    edge_index = torch.stack([row, col])
    values = torch.ones(row.size(0), dtype=torch.float32, device=args.device)
    a = torch.sparse_coo_tensor(edge_index.to(torch.int32), values, mat.shape).to_sparse_csr()
    dim_size = rowptr.size(0) - 1
    avg_row_len = row.size(0) / dim_size

    def torch_spmm(x):
        return a @ x

    def torch_scatter_row(x):
        x_j = x.gather(0, col.view(-1, 1).expand(-1, x.size(1)))
        row_tmp = row.view(-1, 1).expand_as(x_j) if x.dim() > 1 else row
        return zeros(dim_size, *x.size()[1:], dtype=x.dtype, device=args.device).scatter_add_(0, row_tmp, x_j)

    def pyg_scatter_row(x):
        x_j = x.gather(0, col.view(-1, 1).expand(-1, x.size(1)))
        return scatter(x_j, row, dim=0, dim_size=dim_size, reduce='sum')

    def pyg_segment_coo(x):
        x_j = x.gather(0, col.view(-1, 1).expand(-1, x.size(1)))
        return segment_coo(x_j, row, reduce='sum')

    def pyg_segment_csr(x):
        x_j = x.gather(0, col.view(-1, 1).expand(-1, x.size(1)))
        return segment_csr(x_j, rowptr, reduce='sum')

    def mkl_spmm(x):
        out5 = torch.empty(dim_size, *x.size()[1:], dtype=x.dtype, device=x.device)
        return mkl_single_csr_spmm(edge_index.to(torch.int32), values, x, out5)


    t0, t1, t2, t3, t4, t5 = [], [], [], [], [], []

    for size in sizes:
        x = torch.randn((mat.shape[0], size), device=args.device)
        x = x.squeeze(-1) if size == 1 else x

        t0 += [time_func(torch_spmm, x)]
        t1 += [time_func(torch_scatter_row, x)]
        t2 += [time_func(pyg_scatter_row, x)]
        t3 += [time_func(pyg_segment_coo, x)]
        t4 += [time_func(pyg_segment_csr, x)]
        t5 += [time_func(mkl_spmm, x)]

        del x

    del row, rowptr, mat, torch_scatter_row, pyg_scatter_row, pyg_segment_coo, pyg_segment_csr, values, edge_index
    torch.cuda.empty_cache() if torch.cuda.is_available() and args.device == 'cuda' else None

    ts = torch.tensor([t0, t1, t2, t3, t4, t5])
    winner = torch.zeros_like(ts, dtype=torch.bool)
    winner[ts.argmin(dim=0), torch.arange(len(sizes))] = 1
    winner = winner.tolist()

    name = f'{group}/{name}'
    print(f'{bold(name)} (avg row length: {avg_row_len:.2f}):')
    print('\t'.join(["                   "] + [f'{size:>5}' for size in sizes]))
    print('\t'.join([bold("torch.spmm        ")] + [underline(f'{t:.5f}', f) for t, f in zip(t0, winner[0])]))
    print('\t'.join([bold("torch.scatter_add  ")] + [underline(f'{t:.5f}', f) for t, f in zip(t1, winner[1])]))
    print('\t'.join([bold("pyg_scatter        ")] + [underline(f'{t:.5f}', f) for t, f in zip(t2, winner[2])]))
    print('\t'.join([bold("pyg_segment (COO)  ")] + [underline(f'{t:.5f}', f) for t, f in zip(t3, winner[3])]))
    print('\t'.join([bold("pyg_segment (CSR)  ")] + [underline(f'{t:.5f}', f) for t, f in zip(t4, winner[4])]))
    print('\t'.join([bold("mkl_spmm          ")] + [underline(f'{t:.5f}', f) for t, f in zip(t5, winner[5])]))
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_backward', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    iters = 10
    sizes = [50, 100, 500, 1000, 1500, 2000]

    for _ in range(10):  # Warmup.
        torch.randn(100, 100, device=args.device).sum()
    for dataset in itertools.chain(short_rows, long_rows):
        download(dataset)
        correctness(dataset)
        timing(dataset)