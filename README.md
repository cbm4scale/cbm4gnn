# Compressed Binary Matrix for Graph Neural Networks (cbm4gnn)
This is the official repository for the paper "Accelerating Graph Neural Networks Using a Novel Computation-Friendly Matrix Compression Format".

## Installation
```bash
# Download the repository (due to the anonymous process, repository can be downloaded as zip file only).
cd cbm4gnn
git clone https://github.com/chistopher/arbok.git
python setup.py
export LD_LIBRARY_PATH=./arbok/build/:$LD_LIBRARY_PATH
export PYTHONPATH=./:$PYTHONPATH
```

## Getting Started
```bash
OMP_NUM_THREADS=16 GOMP_CPU_AFFINITY="0-15" python benchmark/cbm_DAD_creation.py
OMP_NUM_THREADS=16 GOMP_CPU_AFFINITY="0-15" python benchmark/alpha_searcher_spmm_cpu.py
OMP_NUM_THREADS=16 GOMP_CPU_AFFINITY="0-15" python benchmark/benchmark_gcn_cpu.py
```

## Project Directories
```
.
│-- benchmark
│   │-- alpha_searcher_AD_spmm_cpu.py
│   │-- alpha_searcher_DAD_spmm_cpu.py
│   │-- alpha_searcher_spmm_cpu.py
│   │-- benchmark_gcn_cpu.py
│   │-- cbm_DAD_creation.py
│   │-- cbm_creation.py
│   └-- utility.py
│-- cbm
│   │-- cbm4ad.py
│   │-- cbm4gcn.py
│   │-- cbm4mm.py
│   │-- cbm_extensions.cpp
│   │-- helpers.hpp
│   │-- setup.py
│   └-- utility.py
│-- gnns
│   │-- graph_convolutional_network
│   │   │-- __init__.py
│   │   │-- base_message_passing.py
│   │   │-- cbm_gcn.py
│   │   └-- mkl_gcn.py
│   └-- utility.py
│-- metrics
│   │-- cbm_metrics.cpp
│   │-- cbm_metrics.py
│   └-- setup.py
│-- LICENSE
│-- README.md
│-- datasets_statistics.py
│-- logger.py
│-- requirements_dev.txt
│-- setup.py
│-- tutorial_normalized_spmm.py
└-- tutorial_spmm.py
```

## Citation
To be updated after the acceptance of the paper.
