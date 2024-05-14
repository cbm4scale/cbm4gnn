# cbm4gnn
Repository for accelerating unweighted graph neural networks via the compressed binary matrix format. 

## Installation
```bash
git clone https://github.com/cbm4scale/cbm4gnn.git --recursive 
cd cbm4gnn
git submodule init
git submodule update
python setup.py
export LD_LIBRARY_PATH=./arbok/build/:$LD_LIBRARY_PATH
export PYTHONPATH=./:$PYTHONPATH
```

# Usage
```bash
# Parallel CPU benchmark
python OMP_NUM_THREADS=16 GOMP_CPU_AFFINITY="0-15" python benchmark/benchmark_spmm_cpu_parallel.py  
pyhton OMP_NUM_THREADS=16 GOMP_CPU_AFFINITY="0-15" python benchmark/benchmark_gcn_cpu_parallel.py

# Sequential CPU benchmark
python benchmark/benchmark_spmm_cpu_sequential.py
python benchmark/benchmark_gcn_cpu_sequential.py
```

# Search for the best CBM alpha
```bash
OMP_NUM_THREADS=16 GOMP_CPU_AFFINITY="0-15" python alpha_searcher/spmm_cpu_parallel.py
python alpha_searcher/spmm_cpu_sequential.py 
```