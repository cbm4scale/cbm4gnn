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