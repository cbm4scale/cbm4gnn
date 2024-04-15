# To build the image, run the following command:
# docker build -t cbm4gnn .
# To run the image, run the following command:
# docker run --gpus all --rm -ti --ipc=host --name cbm4gnn_instance cbm4gnn /bin/bash
# Verify the installation the installation of MKL by running the following commands:
# python -m timeit --setup="import torch; net = torch.nn.Linear(1000, 1000); batch = torch.rand(1000, 1000)" "net(batch)"
# python -m timeit --setup="import torch; from torch.utils import mkldnn as mkldnn_utils; net = torch.nn.Linear(1000, 1000); net = mkldnn_utils.to_mkldnn(net); batch = torch.rand(1000, 1000); batch = batch.to_mkldnn()" "net(batch)"
# The two commands should return similar results, which indicates that MKL is being used by PyTorch.

# Base Image
ARG BASE_IMAGE=nvidia/cuda:12.1.0-devel-ubuntu22.04
FROM ${BASE_IMAGE} as base

# Set non-interactive shell
ARG DEBIAN_FRONTEND=noninteractive

# Install common dependencies and utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ \
        ca-certificates \
        wget \
        sudo \
        build-essential \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Intel oneAPI keys and repository (if needed for your specific application)
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update && apt-get install -y --no-install-recommends intel-oneapi-mkl intel-oneapi-mkl-devel \
    && rm -rf /var/lib/apt/lists/*

# Configure MKL for optimal performance
ENV MKL_THREADING_LAYER=GNU
ENV MKL_SERVICE_FORCE_INTEL=1
ENV MKLROOT /opt/intel/oneapi/mkl/latest
ENV PATH /opt/intel/oneapi/bin:$PATH
ENV LD_LIBRARY_PATH /opt/intel/oneapi/mkl/latest/lib/intel64:$LD_LIBRARY_PATH

# Configure ccache
RUN /usr/sbin/update-ccache-symlinks \
    && mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV CC /usr/bin/gcc
ENV CXX /usr/bin/g++

# Setup Miniconda
ARG PYTHON_VERSION=3.11
ARG TARGETPLATFORM
ARG MINICONDA_ARCH=x86_64
RUN case ${TARGETPLATFORM} in \
         "linux/arm64")  MINICONDA_ARCH=aarch64  ;; \
    esac \
    && curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${MINICONDA_ARCH}.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake conda-build pyyaml numpy \
    && /opt/conda/bin/conda clean -ya \
    && /opt/conda/bin/conda install -c conda-forge gcc=12.1.0 --yes
# Create a symbolic link for python3 and set environment variables
RUN apt-get update && apt-get install -y g++
RUN ln -s /opt/conda/bin/python /usr/bin/python3
ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /opt/conda/lib:$LD_LIBRARY_PATH
ENV PATH /usr/local/cuda/bin:/usr/bin:$PATH

# Final Image
FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg-dev \
        libpng-dev \
        libgomp1 \
        build-essential \
        g++ \
    && rm -rf /var/lib/apt/lists/*
LABEL com.nvidia.volumes.needed="nvidia_driver"
COPY --from=base /opt/conda /opt/conda
COPY --from=base /opt/intel /opt/intel

# Set MKL environment variables
RUN bash -c "source /opt/intel/oneapi/setvars.sh"

ENV CC /usr/bin/gcc
ENV CXX /usr/bin/g++
ENV CUDA_HOME /usr/local/cuda
ENV PATH /opt/conda/bin:/opt/intel/oneapi/bin:${CUDA_HOME}/bin:$PATH
ENV LD_LIBRARY_PATH /opt/intel/oneapi/mkl/latest/lib/intel64:${CUDA_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV MKLROOT /opt/intel/oneapi/mkl/latest


COPY ./ /workspace
WORKDIR /workspace
# Install additional Python packages
RUN /opt/conda/bin/pip install -r requirements_dev.txt
ENV PYTHONPATH /workspace:$PYTHONPATH
