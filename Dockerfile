# To build the image, run the following command:
# docker build -t cbm4gnn .
# To run the image, run the following command:
# docker run --gpus all --rm -ti --ipc=host --name cbm4gnn_instance cbm4gnn /bin/bash

# Base Image
ARG BASE_IMAGE=nvidia/cuda:12.1.0-devel-ubuntu22.04
FROM ${BASE_IMAGE} as base

# Set non-interactive shell
ARG DEBIAN_FRONTEND=noninteractive

# Set the working directory
LABEL com.nvidia.volumes.needed="nvidia_driver"

# Install a supported version of GCC before installing other packages
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get install -y gcc-11 g++-11 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
    && gcc --version && g++ --version

# Install common dependencies and utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
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
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
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
RUN bash -c "source /opt/intel/oneapi/setvars.sh"

# Configure ccache
RUN /usr/sbin/update-ccache-symlinks \
    && mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV CC /usr/bin/gcc
ENV CXX /usr/bin/g++

# Add a symbolic link for python3.11
RUN ln -s /usr/bin/python3.11 /usr/bin/python
# Clone PyTorch repository and checkout to version 2.2
RUN git clone https://github.com/pytorch/pytorch /opt/pytorch \
    && cd /opt/pytorch \
    && git checkout orig/release/2.2

# Update submodules and install dependencies
RUN cd /opt/pytorch \
    && git submodule update --init --recursive \
    && python3.11 -m pip install -r requirements.txt

# This to copy a recent version of the select_compute_arch.cmake file to fix the issue with the CUDA version
COPY ./select_compute_arch.cmake /opt/pytorch/cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake

# Preparing the PyTorch directory and initiating the build configuration
RUN cd /opt/pytorch \
    && python3.11 setup.py develop

# Set the pytorch version
RUN PYTORCH_VERSION=$(python3.11 -c "import torch; print(torch.__version__)") && \
    echo "PyTorch version is $PYTORCH_VERSION"

# Clone PyTorch-Scatter repository
RUN git clone https://github.com/rusty1s/pytorch_scatter.git /opt/pytorch_scatter

# Use the PyTorch version to set the version of torch_scatter dynamically
RUN cd /opt/pytorch_scatter && \
    python3.11 setup.py install develop

COPY ./ /workspace
WORKDIR /workspace
ENV PYTHONPATH /workspace:$PYTHONPATH

# Install the required Python packages
RUN python3.11 -m pip install -r requirements_dev.txt