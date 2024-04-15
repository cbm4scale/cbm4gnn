# To build the image, run the following command:
# docker build -t cbm4gnn .
# To run the image, run the following command:
# docker run --gpus all --rm -ti --ipc=host --name cbm4gnn_instance cbm4gnn  /bin/bash

# Base Image
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE} as base

# Set non-interactive shell
ARG DEBIAN_FRONTEND=noninteractive

# Install common dependencies and utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        gpg \
        sudo \
        gpg-agent \
        build-essential \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Intel oneAPI keys and repository
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update && apt-get install -y --no-install-recommends intel-oneapi-mkl intel-oneapi-mkl-devel \
    && rm -rf /var/lib/apt/lists/*

# Configure MKL for optimal performance
ENV MKL_THREADING_LAYER=GNU
ENV MKL_SERVICE_FORCE_INTEL=1

# Configure ccache
RUN /usr/sbin/update-ccache-symlinks \
    && mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache

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
    && /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake conda-build pyyaml numpy ipython \
    && /opt/conda/bin/conda clean -ya

# Set MKL environment variables before building PyTorch
RUN bash -c "source /opt/intel/oneapi/setvars.sh"

# Clone PyTorch and install dependencies
RUN git clone --recursive https://github.com/pytorch/pytorch /opt/pytorch \
    && cd /opt/pytorch \
    && git submodule sync \
    && git submodule update --init --recursive \
    && export _GLIBCXX_USE_CXX11_ABI=1 \
    && export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} \
    && /opt/conda/bin/pip install -r requirements.txt

# Build PyTorch with non-interactive cmake configurations
RUN cd /opt/pytorch \
    && /opt/conda/bin/python setup.py build --cmake-only \
    && mkdir -p build \
    && cd build \
    && cmake -DBUILD_PYTHON=True -DBUILD_TEST=True -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_INSTALL_PREFIX=/opt/pytorch/torch \
       -DCMAKE_PREFIX_PATH="/opt/conda/lib/python3.11/site-packages" \
       -DNUMPY_INCLUDE_DIR="/opt/conda/lib/python3.11/site-packages/numpy/core/include" \
       -DPYTHON_EXECUTABLE="/opt/conda/bin/python" \
       -DPYTHON_INCLUDE_DIR="/opt/conda/include/python3.11" \
       -DPYTHON_LIBRARY="/opt/conda/lib/libpython3.11.a" \
       -DTORCH_BUILD_VERSION="2.4.0a0+git03a05e7" -DUSE_NUMPY=True ..

RUN cd /opt/pytorch \
    && export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} \
    && /opt/conda/bin/python setup.py develop

# Final Image
FROM ${BASE_IMAGE}
LABEL com.nvidia.volumes.needed="nvidia_driver"
COPY --from=base /opt/conda /opt/conda
COPY --from=base /opt/intel /opt/intel
COPY --from=base /opt/pytorch /workspace

# Set MKL environment variables
RUN bash -c "source /opt/intel/oneapi/setvars.sh"

RUN /opt/conda/bin/pip install -r requirements_dev.txt

ENV PATH /opt/conda/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH

COPY ./ /workspace
WORKDIR /workspace
