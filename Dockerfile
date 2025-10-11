# -------------------------- Build-time arguments (defaults) --------------------------
ARG ROS2_DISTRIBUTION=humble

# CUDA 12.9 family
ARG CUDA_MAJOR_MINOR=12.9
ARG CUDA_TOOLKIT_PKG=cuda-toolkit-12-9
ARG CUDA_KEYRING_VER=1.1-1

# TensorRT 10.13.3 (for CUDA 12.9)
ARG TENSORRT_VERSION=10.13.3
# Tag used in NVIDIA repo path (usually "<TENSORRT_VERSION>-cuda<CUDA_MAJOR_MINOR>")
ARG TENSORRT_DEB_TAG=10.13.3-cuda12.9
# Exact apt package version suffix (from NVIDIA release notes / repo metadata)
# Example: 10.13.3.9-1+cuda12.9
ARG TENSORRT_APT_PKG_VER=10.13.3.9-1+cuda12.9

# PyTorch
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu129

# cuda-python wheel version
ARG CUDA_PYTHON_VERSION=12.9.2

# Parallel build cores for colcon / make
ARG BUILD_CORES=4

# By default, use ROS 2 Humble as the base image
FROM osrf/ros:${ROS2_DISTRIBUTION}-desktop

ARG ROS2_DISTRIBUTION
ARG CUDA_MAJOR_MINOR
ARG CUDA_TOOLKIT_PKG
ARG CUDA_KEYRING_VER
ARG TENSORRT_VERSION
ARG TENSORRT_DEB_TAG
ARG TENSORRT_APT_PKG_VER
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCH_INDEX_URL
ARG CUDA_PYTHON_VERSION
ARG BUILD_CORES

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /dinov3_ros

# Copy repo
COPY . /dinov3_ros

# Core build deps + common tooling (one apt layer), then clean
RUN apt-get update && apt-get install -y --no-install-recommends \
      libboost-all-dev \
      build-essential \
      python3-pip \
      python3-venv \
      ca-certificates \
      gnupg \
      wget \
    && rm -rf /var/lib/apt/lists/*

# -------------------------- Python deps (package + torch) --------------------------
# Install your package (editable) excluding heavy torch inside; then torch/vision via index
RUN pip3 install --no-cache-dir -e .
RUN pip3 install --no-cache-dir \
      torch torchvision --index-url ${TORCH_INDEX_URL}

# Install ROS 2 dependencies through rosdep
RUN source /opt/ros/${ROS2_DISTRIBUTION}/setup.bash && \
    if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then \
        rosdep init; \
    fi && \
    apt-get update && \
    rosdep update -y && \
    rosdep install --from-paths . --ignore-src -r -y

# ---------------------- Packages build ------------------------------------
RUN export MAKEFLAGS="-j ${BUILD_CORES}" && \
    echo "source /opt/ros/${ROS2_DISTRIBUTION}/setup.bash" >> ~/.bashrc && \
    echo "source /dinov3_ros/ros2_ws/install/setup.bash" >> ~/.bashrc && \
    source /opt/ros/${ROS2_DISTRIBUTION}/setup.bash &&\
    cd ros2_ws &&\
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    . install/setup.bash

ARG CUDA_TOOLKIT_PKG=cuda-toolkit-12-9

# -------------------------- CUDA Toolkit --------------------------
# Install CUDA keyring + toolkit tied to the chosen major.minor
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends wget gnupg ca-certificates; \
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_${CUDA_KEYRING_VER}_all.deb; \
    dpkg -i cuda-keyring_${CUDA_KEYRING_VER}_all.deb; \
    rm -f cuda-keyring_${CUDA_KEYRING_VER}_all.deb; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ${CUDA_TOOLKIT_PKG}; \
    rm -rf /var/lib/apt/lists/*

# Make CUDA tools/libs easy to find
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install NVIDIA TensorRT local repository package
RUN export OS_TAG=ubuntu2204; \
    wget -q https://developer.download.nvidia.com/compute/tensorrt/${TENSORRT_VERSION}/local_installers/nv-tensorrt-local-repo-${OS_TAG}-${TENSORRT_DEB_TAG}_1.0-1_amd64.deb; \
    dpkg -i nv-tensorrt-local-repo-${OS_TAG}-${TENSORRT_DEB_TAG}_1.0-1_amd64.deb; \
    cp /var/nv-tensorrt-local-repo-${OS_TAG}-${TENSORRT_DEB_TAG}/*-keyring.gpg /usr/share/keyrings/ || true; \
    apt-get update; \
    rm -f nv-tensorrt-local-repo-${OS_TAG}-${TENSORRT_DEB_TAG}_1.0-1_amd64.deb

# Install the full TensorRT runtime
RUN apt-get install -y \
    {,python3-}libnvinfer*"=${TENSORRT_APT_PKG_VER}" \
    libnvonnxparsers*"=${TENSORRT_APT_PKG_VER}" \
    tensorrt*"=${TENSORRT_APT_PKG_VER}" \
  && rm -rf /var/lib/apt/lists/*

# cuda-python (driver/runtime bindings) pinned to matching CUDA
RUN python3 -m pip install --no-cache-dir cuda-python==${CUDA_PYTHON_VERSION}

