# -------------------------- Build-time arguments (defaults) --------------------------
# Build arguments
ARG ROS2_DISTRIBUTION="humble"
# PyTorch
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu129
# Parallel build cores for colcon / make
ARG BUILD_CORES=4

# By default, use ROS 2 Humble as the base image
FROM osrf/ros:${ROS2_DISTRIBUTION}-desktop
ARG ROS2_DISTRIBUTION
ARG TORCH_INDEX_URL
ARG BUILD_CORES

WORKDIR /dinov3_ros
SHELL ["/bin/bash", "-c"]
COPY . /dinov3_ros

# # Install dependencies
RUN apt-get update &&  apt-get install -y \
    libboost-all-dev \
    build-essential \
    python3-pip

# Install subpackage with the models and inference files (except pytorch)
RUN pip install -e . 

# Install your pytorch version 
RUN pip3 install torch torchvision --index-url ${TORCH_INDEX_URL}

# Install ROS 2 dependencies through rosdep
RUN source /opt/ros/${ROS2_DISTRIBUTION}/setup.bash && \
    if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then \
        rosdep init; \
    fi && \
    apt-get update && \
    rosdep update -y && \
    rosdep install --from-paths . --ignore-src -r -y

# ---------------------- Packages build ------------------------------------
RUN export MAKEFLAGS="-j {BUILD_CORES}" && \
    echo "source /opt/ros/${ROS2_DISTRIBUTION}/setup.bash" >> ~/.bashrc && \
    echo "source /dinov3_ros/ros2_ws/install/setup.bash" >> ~/.bashrc && \
    source /opt/ros/${ROS2_DISTRIBUTION}/setup.bash &&\
    cd ros2_ws &&\
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    . install/setup.bash


