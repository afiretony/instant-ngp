# If you don't build this image, you can start from the base image and just run the commands manually:
#  $ docker run --runtime=nvidia --name=my-instant-ngp -it -d --net=host -v`pwd`:/opt/instant-ngp:z -w /opt/instant-ngp -v/:/outer_root pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel sleep infinity
# then $ docker exec -it my-instant-ngp bash
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

ENV PYTHONDONTWRITEBYTECODE 1

## General, instant-ngp doesn't use everything
RUN \
  apt-get update && \
  apt-get install -y python3 python3-pip unzip && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y libopencv-dev && \
  apt-get install -y \
    wget unzip git vim less sudo curl unzip


## CMake >=3.19 required for instant-ngp.  There's no package for base image so compile from source
RUN \
  apt-get update && apt-get install -y libssl-dev && \
  mkdir -p /opt/cmake && \
  cd /opt/cmake && \
  wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1.tar.gz && \
  tar xvfz cmake-3.22.1.tar.gz && \
  cd cmake-3.22.1 && \
  ./bootstrap && make -j && sudo make install && \
  cmake --version

## COLMAP
RUN \
  apt-get install -y \
    libatlas-base-dev libsuitesparse-dev build-essential libeigen3-dev libgflags-dev libgoogle-glog-dev
RUN \
  cd /opt && \
    (git clone https://github.com/ceres-solver/ceres-solver || true) && \
    cd ceres-solver && \
    git checkout 2.0.0 && \
    mkdir -p build && cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j && \
    make install

RUN \
  apt-get install -y \
    git \
    cmake \
    build-essential \
    libboost-all-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-qt5-dev \
    libcgal-dev

RUN \
  cd /opt && \
    (git clone https://github.com/colmap/colmap || true) && \
    cd colmap && \
    git checkout 3.6 && \
    mkdir -p build && cd build && \
    cmake .. && \
    make -j && \
    make install && \
    colmap -h

## instant-ngp
RUN apt-get install -y build-essential git \
             python3-dev python3-pip libopenexr-dev \
             libglfw3-dev libglew-dev libomp-dev  \
             libxinerama-dev libxcursor-dev libxi-dev

RUN apt-get install -y ffmpeg
RUN pip3 install imageio-ffmpeg imageio

# For whatever reason, the instant-ngp build can't find cuda w/out this stuff despite the canonical paths looking OK
ENV PATH /usr/local/cuda-11.3/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH

# Note!!! change this if you don't have 30x series GPU
ENV TCNN_CUDA_ARCHITECTURES 86

RUN \
  cd /opt && \
  git clone --recursive https://github.com/NVlabs/instant-ngp && \
  cd instant-ngp && \
  pip3 install -r requirements.txt && \
  cmake . -B build && \
  cmake --build build --config RelWithDebInfo -j --target all && \
  python3 scripts/run.py --help