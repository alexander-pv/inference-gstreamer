
FROM nvcr.io/nvidia/deepstream:5.0-20.07-devel

RUN apt-get update && \
      apt-get -y install sudo

RUN useradd -s /bin/bash -d /home/ubuntu/ -m -G sudo ubuntu
RUN passwd -d ubuntu
USER ubuntu

RUN sudo apt-get install -y --no-install-recommends \
        wget\
        tmux \
        ffmpeg\
        tar \
        vim  \
        git \
        libpython3-dev \
        libxml2-dev \
        libxslt1-dev \
        python3-pip \
        python3-dev \
        python3-qpid-proton\
        python3-gi \
        python3-opencv \
        net-tools \
        linux-libc-dev \
        libglew2.0 \
        libssl1.0.0 \
        libjpeg8 \
        libjson-glib-1.0-0 \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer-plugins-bad1.0-dev  \
        gstreamer1.0-plugins-base  \
        gstreamer1.0-plugins-good  \
        gstreamer1.0-plugins-bad  \
        gstreamer1.0-plugins-ugly  \
        gstreamer1.0-libav  \
        gstreamer1.0-tools  \
        gstreamer1.0-x  \
        gstreamer1.0-alsa  \
        gstreamer1.0-gl  \
        gstreamer1.0-gtk3  \
        gstreamer1.0-qt5  \
        gstreamer1.0-pulseaudio \
        libcurl4 \
        libcurl3-gnutls \
        libuuid1 \
        libjansson4 \
        libjansson-dev \
        librabbitmq4 \
        libgles2-mesa \
        libgstrtspserver-1.0-0 \
        libv4l-dev \
        gdb \
        bash-completion \
        libboost-dev \
        uuid-dev \
        libgstrtspserver-1.0-0 \
        libgstrtspserver-1.0-0-dbg \
        libgstrtspserver-1.0-dev \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libglew-dev \
        libssl-dev \
        libopencv-dev \
        freeglut3-dev \
        libjpeg-dev \
        libcurl4-gnutls-dev \
        libjson-glib-dev \
        libboost-dev \
        librabbitmq-dev \
        libgles2-mesa-dev \
        libgtk-3-dev \
        libgdk3.0-cil-dev \
        pkg-config \
        cmake \
        protobuf-compiler \
        libprotobuf-dev \
        python-pybind11 \
        libxau-dev \
        libxdmcp-dev \
        libxcb1-dev \
        libxext-dev \
        libx11-dev \
        rsyslog \
        gstreamer1.0-rtsp \
        libcudnn7\
        libcudnn7-dev \
        cuda-cudart-10-1 \
    && sudo rm -rf /usr/share/doc/* /usr/share/man/* /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && sudo apt autoremove

RUN cd /opt/nvidia/deepstream/deepstream/lib && sudo -H python3 setup.py install

RUN python3 -m pip install pandas

WORKDIR /home/ubuntu
