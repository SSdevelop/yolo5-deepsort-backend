FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-11.8/
ENV TZ=Asia/Hong_Kong

# COPY . /app

WORKDIR /app

RUN apt-get update --fix-missing && apt-get upgrade -y
RUN apt install -y software-properties-common
RUN apt-get install -y build-essential
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.10 python3.10-distutils python3.10-dev python3-pip --fix-missing
RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.10 /usr/bin/python3
RUN apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev --fix-missing
RUN apt-get install -y ffmpeg libavfilter-dev libavutil-dev libsm6 libxext6 x264 libx264-dev --fix-missing 
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3
RUN apt-get install -y wget curl 

RUN pip install --upgrade pip setuptools wheel
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN python3 -m pip install --no-binary opencv-python opencv-python

COPY requirements.txt .
RUN pip install -r requirements.txt

# RUN cd model && \  
#     git clone https://github.com/IDEA-Research/GroundingDINO.git && cd GroundingDINO && \
#     git checkout 858efccbad1aed50644f0185e49f4254a9af7560 && python3 setup.py develop && python3 setup.py install && mkdir weights && cd weights && \
#     wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth && \
#     cd /app