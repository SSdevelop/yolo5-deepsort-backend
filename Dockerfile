FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV TZ=Asia/Hong_Kong

COPY . /app

WORKDIR /app

RUN apt-get update
RUN apt install software-properties-common -y
RUN apt-get install -y build-essential
RUN apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
RUN apt-get install -y ffmpeg libavfilter-dev libavutil-dev libsm6 libxext6 --fix-missing
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install python3.10 python3.10-distutils python3-pip -y
RUN python3.10 -m pip install --upgrade pip setuptools wheel
RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.10 /usr/bin/python3
RUN apt-get install -y wget

RUN pip install -r requirements.txt 
RUN cd model/GroundingDINO && \  
pip install -e .

CMD ["python3", "app.py"]

EXPOSE 5000