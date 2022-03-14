# Use nvidia/cuda image
FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04
MAINTAINER Hiren Namera
# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# install anaconda
RUN apt-get update
RUN apt-get install -y wget bzip2 ca-certificates ffmpeg libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
        apt-get clean
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH /opt/conda/bin:$PATH

COPY app /app

WORKDIR /app

# setup conda virtual environment
# COPY ./requirements.yaml /tmp/requirements.yaml
RUN conda create -n yolo_hrn python=3.8 -y

# RUN conda activate yolo_hrn
# RUN echo "conda activate yolo_hrn" >> ~/.bashrc
# RUN apt-get install ffmpeg libsm6   -y
ENV PATH /opt/conda/envs/yolo_hrn/bin:$PATH
ENV CONDA_DEFAULT_ENV yolo_hrn

RUN pip install -r requirements.txt

RUN conda install pytorch torchvision cudatoolkit=10.2 -c pytorch -y
EXPOSE 80

# Set the working directory

# Run the flask server for the endpoints
CMD python -u app.py