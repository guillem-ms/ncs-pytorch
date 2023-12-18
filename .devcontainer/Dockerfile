ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8.5"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# apt and apt-get update
RUN apt-get update && apt-get install -y nano git unzip wget && apt-get clean

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3 \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Set up the Conda environment
ENV PATH /root/miniconda3/bin:$PATH

RUN . /root/miniconda3/etc/profile.d/conda.sh \
    && conda env create -f hood.yml 

# Set the Conda environment to activate on run
RUN echo "source activate main" > /root/.bashrc

ENV PROJECT_PATH=/root/ncs-py
RUN mkdir $PROJECT_PATH
WORKDIR $PROJECT_PATH

