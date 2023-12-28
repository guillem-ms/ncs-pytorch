FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ARG USERNAME="gmartinez"
ARG USER_UID=1001
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# apt and apt-get update
RUN apt-get update && apt-get install -y nano git unzip wget && apt-get clean

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/$USERNAME/miniconda3 \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Set up the Conda environment
ENV PATH /home/$USERNAME/miniconda3/bin:$PATH

COPY .devcontainer/enviroment.yml /home/$USERNAME/enviroment.yml

RUN . /home/$USERNAME/miniconda3/etc/profile.d/conda.sh \
    && conda env create -f /home/$USERNAME/enviroment.yml 

# Set the Conda environment to activate on run
RUN echo "source activate ncs-py" > /home/$USERNAME/.bashrc

# Set the default user. Omit if you want to keep the default as root.
USER $USERNAME