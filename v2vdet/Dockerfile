FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda4-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Set up Env
ENV PATH /opt/conda/bin:$PATH
COPY environment.yml .
RUN conda env create -f environment.yml
RUN conda install openssl=3.0.16

# 設定 bash 啟動時自動初始化 conda
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# 創建一個純粹的互動式環境
WORKDIR /app

# 啟動 bash 終端機
CMD ["/bin/bash"]
