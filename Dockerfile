# 使用 CUDA 11.6 开发版镜像（Ubuntu 20.04），便于编译 CUDA 代码
FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

# 安装必要的系统工具（wget、git、bzip2、ca-certificates等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    wget \
    git \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda 和依赖
RUN apt-get update && apt-get install -y wget build-essential && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 将 Miniconda 加入 PATH
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
# 设置工作目录为 /workspace/SPIn-NeRF
# 运行容器时可以将本地的 SPIn-NeRF 文件夹挂载到该目录
WORKDIR /workspace/SPIn-NeRF

# 复制 conda 环境配置文件 spinnerf_docker.yml 到当前工作目录
# 注意：spinnerf_docker.yml 应位于 SPIn-NeRF 文件夹的一级目录下
COPY spinnerf_docker.yml ./

# 通过 conda 创建虚拟环境（假定 yml 文件中定义的环境名称为 spinnerf）
RUN conda env create -f spinnerf_docker.yml

# 更新 conda（可选）并清理缓存
RUN conda update -n base -c defaults conda && conda clean -afy

# 设置默认使用 conda 环境中的 Python（环境名称 spinnerf）
ENV CONDA_DEFAULT_ENV=spinnerf
ENV PATH=$CONDA_DIR/envs/spinnerf/bin:$PATH

# 自动激活 conda 环境：初始化 bash 的 conda 配置，并在 .bashrc 中追加激活命令
RUN /opt/conda/bin/conda init bash && echo "conda activate spinnerf" >> ~/.bashrc

# 克隆三个需要安装的 Git 仓库，并添加 --recursive 参数以确保子模块也被克隆
RUN git clone --recursive https://github.com/openai/CLIP.git && \
    git clone --recursive https://github.com/NVlabs/tiny-cuda-nn.git && \
    git clone --recursive https://github.com/kornia/kornia.git
# ENV TCNN_CUDA_ARCHITECTURES="7.5"
# 使用 editable 模式安装这三个包，其中 tiny-cuda-nn 只安装其 bindings/torch 子目录
RUN export TCNN_CUDA_ARCHITECTURES="75, 80, 86, 90" && \
    echo "TCNN_CUDA_ARCHITECTURES is set to '$TCNN_CUDA_ARCHITECTURES'" && \
    pip install ./CLIP && \
    pip install ./tiny-cuda-nn/bindings/torch && \
    pip install ./kornia

# 声明挂载点，方便在运行容器时将本地 SPIn-NeRF 文件夹挂载进来
VOLUME ["/workspace/SPIn-NeRF"]

# 默认启动 bash（注意，此处 .bashrc 会自动激活 spinnerf 环境）
CMD ["/bin/bash"]