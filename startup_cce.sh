#!/usr/bin/env bash

code_path=$(pwd)
# 1. 安装依赖
cp -r /opt/huawei/dataset/ag_data_wulan/data/pkgs_x86 /cache/
# 由于numpy版本冲突，实际需要安装<2.0.0版本
# 先安装好torch和vllm后，再通过requirements.txt安装剩下的全部依赖，会降级numpy版本
# 1. 进入离线包路径，修改成实际使用路径
cd /cache/pkgs_x86/

# 2. 安装 torch 和 vllm
pip install --no-index --find-links=./ torch==2.8.0+cu126 torchaudio==2.8.0+cu126 torchvision vllm==0.11.0
# 安装sglang
pip install --no-index --find-links=./ sglang==0.5.5.post1

# 3. 基于源码安装func_timeout、jieba、PyExt
cd func_timeout
python setup.py install
cd ..

cd jieba_source/jieba
python setup.py install
cd ../../

cd PyExt/
python setup.py install
cd ../

cd timeout-decorator/
python setup.py install
cd ../

cd tau2-bench
pip install --no-index --find-links=../  -e .
cd ../

# 4. 安装剩余依赖
pip install --no-index --find-links=./ -r requirements.txt

# 5. 更新flash-attn版本至2.8.3
pip install flash_attn-2.7.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
#pip install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

# 6. 更新transformer_engine[pytorch]
#pip wheel ./transformer_engine_torch-2.9.0.tar.gz \
#    --no-deps \
#    --no-build-isolation \
#    -w ./output_wheels_dir
pip install --no-index --find-links=./ transformer_engine[pytorch]

cd $code_path
echo code_path: $(pwd)

export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
#export PYTORCH_NPU_ALLOC_CONF="expandable_segments:False"

# 设置 NCCL参数, 和参考代码一致
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=128
export NCCL_IB_HCA='^=mlx5_bond_0'
export NCCL_DEBUG="INFO"
export NCCL_P2P_LEVEL="NVL"
export NCCL_IB_DISABLE=0
export NCCL_IBEXT_DISABLE=1
export NCCL_TIMEOUT=3600
# 启用高性能通信
export NCCL_P2P_DISABLE=0
export NCCL_NET_GDR_LEVEL="AUTO"

# 启动训练
#bash run_trainer_mtp.sh "$@"
bash ${run_shell_script} "$@"