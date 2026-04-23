#!/usr/bin/env bash

ls
code_path=$(pwd)

yes | cp -rf /opt/huawei/dataset/ag_data/code/AReaL_v1/. ./

# 1. 安装依赖
# 由于numpy版本冲突，实际需要安装<2.0.0版本
# 先安装好torch和vllm后，再通过requirements.txt安装剩下的全部依赖，会降级numpy版本
# 1. 进入离线包路径，修改成实际使用路径
cd /opt/huawei/dataset/ag_data/pkg_x86

# 2. 安装 torch 和 vllm
pip install --no-index --find-links=./ torch==2.9.1 torchaudio torchvision vllm==0.15.0
# 安装sglang
pip install --no-index --find-links=./ sglang==0.5.10

# 3. 安装 stable-worldmodel[train,env]
pip install --no-index --find-links=./ stable-worldmodel[train,env]

# 5. 更新flash-attn版本至2.8.3
pip install --no-index --find-links=./ transformer_engine[pytorch]==2.13.0 --no-build-isolation
pip install --no-index --find-links=./  flash_attn==2.8.3 --no-build-isolation

# 4. 安装剩余依赖
pip uninstall transformers -y
pip install --no-index --find-links=./ -r requirements.txt
pip uninstall -y tensorboard tensorboard-data-server protobuf
pip install --no-index --find-links=./  "protobuf<5.0" tensorboard
pip uninstall opencv-python opencv -y
yes | rm -rf /usr/local/lib/python3.10/dist-packages/opencv*
yes | rm -rf /usr/local/lib/python3.10/dist-packages/cv2*
pip install --no-index --find-links=./ opencv-python-headless
pip install --no-index --find-links=./ numpy==1.26.4

cd $code_path
echo code_path: $(pwd)

# =========================================================================
# 执行自定义模型热注入
# =========================================================================
echo "执行框架源码修改..."
python patch/custom_infer_model/inject_custom_models.py
# =========================================================================

# webstudio为eth0, 训练任务为bond0
#export GLOO_SOCKET_IFNAME=eth0
#export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0

# 设置 NCCL参数
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

export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# 启动训练
bash ${run_shell_script} "$@"