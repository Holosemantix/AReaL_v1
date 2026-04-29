#!/usr/bin/env bash

NNODES="$MA_NUM_HOSTS"
GPUS_PER_NODE="$MA_NUM_GPUS"
TOTAL_GPUS=$(($GPUS_PER_NODE * $NNODES))
NODE_RANK="$VC_TASK_INDEX"
WORKER_HOSTS="$VC_WORKER_HOSTS"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
MASTER_IP=$(python3 -c "import socket; print(socket.gethostbyname('${MASTER_ADDR}'))")
MASTER_PORT="6060"
DEVICE=$(command -v nvidia-smi >/dev/null && echo "cuda" || (command -v npu-smi >/dev/null && echo "npu" || echo "none"))

echo "------> system config <------"
echo "WORKER_HOSTS: ${WORKER_HOSTS}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_IP: ${MASTER_IP}"
echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "TOTAL_GPUS: ${TOTAL_GPUS}"
echo "------>  <------"

# ==========================================
# 构建 Hydra 参数数组：仅当变量有值时才添加
# 格式为 key=value (无 -- 前缀)
# ==========================================
CMD_ARGS=()

# 定义添加参数的辅助函数
add_override() {
    local key="$1"
    local value="$2"
    # -n 判断字符串长度是否非0
    if [ -n "$value" ]; then
        # Hydra 格式：key=value
        CMD_ARGS+=("$key=$value")
    fi
}

# 依次处理所有可选参数
add_override "experiment_name" "${experiment_name}"
add_override "trial_name" "${trial_name}"
add_override "total_train_epochs" "${total_train_epochs}"
add_override "dynamic_bs" "${dynamic_bs}"
add_override "allocation_mode" "${allocation_mode}"
add_override "cluster.fileroot" "${cluster_fileroot}"
add_override "cluster.name_resolve.nfs_record_root" "${cluster_name_resolve_nfs_record_root}"

# Generation Config
add_override "gconfig.n_samples" "${gconfig_n_samples}"
add_override "gconfig.temperature" "${gconfig_temperature}"
add_override "gconfig.max_new_tokens" "${gconfig_max_new_tokens}"
add_override "gconfig.enable_thinking" "${gconfig_enable_thinking}"

# Eval Generation Config
add_override "eval_gconfig.n_samples" "${eval_gconfig_n_samples}"
add_override "eval_gconfig.temperature" "${eval_gconfig_temperature}"
add_override "eval_gconfig.top_p" "${eval_gconfig_top_p}"
add_override "eval_gconfig.top_k" "${eval_gconfig_top_k}"
add_override "eval_gconfig.max_new_tokens" "${eval_gconfig_max_new_tokens}"

# Rollout
add_override "rollout.max_concurrent_rollouts" "${rollout_max_concurrent_rollouts}"
add_override "rollout.max_head_offpolicyness" "${rollout_max_head_offpolicyness}"
add_override "rollout.dump_to_file" "${rollout_dump_to_file}"

# Actor
add_override "actor.dtype" "${actor_dtype}"
add_override "actor.path" "${actor_path}"
add_override "actor.optimizer.lr" "${actor_optimizer_lr}"
add_override "actor.optimizer.type" "${actor_optimizer_type}"
add_override "actor.importance_sampling_level" "${actor_importance_sampling_level}"
add_override "actor.kl_ctl" "${actor_kl_ctl}"
add_override "actor.use_sapo_loss" "${actor_use_sapo_loss}"
add_override "actor.use_decoupled_loss" "${actor_use_decoupled_loss}"
add_override "actor.overlong_reward_penalty" "${actor_overlong_reward_penalty}"
add_override "actor.overlong_tokens" "${actor_overlong_tokens}"
add_override "actor.overlong_penalty_factor" "${actor_overlong_penalty_factor}"
add_override "actor.mb_spec.max_tokens_per_mb" "${actor_mb_spec_max_tokens_per_mb}"
add_override "actor.reward_norm.denominator" "${actor_reward_norm_denominator}"
add_override "actor.ig_reward_params.compute_backend" "${actor_ig_reward_params_compute_backend}"
add_override "actor.ig_reward_params.mini_batch_size" "${actor_ig_reward_params_mini_batch_size}"
add_override "actor.ig_reward_params.reward_mode" "${actor_ig_reward_params_reward_mode}"
add_override "actor.ig_reward_params.beta" "${actor_ig_reward_params_beta}"
add_override "actor.ig_reward_params.lambda_val" "${actor_ig_reward_params_lambda_val}"
add_override "actor.ig_reward_params.use_watermark_selection" "${actor_ig_reward_params_use_watermark_selection}"
add_override "actor.ig_reward_params.use_peak_selection" "${actor_ig_reward_params_use_peak_selection}"
add_override "actor.ig_reward_params.enable_varlen_packing" "${actor_ig_reward_params_enable_varlen_packing}"
add_override "actor.ig_reward_params.max_tokens_per_forward" "${actor_ig_reward_params_max_tokens_per_forward}"

# Ref & Engines
add_override "ref.path" "${ref_path}"
add_override "sglang.mem_fraction_static" "${sglang_mem_fraction_static}"
add_override "sglang.context_length" "${sglang_context_length}"
add_override "vllm.max_model_len" "${vllm_max_model_len}"

# Datasets
add_override "train_dataset.path" "${train_dataset_path}"
add_override "train_dataset.batch_size" "${train_dataset_batch_size}"
add_override "train_dataset.max_length" "${train_dataset_max_length}"
add_override "valid_dataset.batch_size" "${valid_dataset_batch_size}"
add_override "valid_dataset.path" "${valid_dataset_path}"

# Utilities
add_override "saver.freq_epochs" "${saver_freq_epochs}"
add_override "saver.freq_steps" "${saver_freq_steps}"
add_override "recover.mode" "${recover_mode}"
add_override "evaluator.freq_epochs" "${evaluator_freq_epochs}"
add_override "evaluator.freq_steps" "${evaluator_freq_steps}"

add_override "stats_logger.swanlab.mode" "${swanlab_mode}"
add_override "stats_logger.swanlab.project" "${swanlab_project}"
add_override "stats_logger.swanlab.name" "${swanlab_name}"
add_override "stats_logger.swanlab.config" "${swanlab_config}"
add_override "stats_logger.swanlab.logdir" "${swanlab_logdir}"
add_override "stats_logger.swanlab.api_key" "${swanlab_api_key}"

# Cluster Info (即使有值也只在传入时覆盖，否则用 config 默认)
add_override "cluster.n_nodes" "${NNODES}"
add_override "cluster.n_gpus_per_node" "${GPUS_PER_NODE}"

# ==========================================
# 2. 启动训练
# ==========================================

if [ "$NNODES" = "1" ]; then
  # 单节点local启动
  add_override "scheduler.type" "local"
  # --config 是特殊的，需要保留 --
  # "${CMD_ARGS[@]}" 展开为 Hydra 的 overrides
  python3 -m "${startup_file}" \
  --config "${config}" \
  "${CMD_ARGS[@]}"

else
  # 多节点ray启动
  add_override "scheduler.type" "ray"

  if [ "${NODE_RANK}" = "0" ]; then
    # 主节点启动
    if [ "$DEVICE" = "npu" ]; then
      ray start --head --port $MASTER_PORT --dashboard-host=0.0.0.0 --dashboard-port=8260 --resources='{"NPU": '$GPUS_PER_NODE'}'
    else
      ray start --head --port $MASTER_PORT --dashboard-host=0.0.0.0 --dashboard-port=8260 --num-gpus=$GPUS_PER_NODE
    fi

    sleep 5

    while true; do
      ray_status_output=$(ray status 2>/dev/null || echo "")

      if [ "$ray_status_output" = "No cluster status. It may take a few seconds for the Ray internal services to start up." ]; then
        echo "$ray_status_output"
        continue
      fi

      # 尝试提取 GPU 数量
      if [ "$DEVICE" = "npu" ]; then
        gpu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
      else
        gpu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*GPU)' | head -n 1 2>/dev/null || echo "")
      fi

      # 检查 gpu_count 是否为空
      if [ -z "$gpu_count" ]; then
        echo "无法获取 GPU 数量，Ray 状态可能未就绪..."
        sleep 5
        continue
      fi

      # 转换 GPU 数量为整数
      gpu_count_int=$(echo "$gpu_count" | awk '{print int($1)}')
      device_count=$((gpu_count_int / $GPUS_PER_NODE))

      # 判断 device_count 是否与 NNODES 相等
      if [ "$device_count" -eq "$NNODES" ]; then
        echo "Ray cluster is ready with $device_count devices (from $gpu_count resources), starting Python script."
        ray status

        # 使用数组传参，Hydra 格式
        python3 -m "${startup_file}" \
        --config "${config}" \
        "${CMD_ARGS[@]}"

        break
      else
        echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
        sleep 5
      fi
    done
  else
    # 子节点尝试往主节点注册ray直到成功
    while true; do
      if [ "$DEVICE" = "npu" ]; then
        ray start --address="$MASTER_IP:$MASTER_PORT" --resources='{"NPU": '$GPUS_PER_NODE'}'
      else
        ray start --address="$MASTER_IP:$MASTER_PORT" --num-gpus=$GPUS_PER_NODE
      fi

      # 检查连接是否成功
      ray status
      if [ $? -eq 0 ]; then
        echo "Successfully connected to the Ray cluster!"
        break
      else
        echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
        sleep 5
      fi
    done
  fi
  sleep 999999
fi
