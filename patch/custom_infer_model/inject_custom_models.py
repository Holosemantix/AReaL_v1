# inject_custom_models.py
import os
import shutil
import sys


def _ensure_in_dict_list(file_path, marker_line, insert_line, check_str):
    """在文件中的某个列表里插入一行，如果还不存在的话。"""
    if not os.path.exists(file_path):
        return False
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    if check_str in content:
        return True
    if marker_line not in content:
        return False
    content = content.replace(marker_line, insert_line + "\n" + marker_line)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return True


def inject_sglang():
    try:
        import sglang
        sglang_dir = os.path.dirname(sglang.__file__)
        sglang_model_dir = os.path.join(sglang_dir, "srt", "models")

        patch_sglang_model_file = "patch/custom_infer_model/sglang/ailab_slm.py"

        shutil.copy(patch_sglang_model_file, sglang_model_dir)
        print("✅ SGLang: 算子和注册表注入成功！")
    except ImportError:
        print("⏭️ SGLang: 未安装，跳过注入。")
    except Exception as e:
        print(f"❌ SGLang: 注入失败，错误信息: {e}")


def inject_vllm():
    try:
        import vllm
        vllm_dir = os.path.dirname(vllm.__file__)
        vllm_model_dir = os.path.join(vllm_dir, "model_executor", "models")

        patch_vllm_model_file = "patch/custom_infer_model/vllm/ailab_slm.py"
        patch_vllm_registry_file = "patch/custom_infer_model/vllm/registry.py"

        shutil.copy(patch_vllm_model_file, vllm_model_dir)
        shutil.copy(patch_vllm_registry_file, vllm_model_dir)
        print("✅ vLLM: 算子和注册表注入成功！")
    except ImportError:
        print("⏭️ vLLM: 未安装，跳过注入。")
    except Exception as e:
        print(f"❌ vLLM: 注入失败，错误信息: {e}")


def inject_transformers():
    """将 AILabSLM 注册为 transformers 内置模型，避免 trust_remote_code 动态模块在 spawn 子进程中无法 pickle 的问题。"""
    try:
        import transformers
        transformers_dir = os.path.dirname(transformers.__file__)
        models_dir = os.path.join(transformers_dir, "models")
        target_dir = os.path.join(models_dir, "ailab_slm")
        os.makedirs(target_dir, exist_ok=True)

        source_dir = "patch/custom_infer_model/huggingface/ailab_slm"
        files_to_copy = [
            "configuration_ailab_slm.py",
            "modeling_ailab_slm.py",
            "tokenization_ailab_slm.py",
            "tokenization_ailab_slm_fast.py",
        ]
        for fname in files_to_copy:
            src = os.path.join(source_dir, fname)
            if os.path.exists(src):
                shutil.copy(src, target_dir)

        # 创建 __init__.py
        init_path = os.path.join(target_dir, "__init__.py")
        init_content = '''# AILabSLM built-in registration
from .configuration_ailab_slm import AILabSLMConfig
from .modeling_ailab_slm import AILabSLMForCausalLM, AILabSLMModel
from .tokenization_ailab_slm import AILabSLMTokenizer
from .tokenization_ailab_slm_fast import AILabSLMTokenizerFast
'''
        with open(init_path, "w", encoding="utf-8") as f:
            f.write(init_content)

        # 注册到 configuration_auto.py
        config_auto = os.path.join(models_dir, "auto", "configuration_auto.py")
        ok1 = _ensure_in_dict_list(
            config_auto,
            '        ("albert", "AlbertConfig"),',
            '        ("ailab_slm", "AILabSLMConfig"),',
            '("ailab_slm", "AILabSLMConfig")',
        )
        if ok1:
            print("✅ Transformers: CONFIG_MAPPING_NAMES 注册成功！")
        else:
            print("⚠️ Transformers: CONFIG_MAPPING_NAMES 注册可能已存在或文件结构不符。")

        # 注册 MODEL_NAMES_MAPPING
        ok1b = _ensure_in_dict_list(
            config_auto,
            '        ("albert", "ALBERT"),',
            '        ("ailab_slm", "AILabSLM"),',
            '("ailab_slm", "AILabSLM")',
        )
        if ok1b:
            print("✅ Transformers: MODEL_NAMES_MAPPING 注册成功！")

        # 注册到 modeling_auto.py (MODEL_MAPPING_NAMES)
        model_auto = os.path.join(models_dir, "auto", "modeling_auto.py")
        ok2 = _ensure_in_dict_list(
            model_auto,
            '        ("albert", "AlbertModel"),',
            '        ("ailab_slm", "AILabSLMModel"),',
            '("ailab_slm", "AILabSLMModel")',
        )
        if ok2:
            print("✅ Transformers: MODEL_MAPPING_NAMES 注册成功！")

        # 注册到 modeling_auto.py (MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
        ok3 = _ensure_in_dict_list(
            model_auto,
            '        ("albert", "AlbertForMaskedLM"),',
            '        ("ailab_slm", "AILabSLMForCausalLM"),',
            '("ailab_slm", "AILabSLMForCausalLM")',
        )
        if ok3:
            print("✅ Transformers: MODEL_FOR_CAUSAL_LM_MAPPING_NAMES 注册成功！")

        # 注册到 tokenization_auto.py
        tok_auto = os.path.join(models_dir, "auto", "tokenization_auto.py")
        ok4 = _ensure_in_dict_list(
            tok_auto,
            '        ("albert", ("AlbertTokenizer", "AlbertTokenizerFast")),',
            '        ("ailab_slm", ("AILabSLMTokenizer", "AILabSLMTokenizerFast")),',
            '("ailab_slm", ("AILabSLMTokenizer", "AILabSLMTokenizerFast"))',
        )
        if ok4:
            print("✅ Transformers: TOKENIZER_MAPPING_NAMES 注册成功！")

        print("✅ Transformers: ailab_slm 内置模型注入完成！")
    except Exception as e:
        print(f"❌ Transformers: 注入失败，错误信息: {e}")
        import traceback
        traceback.print_exc()


def patch_transformers_modules_path():
    """
    sglang 0.5.10+ 强制使用 spawn 启动 scheduler 子进程，pickle 需要子进程能 import
    transformers_modules.xxx 动态模块。把 huggingface modules 缓存目录加入 PYTHONPATH，
    确保 spawn 子进程可以找到这些动态模块。
    """
    try:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "modules")
        if not os.path.exists(cache_dir):
            print("⏭️ HuggingFace modules cache 目录不存在，跳过 PYTHONPATH patch。")
            return

        # 查找所有 transformers_modules 子目录
        transformers_modules_root = os.path.join(cache_dir, "transformers_modules")
        if not os.path.exists(transformers_modules_root):
            print("⏭️ transformers_modules 目录不存在，跳过 PYTHONPATH patch。")
            return

        # 需要把 modules 目录本身加入 PYTHONPATH，这样子进程才能 import transformers_modules.xxx
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        if cache_dir in current_pythonpath.split(os.pathsep):
            print("⏭️ PYTHONPATH 已经包含 HuggingFace modules 缓存目录。")
            return

        if current_pythonpath:
            new_pythonpath = cache_dir + os.pathsep + current_pythonpath
        else:
            new_pythonpath = cache_dir

        os.environ["PYTHONPATH"] = new_pythonpath
        # 同时更新当前进程的 sys.path，确保本进程也能找到
        if cache_dir not in sys.path:
            sys.path.insert(0, cache_dir)

        print(f"✅ PYTHONPATH patched with HuggingFace modules cache: {cache_dir}")
    except Exception as e:
        print(f"❌ PYTHONPATH patch 失败: {e}")


if __name__ == "__main__":
    print("🚀 开始执行 AILab SLM 框架热注入...")
    inject_sglang()
    inject_vllm()
    inject_transformers()
    patch_transformers_modules_path()
    print("✨ 模型注入环节完成。")
