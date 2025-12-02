import os
import sys
import json
import shutil

from collections import OrderedDict
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, AutoModel


vit_300m_config = {
    "architectures": [
        "InternVisionModel"
    ],
    "attention_bias": True,
    "attention_dropout": 0.0,
    "dropout": 0.0,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "hidden_size": 1024,
    "image_size": [
        448,
        448
    ],
    "initializer_factor": 0.1,
    "initializer_range": 1e-10,
    "intermediate_size": 4096,
    "layer_norm_eps": 1e-06,
    "layer_scale_init_value": 0.1,
    "model_type": "internvl_vision",
    "norm_type": "layer_norm",
    "num_attention_heads": 16,
    "num_channels": 3,
    "num_hidden_layers": 24,
    "patch_size": [
        14,
        14
    ],
    "projection_dropout": 0.0,
    "torch_dtype": "bfloat16",
    "use_absolute_position_embeddings": True,
    "use_mask_token": False,
    "use_mean_pooling": True,
    "use_qk_norm": False
}


vit_6b_config = {
    "architectures": [
        "InternVisionModel"
    ],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "dropout": 0.0,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "hidden_size": 3200,
    "image_size": [
        448,
        448
    ],
    "initializer_factor": 0.1,
    "initializer_range": 1e-10,
    "intermediate_size": 12800,
    "layer_norm_eps": 1e-06,
    "layer_scale_init_value": 0.1,
    "model_type": "internvl_vision",
    "norm_type": "rms_norm",
    "num_attention_heads": 25,
    "num_channels": 3,
    "num_hidden_layers": 45,
    "patch_size": [
        14,
        14
    ],
    "projection_dropout": 0.0,
    "torch_dtype": "bfloat16",
    "use_absolute_position_embeddings": True,
    "use_mask_token": False,
    "use_mean_pooling": True,
    "use_qk_norm": True
}


def convert_chat_config_to_hf(hf_config_path: str, vit_config: dict) -> None:
    new_cfg = vit_config.copy()
    with open(hf_config_path, "w") as f:
        json.dump(new_cfg, f, indent=2)

    print(f"转换完成，输出文件: {hf_config_path}")


def convert_keys_to_hf(custom_state_dict):
    new_state_dict = OrderedDict()
    qkv_split_buffer = {}

    for key, value in custom_state_dict.items():
        # === embeddings ===
        if key == "embeddings.class_embedding":
            new_key = "embeddings.cls_token"
        elif key.startswith("embeddings.patch_embedding"):
            new_key = key.replace(
                "embeddings.patch_embedding",
                "embeddings.patch_embeddings.projection"
            )
        elif key == "embeddings.position_embedding":
            new_key = "embeddings.position_embeddings"

        # === encoder ===
        elif key.startswith("encoder.layers."):
            parts = key.split(".")
            layer_id = parts[2]
            suffix = ".".join(parts[3:])
            base = f"encoder.layer.{layer_id}."

            if suffix.startswith("attn.qkv.weight"):
                qkv_split_buffer[(layer_id, "weight")] = value
                continue
            elif suffix.startswith("attn.qkv.bias"):
                qkv_split_buffer[(layer_id, "bias")] = value
                continue
            elif suffix.startswith("attn.proj."):
                new_key = base + "attention.projection_layer." + suffix.split(".")[-1]
            elif suffix.startswith("norm1."):
                new_key = base + "layernorm_before." + suffix.split(".")[-1]
            elif suffix.startswith("norm2."):
                new_key = base + "layernorm_after." + suffix.split(".")[-1]
            elif suffix == "ls1":
                new_key = base + "lambda_1"
            elif suffix == "ls2":
                new_key = base + "lambda_2"
            else:
                new_key = base + suffix

        if '.attn.q_norm.' in new_key:
            new_key = new_key.replace('.attn.q_norm.', '.attention.q_norm.')

        if '.attn.k_norm.' in new_key:
            new_key = new_key.replace('.attn.k_norm.', '.attention.k_norm.')

        if '.attn.v_norm.' in new_key:
            new_key = new_key.replace('.attn.v_norm.', '.attention.v_norm.')

        new_state_dict[new_key] = value

    # === 6. 拆分 QKV ===
    for (layer_id, typ), tensor in qkv_split_buffer.items():
        d = tensor.shape[0] // 3
        q, k, v = tensor[:d], tensor[d:2 * d], tensor[2 * d:]
        base = f"encoder.layer.{layer_id}.attention."
        if typ == "weight":
            new_state_dict[base + "q_proj.weight"] = q
            new_state_dict[base + "k_proj.weight"] = k
            new_state_dict[base + "v_proj.weight"] = v
        else:
            new_state_dict[base + "q_proj.bias"] = q
            new_state_dict[base + "k_proj.bias"] = k
            new_state_dict[base + "v_proj.bias"] = v

    return new_state_dict


if __name__ == "__main__":
    # mllm_custom_path = sys.argv[1]
    # mllm_save_path = os.path.join(
    #     "/mnt/shared-storage-user/intern7shared/wangweiyun/OpenGVLab-rc1-hf",
    #     f"{os.path.basename(mllm_custom_path)}-HF",
    # )
    
    # mllm_custom_path = "/mnt/shared-storage-user/intern7shared/share_ckpt_hf/InternViT-300M-448px-V2_5"
    # mllm_save_path = "/mnt/shared-storage-user/intern7shared/share_ckpt_hf/InternViT-300M-448px-V2_5-HF"
    
    mllm_custom_path = "/mnt/shared-storage-user/intern7shared/share_ckpt_hf/InternViT-6B-448px-V2_5"
    mllm_save_path = "/mnt/shared-storage-user/intern7shared/share_ckpt_hf/InternViT-6B-448px-V2_5-HF"

    print(f"{mllm_custom_path=}")
    print(f"{mllm_save_path=}")

    os.makedirs(mllm_save_path, exist_ok=True)

    # 1. 将自定义模型配置转换为 HF 格式并保存
    if '6B' in mllm_custom_path:
        vit_config = vit_6b_config
        print('Use ViT-6B')
    else:
        vit_config = vit_300m_config
        print('Use ViT-300M')

    hf_config_path = os.path.join(mllm_save_path, "config.json")
    convert_chat_config_to_hf(hf_config_path, vit_config)

    config = AutoConfig.from_pretrained(mllm_save_path)
    model = AutoModel.from_config(config)

    print(f"模型已加载到 GPU，并使用转换后的 HF config: {hf_config_path}")

    # 加载 HF safetensors 权重
    checkpoint_paths = [os.path.join(mllm_custom_path, f) for f in os.listdir(mllm_custom_path) if f.endswith('.safetensors')]
    print(f"\n🔍 Found checkpoint files: {checkpoint_paths}")
    model_state_dict_hf = {}
    for checkpoint_path in checkpoint_paths:
        with safe_open(checkpoint_path, framework="pt") as f:
            for k in f.keys():
                model_state_dict_hf[k] = f.get_tensor(k)

    # 转换为旧 key 命名风格
    model_state_dict = convert_keys_to_hf(model_state_dict_hf)

    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=True)
    print(f"\n❌ Missing keys: {missing_keys}")
    print(f"⚠️ Unexpected keys: {unexpected_keys}")

    model.save_pretrained(mllm_save_path)
