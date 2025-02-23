"""This file converts the DCP format llama checkpoint to HuggingFace format."""

import json
import os

import fire
from safetensors.torch import save_file

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import torch
import torch.distributed.checkpoint as DCP
from transformers import AutoTokenizer
from torchtitan.checkpoint import TrainState
from torchtitan.config_manager import JobConfig
from torchtitan.models import model_name_to_cls, models_config
from torchtitan.models.llama.model import ModelArgs
from torchtitan.components.optimizer import build_optimizers


def convert_llama_to_hf(
    model_state_dict: dict[str, torch.Tensor],
    model_args: ModelArgs,
    torch_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    hf_state_dict = {}
    for key, value in model_state_dict.items():
        if key == "freqs_cis":
            continue

        value = value.to(torch_dtype)
        if key.startswith("tok_embeddings."):
            hf_key = key.replace("tok_embeddings.", "model.embed_tokens.")
            if model_args.tie_word_embeddings:
                hf_state_dict["lm_head.weight"] = value
        elif key.startswith("output"):
            hf_key = key.replace("output", "lm_head")
        elif key.startswith("layers."):
            layer_idx = int(key.split(".")[2])
            if key.startswith(f"layers.{layer_idx}.attention.wq"):
                hf_key = key.replace(
                    f"layers.{layer_idx}.attention.wq",
                    f"layers.{layer_idx}.self_attn.q_proj",
                )
            elif key.startswith(f"layers.{layer_idx}.attention.wk"):
                hf_key = key.replace(
                    f"layers.{layer_idx}.attention.wk",
                    f"layers.{layer_idx}.self_attn.k_proj",
                )
            elif key.startswith(f"layers.{layer_idx}.attention.wv"):
                hf_key = key.replace(
                    f"layers.{layer_idx}.attention.wv",
                    f"layers.{layer_idx}.self_attn.v_proj",
                )
            elif key.startswith(f"layers.{layer_idx}.attention.wo"):
                hf_key = key.replace(
                    f"layers.{layer_idx}.attention.wo",
                    f"layers.{layer_idx}.self_attn.o_proj",
                )
            elif key.startswith(f"layers.{layer_idx}.attention_norm"):
                hf_key = key.replace(
                    f"layers.{layer_idx}.attention_norm",
                    f"layers.{layer_idx}.input_layernorm",
                )
            elif key.startswith(f"layers.{layer_idx}.ffn_norm"):
                hf_key = key.replace(
                    f"layers.{layer_idx}.ffn_norm",
                    f"layers.{layer_idx}.post_attention_layernorm",
                )
            elif key.startswith(f"layers.{layer_idx}.feed_forward.w1"):
                hf_key = key.replace(
                    f"layers.{layer_idx}.feed_forward.w1",
                    f"layers.{layer_idx}.mlp.gate_proj",
                )
            elif key.startswith(f"layers.{layer_idx}.feed_forward.w2"):
                hf_key = key.replace(
                    f"layers.{layer_idx}.feed_forward.w2",
                    f"layers.{layer_idx}.mlp.down_proj",
                )
            elif key.startswith(f"layers.{layer_idx}.feed_forward.w3"):
                hf_key = key.replace(
                    f"layers.{layer_idx}.feed_forward.w3",
                    f"layers.{layer_idx}.mlp.up_proj",
                )
        else:
            hf_key = key
        hf_state_dict[hf_key] = value
    return hf_state_dict


def convert_config_to_hf(model_args: ModelArgs, dtype: str, tokenizer) -> dict:
    if model_args.ffn_dim is not None:
        intermediate_size = model_args.ffn_dim
    elif model_args.ffn_dim_multiplier is not None:
        intermediate_size = int(model_args.ffn_dim_multiplier * model_args.dim * 8 / 3)
    else:
        intermediate_size = int(model_args.dim * 8 / 3)
    # copied from https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/config.json
    config = {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "hidden_act": "silu",
        "hidden_size": model_args.dim,
        "initializer_range": 0.02,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": model_args.max_seq_len,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": model_args.n_heads,
        "num_hidden_layers": model_args.n_layers,
        "num_key_value_heads": model_args.n_kv_heads,
        "pretraining_tp": 1,
        "rms_norm_eps": model_args.norm_eps,
        "rope_theta": model_args.rope_theta,
        "tie_word_embeddings": False,
        "torch_dtype": dtype,
        "use_cache": True,
        "vocab_size": model_args.vocab_size,
    }
    return config


def ugly_load_job_config(config_file: str) -> JobConfig:
    job_config = JobConfig()
    args, _ = job_config.parse_args_from_command_line([])
    args_dict = job_config._args_to_two_level_dict(args)
    try:
        with open(config_file, "rb") as f:
            for k, v in tomllib.load(f).items():
                # to prevent overwrite of non-specified keys
                args_dict[k] |= v
    except (FileNotFoundError, tomllib.TOMLDecodeError) as e:
        print(f"Error while loading the configuration file: {config_file}")
        print(f"Error details: {str(e)}")
        raise e
    job_config.args_dict = args_dict
    for k, v in args_dict.items():
        class_type = type(k.title(), (), v)
        setattr(job_config, k, class_type())
    # print(job_config.to_dict())
    return job_config


def main(
    model_name: str,
    tokenizer_name: str,
    flavor: str,
    config_file: str,
    checkpoint_id: str,
    dtype: str | None = None,
):
    if dtype is None:
        dtype = "bfloat16"
    torch_dtype = getattr(torch, dtype)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    job_config = ugly_load_job_config(config_file)
    model_cls = model_name_to_cls[model_name]
    model_args = models_config[model_name][flavor]
    model_args.norm_type = job_config.model.norm_type
    model_args.vocab_size = len(tokenizer)
    model_args.max_seq_len = job_config.training.seq_len
    model = model_cls.from_model_args(model_args)

    optimizer = build_optimizers([model], job_config).optimizers[0]
    states = {
        "model": model.state_dict(),
        "train_state": TrainState(),
        "lr_scheduler": {
            "base_lrs": [job_config.optimizer.lr],
            "last_epoch": 1,
            "verbose": False,
            "_step_count": 1,
            "_get_lr_called_within_step": False,
            "_last_lr": [job_config.optimizer.lr],
            "lr_lambdas": [{}],
        },  # NOTE: this is just a dummy state_dict, cuz we don't need it
        "optimizer": optimizer.state_dict(),
    }

    DCP.load(states, checkpoint_id=checkpoint_id)

    # delete everything other than model state
    for k in list(states.keys()):
        if k != "model":
            del states[k]

    model_state_dict = states["model"]
    hf_state_dict = convert_llama_to_hf(model_state_dict, model_args, torch_dtype)
    hf_config_json = convert_config_to_hf(model_args, dtype)

    os.makedirs(checkpoint_id + "-hf", exist_ok=True)

    # Save tensors

    # Add metadata for safetensors
    metadata = {"format": "pt", "framework": "pytorch", "task": "text-generation"}
    save_dir = os.path.join(checkpoint_id + "-hf")
    save_file(
        hf_state_dict,
        os.path.join(save_dir, "model.safetensors"),
        metadata=metadata,
    )
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(hf_config_json, f)


if __name__ == "__main__":
    fire.Fire(main)
