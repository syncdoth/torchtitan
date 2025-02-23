import gc
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import fire
import torch
import torch.distributed.checkpoint as DCP
from transformers import AutoModelForCausalLM
from torchtitan.models import models_config
from torchtitan.models.llama.model import precompute_freqs_cis

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def copy_weights_hf_llama(
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, torch.Tensor],
) -> None:
    weight_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w3.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }

    for name, param in hf_weights.items():
        if "model.layers" in name:
            layer_i = name.replace("model.layers.", "").split(".")[0]
            from_name = name.replace(layer_i, "{}")
            to_name = weight_map[from_name]
            if to_name is None:
                continue
            to_name = to_name.format(layer_i)
        else:
            to_name = weight_map[name]
        state_dict[to_name] = param


@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    hf_model_name_or_path: str,
    model_name: str,
    flavor: str,
    output_dir: str,
    dtype: Optional[str] = None,
) -> None:
    if dtype is not None:
        dtype = getattr(torch, dtype)

    model_args = models_config[model_name][flavor]

    # Load the json file containing weight mapping
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name_or_path, torch_dtype=dtype
    )
    hf_weights = model.state_dict()

    llama_state_dict = {}
    copy_weights_hf_llama(llama_state_dict, hf_weights)
    llama_state_dict["freqs_cis"] = precompute_freqs_cis(
        model_args.dim // model_args.n_heads,
        model_args.max_seq_len,
        model_args.rope_theta,
        return_cos_sin=(
            not model_args.rope_interleaved or model_args.fused_rotary_embedding
        ),
    )
    gc.collect()
    print("Saving converted checkpoint")
    os.makedirs(output_dir, exist_ok=True)
    DCP.save(
        {"model": llama_state_dict}, checkpoint_id=os.path.join(output_dir, "step-0")
    )


if __name__ == "__main__":
    fire.Fire(convert_hf_checkpoint)
