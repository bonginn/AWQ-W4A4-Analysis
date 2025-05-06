import torch
from torch import nn
from functools import partial
from qmodule import W8A8Linear, W4A4Linear

def quantize_llama_like_w8a8(
    model, weight_quant='per_channel', weight_group_size=128, act_quant="per_token", quantize_bmm_input=False
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )
    
    for name, m in model.model.named_modules():
        if isinstance(m, LlamaMLP):
            m.gate_proj = W8A8Linear.from_float(
                m.gate_proj, weight_quant=weight_quant, weight_group_size=weight_group_size, act_quant=act_quant
            )
            m.up_proj = W8A8Linear.from_float(
                m.up_proj, weight_quant=weight_quant, weight_group_size=weight_group_size, act_quant=act_quant
            )
            m.down_proj = W8A8Linear.from_float(
                m.down_proj, weight_quant=weight_quant, weight_group_size=weight_group_size, act_quant=act_quant
            )
        elif isinstance(m, LlamaAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                weight_group_size=weight_group_size,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                weight_group_size=weight_group_size,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                weight_group_size=weight_group_size,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = W8A8Linear.from_float(
                m.o_proj, weight_quant=weight_quant, weight_group_size=weight_group_size, act_quant=act_quant
            )
    return model


def quantize_llama_like_w4a4(
    model, weight_quant='per_channel', weight_group_size=128, act_quant="per_token", quantize_bmm_input=False
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, LlamaMLP):
            m.gate_proj = W4A4Linear.from_float(
                m.gate_proj, weight_quant=weight_quant, weight_group_size=weight_group_size, act_quant=act_quant
            )
            m.up_proj = W4A4Linear.from_float(
                m.up_proj, weight_quant=weight_quant, weight_group_size=weight_group_size, act_quant=act_quant
            )
            m.down_proj = W4A4Linear.from_float(
                m.down_proj, weight_quant=weight_quant, weight_group_size=weight_group_size, act_quant=act_quant
            )
        elif isinstance(m, LlamaAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W4A4Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                weight_group_size=weight_group_size,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = W4A4Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                weight_group_size=weight_group_size,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = W4A4Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                weight_group_size=weight_group_size,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = W4A4Linear.from_float(
                m.o_proj, weight_quant=weight_quant, weight_group_size=weight_group_size, act_quant=act_quant
            )
    return model

def quantize_model(
    model, n_bits=None, weight_quant='per_channel', weight_group_size=128, act_quant="per_token", quantize_bmm_input=False
):
    from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
    assert n_bits in [4, 8], f"n_bit should be 4 or 8, but got {n_bits}"
    if isinstance(model, LlamaPreTrainedModel) and n_bits == 8:
        model = quantize_llama_like_w8a8(
            model, 
            weight_quant=weight_quant, 
            weight_group_size=weight_group_size, 
            act_quant=act_quant, 
            quantize_bmm_input=quantize_bmm_input
        )
    elif isinstance(model, LlamaPreTrainedModel) and n_bits == 4:
        model = quantize_llama_like_w4a4(
            model, 
            weight_quant=weight_quant, 
            weight_group_size=weight_group_size, 
            act_quant=act_quant, 
            quantize_bmm_input=quantize_bmm_input
        )
    else:
        raise NotImplementedError(f"Quantization for {type(model)} is not implemented")
        