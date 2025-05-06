import torch
import torch.nn as nn

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from fake_quant import (
    pseudo_quantize_tensor, 
    quantize_activation_per_tensor_absmax,
    quantize_activation_per_token_absmax,
)

def _ensure_tensor(x, device, dtype):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    if isinstance(x[0], torch.Tensor):           # 1 batch
        x = torch.stack(x, dim=0).unsqueeze(0)   # (1, T, D)
    else:                                        # multi batch
        x = torch.stack([torch.stack(seq, dim=0) for seq in x], dim=0)
    return x.to(device=device, dtype=dtype)


@torch.no_grad()
def smooth_ln_fcs_llama_like(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, LlamaRMSNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

def awq_ln_fcs_llama_like(ln, fcs, x, block=None, kwargs={}, n_grid=20, n_bit=4, q_group_size=128):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, LlamaRMSNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == len(x[0])

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    x = _ensure_tensor(x, device, dtype)

    is_attn_block = hasattr(block, 'q_proj') and hasattr(block, 'k_proj') and hasattr(block, 'v_proj')
    
    with torch.no_grad():
        if is_attn_block:
            batch_size, seq_len = len(x), len(x[0])
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            attn_kwargs = dict(
                hidden_states=x,
                position_ids=position_ids,
            )
            attn_kwargs.update(kwargs)
            
            org_out = block(**attn_kwargs)
        else:
            org_out = block(x, **kwargs)
            
        if isinstance(org_out, tuple):
            org_out = org_out[0]
    
    act_scales = x.abs().view(-1, x.shape[-1]).mean(0)
    # You Can Try :
    # act_scales = x.abs().view(-1, x.shape[-1]).max(0).values
    # act_scales = x.abs().view(-1, x.shape[-1]).mean(0) + x.abs().view(-1, x.shape[-1]).max(0).values 
    act_scales = torch.tensor(act_scales, device=device, dtype=dtype)

    best_error = float("inf")
    best_ratio = -1
    best_scales = None

    history = [] # (alpha, error)

    org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
    for i in range(n_grid):
        alpha = (i / n_grid) 
        
        scales = act_scales.pow(alpha).clamp(min=1e-4).view(-1) # prevent [1, n]
        scales = scales / (scales.max() * scales.min()).sqrt() # Normalize, refer original code from llm-awq.
        # You Also Can Try:
        # scales = 2 ** (torch.log2(scales) + N)
        # N = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        
        # Step 1: quantize weight
        for fc in fcs:
            fc.weight.mul_(scales.view(1, -1)).to(fc.weight.device)
            fc.weight.data = pseudo_quantize_tensor(fc.weight.data, n_bits=n_bit, q_group_size=q_group_size) / scales.view(1, -1)

        # Step 2: quantize input
        x_scaled = x / scales.view(1, -1)
        x_q = quantize_activation_per_token_absmax(x_scaled, n_bits=n_bit)

        # Step 3: forward
        if is_attn_block:
            batch_size, seq_len = len(x_q), len(x_q[0])
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            attn_kwargs = dict(
                hidden_states=x_q,
                position_ids=position_ids,
            )
            
            attn_kwargs.update(kwargs)
            
            out = block(**attn_kwargs)
        else:
            
            out = block(x_q, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        error = (out - org_out).float().pow(2).mean().item() # float to prevent overflow

        # Step 4: find the best ratio
        if error < best_error:
            best_error = error
            best_ratio = alpha
            best_scales = scales.clone()
        history.append((alpha, error))

        # Step 5: restore the original state
        block.load_state_dict(org_sd)

    if best_ratio == -1:
        print(history)
        raise ValueError("No best ratio found.")

    print('history:', history)
    print('best_ratio:', best_ratio)
    
    best_scales = best_scales.to(device=device, dtype=dtype)
    ln.weight.div_(best_scales)
    for fc in fcs:
        fc.weight.mul_(best_scales.view(1, -1))
    


@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):
    for name, module in model.named_modules():
        if isinstance(module, LlamaDecoderLayer):
            attn_ln = module.input_layernorm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = scales[name + ".mlp.gate_proj"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)

@torch.no_grad()
def awq_lm(model, input_feat, n_grid=20, n_bit=4, q_group_size=128):
    for name, module in model.named_modules():
        if isinstance(module, LlamaDecoderLayer):
            # ===QKV===
            attn_ln = module.input_layernorm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_input = input_feat[name + ".self_attn.q_proj"]
            if qkv_input is not None:
                awq_ln_fcs_llama_like(
                    ln=attn_ln,
                    fcs=qkv,
                    x=qkv_input,
                    block=module.self_attn,
                    kwargs={},
                    n_grid=n_grid,
                    n_bit=n_bit,
                    q_group_size=q_group_size,
                )
            
            # ===FFN===
            ffn_ln = module.post_attention_layernorm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            ffn_input = input_feat[name + ".mlp.gate_proj"]
            if ffn_input is not None:
                awq_ln_fcs_llama_like(
                    ln=ffn_ln,
                    fcs=fcs,
                    x=ffn_input,
                    block=module.mlp,
                    kwargs={},
                    n_grid=n_grid,
                    n_bit=n_bit,
                    q_group_size=q_group_size,
                )
