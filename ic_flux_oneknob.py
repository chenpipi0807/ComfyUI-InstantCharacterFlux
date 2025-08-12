# ic_flux_oneknob.py
# One-knob bridge for InstantCharacter → FLUX
# - 注入 t2i-adapter 载荷
# - 同时把（按强度放大后的）IC tokens 拼到 cond part[0]
# 仅 3 个控制：intensity, start_percent, end_percent

import torch
from typing import Any, Dict, List, Tuple

def _align_hidden(tok: torch.Tensor, host_H: int) -> torch.Tensor:
    # 对齐 hidden 维以便拼接
    B, T, Ht = tok.shape
    if Ht == host_H:
        return tok
    if Ht < host_H:
        pad = torch.zeros((B, T, host_H - Ht), device=tok.device, dtype=tok.dtype)
        return torch.cat([tok, pad], dim=-1)
    else:
        return tok[:, :, :host_H]

def _inject_list(host: Dict[str, Any], key: str, payload: Dict[str, Any], dedupe=True):
    lst = host.get(key, [])
    if not isinstance(lst, list):
        lst = []
    if dedupe:
        lst = [p for p in lst if not (isinstance(p, dict) and p.get("type") == "instantcharacter_flux_t2i")]
    lst.append(payload)
    host[key] = lst
    return lst

class ICFluxOneKnob:
    CATEGORY = "InstantCharacterFlux"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "ic_image_tokens": ("IC_IMAGE_TOKENS",),
            "intensity": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 3.0, "step": 0.05}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        }, "optional": {
            "debug": ("BOOLEAN", {"default": True}),
        }}

    def apply(self,
              conditioning,
              ic_image_tokens,
              intensity: float,
              start_percent: float,
              end_percent: float,
              debug: bool = True):

        if not conditioning:
            return (conditioning,)

        tokens, meta = ic_image_tokens   # tokens: [B, T, H]
        if not torch.is_tensor(tokens):
            if debug: print("[IC One-Knob][ERR] tokens is not Tensor; skip.")
            return (conditioning,)

        # —— 统一强度策略 ——
        # 1) t2i-adapter 的 strength = intensity
        # 2) concat 时把 tokens 放大同样的 intensity（效果更直观）
        strength = float(max(0.0, intensity))
        tok_for_concat = tokens * strength

        # 构造 t2i 载荷（不复制大 tensor，直接引用）
        payload = {
            "type": "instantcharacter_flux_t2i",
            "tokens": tokens,   # 原 tokens（未放大），避免额外显存
            "hidden": int(meta.get("hidden", tokens.shape[-1])),
            "tokens_count": int(tokens.shape[1]),
            "strength": strength,
            "start": float(start_percent),
            "end": float(end_percent),
        }
        # 把 blocks 从 meta 透传（有就用，没有就让下游默认）
        if isinstance(meta, dict):
            if "blocks_single" in meta: payload["blocks_single"] = list(meta["blocks_single"])
            if "blocks_double" in meta: payload["blocks_double"] = list(meta["blocks_double"])

        out = []
        for idx, item in enumerate(conditioning):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                out.append(item); continue

            c, w, *rest = item

            # 找到 cond Tensor
            if torch.is_tensor(c):
                cond_tensor = c
                carried_opts = None
            elif isinstance(c, dict) and torch.is_tensor(c.get("cond", None)):
                cond_tensor = c["cond"]
                carried_opts = {
                    "transformer_options": dict(c.get("transformer_options", {})) if isinstance(c.get("transformer_options", {}), dict) else {},
                    "model_options": dict(c.get("model_options", {})) if isinstance(c.get("model_options", {}), dict) else {},
                }
            else:
                out.append(item)
                if debug: print(f"[IC One-Knob][SKIP] item[{idx}] no Tensor cond; untouched.")
                continue

            # 第3位 options dict
            if len(rest) >= 1 and isinstance(rest[0], dict):
                opts = dict(rest[0]); rest2 = rest[1:]
            else:
                opts = {}; rest2 = rest

            topts = dict(opts.get("transformer_options", {}))
            mopts = dict(opts.get("model_options", {}))

            # 合并 c 里携带的（若有）
            if isinstance(carried_opts, dict):
                for k, v in carried_opts.get("transformer_options", {}).items():
                    topts.setdefault(k, v)
                for k, v in carried_opts.get("model_options", {}).items():
                    mopts.setdefault(k, v)

            # 写入 t2i_adapter（并做去重）；顺带写 adapters 兼容不同实现
            bt = len(topts.get("t2i_adapter", [])) if isinstance(topts.get("t2i_adapter", []), list) else 0
            bm = len(mopts.get("t2i_adapter", [])) if isinstance(mopts.get("t2i_adapter", []), list) else 0
            at = len(_inject_list(topts, "t2i_adapter", payload, dedupe=True))
            am = len(_inject_list(mopts, "t2i_adapter", payload, dedupe=True))
            _inject_list(topts, "adapters", payload, dedupe=True)
            _inject_list(mopts, "adapters", payload, dedupe=True)

            # 拼接：把按强度放大的 tok_for_concat 贴到 cond part[0]
            tok2 = _align_hidden(tok_for_concat, cond_tensor.shape[-1]).to(
                device=cond_tensor.device, dtype=cond_tensor.dtype)
            new_cond = torch.cat([cond_tensor, tok2], dim=1)

            if debug:
                print(f"[IC One-Knob] item[{idx}] "
                      f"inject strength={strength:.3f} (t2i: t {bt}->{at} | m {bm}->{am}); "
                      f"concat: {tuple(cond_tensor.shape)} + {tuple(tok2.shape)} -> {tuple(new_cond.shape)}")

            opts["transformer_options"] = topts
            opts["model_options"] = mopts
            out.append((new_cond, w, opts, *rest2))

        return (out,)

# —— register ——
NODE_CLASS_MAPPINGS = {"ICFluxOneKnob": ICFluxOneKnob}
NODE_DISPLAY_NAME_MAPPINGS = {"ICFluxOneKnob": "IC → FLUX One-Knob"}
