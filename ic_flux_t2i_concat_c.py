# ic_flux_t2i_concat_c.py  (fixed: no boolean on Tensor)
import torch
from typing import Any, Dict, List

TARGET_TYPES = ("instantcharacter_flux_t2i",)

def _get_tokens_from(opts: Dict[str, Any]):
    if not isinstance(opts, dict): 
        return None
    for key in ("t2i_adapter", "adapters"):
        v = opts.get(key, [])
        if isinstance(v, list):
            for p in v:
                if isinstance(p, dict) and p.get("type") in TARGET_TYPES:
                    t = p.get("tokens", None)
                    if torch.is_tensor(t):
                        return t
    return None

def _align_hidden(tok: torch.Tensor, host_H: int) -> torch.Tensor:
    B, T, Ht = tok.shape
    if Ht == host_H:
        return tok
    if Ht < host_H:
        pad = torch.zeros((B, T, host_H - Ht), device=tok.device, dtype=tok.dtype)
        return torch.cat([tok, pad], dim=-1)
    else:
        return tok[:, :, :host_H]

class ICFluxT2IConcatC:
    CATEGORY = "InstantCharacterFlux"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "scale": ("FLOAT", {"default": 1.0, "min":0.0, "max":8.0, "step":0.05}),
            "debug": ("BOOLEAN", {"default": True}),
        }}

    def apply(self, conditioning, scale, debug):
        if not conditioning:
            return (conditioning,)
        out = []
        for idx, item in enumerate(conditioning):
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                out.append(item); continue

            c, w, *rest = item
            opts = dict(rest[0]) if (len(rest) >= 1 and isinstance(rest[0], dict)) else {}
            rest2 = rest[1:] if (len(rest) >= 1 and isinstance(rest[0], dict)) else rest

            topts = dict(opts.get("transformer_options", {}))
            mopts = dict(opts.get("model_options", {}))

            # ❗不要用 "or" 链接张量
            t1 = _get_tokens_from(topts)
            tok = t1 if t1 is not None else _get_tokens_from(mopts)

            if tok is None or (not torch.is_tensor(c)) or c.dim() != 3:
                out.append(item); continue

            if scale != 1.0:
                tok = tok * scale

            # 对齐 hidden，并迁移到 c 的 device/dtype 再拼接
            tok2 = _align_hidden(tok, c.shape[-1]).to(device=c.device, dtype=c.dtype)
            new_c = torch.cat([c, tok2], dim=1)

            if debug:
                print(f"[IC→t2i][CONCAT-C] item[{idx}] cond: {tuple(c.shape)} + {tuple(tok2.shape)} -> {tuple(new_c.shape)}")

            opts["transformer_options"] = topts
            opts["model_options"] = mopts
            out.append((new_c, w, opts, *rest2))
        return (out,)

NODE_CLASS_MAPPINGS = {"IC_FluxT2IConcatC": ICFluxT2IConcatC}
NODE_DISPLAY_NAME_MAPPINGS = {"IC_FluxT2IConcatC": "IC → FLUX Concat to cond (part[0])"}
