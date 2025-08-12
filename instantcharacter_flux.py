# instantcharacter_flux.py
# ComfyUI-InstantCharacterFlux (bin-deepflatten + dual-inject)
import os, math
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.torch import load_file as safetensors_load
    _HAS_ST = True
except Exception:
    _HAS_ST = False

try:
    from transformers import AutoImageProcessor, AutoModel, AutoConfig
    _HAS_TXF = True
except Exception:
    _HAS_TXF = False


# ---------- paths ----------
def _models_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    comfy_root = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.join(comfy_root, "models")

def _ic_root() -> str:
    return os.path.join(_models_root(), "instantCharacter")

def _default_ic_weights() -> str:
    return os.path.join(_ic_root(), "instantcharacter_ip-adapter.safetensors")

def _abs_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_models_root(), os.path.normpath(p))

def _ensure_dir(p: str) -> str:
    if not os.path.isdir(p):
        raise FileNotFoundError(p)
    return p

def _empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------- modules ----------
class Adapter2Layer(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_tokens: int, mid: Optional[int] = None):
        super().__init__()
        mid = mid or max(in_dim, hidden)
        self.n_tokens = n_tokens
        self.hidden = hidden
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_tokens * hidden),
        )
    def forward(self, x):
        y = self.proj(x)
        return y.view(x.shape[0], self.n_tokens, self.hidden)


class VisionEncoder(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        if not _HAS_TXF:
            raise RuntimeError("transformers not available.")
        self.proc = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
        self.model.eval()
        if hasattr(cfg, "vision_config") and hasattr(cfg.vision_config, "hidden_size"):
            self.out_dim = cfg.vision_config.hidden_size
        elif hasattr(cfg, "hidden_size"):
            self.out_dim = cfg.hidden_size
        else:
            self.out_dim = 1024

    @torch.no_grad()
    def forward(self, image_bhwc: torch.Tensor, device: torch.device) -> torch.Tensor:
        x = (image_bhwc.clamp(0, 1).cpu()).permute(0, 3, 1, 2)
        inputs = self.proc(images=x, return_tensors="pt", do_rescale=False)
        pixel_values = inputs["pixel_values"].to(device=device)
        if hasattr(self.model, "vision_model"):   # SigLIP
            out = self.model.vision_model(pixel_values=pixel_values)
        else:                                     # DINOv2
            out = self.model(pixel_values=pixel_values)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            pooled = out.last_hidden_state.mean(dim=1)
        else:
            raise RuntimeError("vision model outputs not supported.")
        return pooled


# ---------- weight loader ----------
def _flatten_state(obj, prefix="", out=None):
    """Deep-flatten nested dict/list/nn.Modules to flat dict with dot keys."""
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten_state(v, f"{prefix}{k}." if prefix else f"{k}.", out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flatten_state(v, f"{prefix}{i}.", out)
    elif torch.is_tensor(obj):
        out[prefix[:-1]] = obj
    else:
        # try state_dict() for modules / containers
        if hasattr(obj, "state_dict"):
            sd = obj.state_dict()
            for k, v in sd.items():
                out[(prefix + k)] = v
        # else ignore
    return out


class ICWeights(nn.Module):
    def __init__(self, weights_path: str, default_tokens=128, default_hidden=3072):
        super().__init__()
        ext = os.path.splitext(weights_path)[1].lower()

        if ext == ".safetensors":
            if not _HAS_ST:
                raise RuntimeError("Please install safetensors.")
            flat = safetensors_load(weights_path, device="cpu")
            self.state = dict(flat)
        else:
            raw = torch.load(weights_path, map_location="cpu")  # nested dict
            self.state = _flatten_state(raw)

        # meta (optional)
        self.image_tokens = int(self.state.get("meta.image_tokens", torch.tensor(default_tokens)).item()
                                if isinstance(self.state.get("meta.image_tokens", None), torch.Tensor) else default_tokens)
        self.hidden = int(self.state.get("meta.hidden", torch.tensor(default_hidden)).item()
                          if isinstance(self.state.get("meta.hidden", None), torch.Tensor) else default_hidden)



        self.blocks_single = list(range(38))
        self.blocks_double = list(range(19))

    def load_proj_into(self, mlp: Adapter2Layer, prefix: str) -> int:
        sd = mlp.state_dict()
        want = {
            "proj.0.weight": sd["proj.0.weight"].shape,
            "proj.0.bias":   sd["proj.0.bias"].shape,
            "proj.2.weight": sd["proj.2.weight"].shape,
            "proj.2.bias":   sd["proj.2.bias"].shape,
        }
        loaded = 0
        for k in want.keys():
            kk = f"{prefix}{k}"
            if kk in self.state and tuple(self.state[kk].shape) == tuple(want[k]):
                sd[k] = self.state[kk]
                loaded += 1
                print(f"[IC][HIT] {k:<16} <=  {kk}")
        if loaded > 0:
            mlp.load_state_dict(sd, strict=False)
        return loaded

    def reconstruct_into(self, mlp: Adapter2Layer, in_dim: int, tokens: int, hidden: int) -> int:
        """
        Build a 2-layer head with robust shape handling:
          Layer1(in_dim->hidden): seed from ip_adapter.*.to_k_ip.* if possible
          - Handles row/col mismatch between src (e.g., 3072x4096) and target (hidden x in_dim, e.g., 4096x2688)
          Layer2(hidden->tokens*hidden): block-diagonal repeat-identity (+tiny noise)
        """
        seeded = 0
        sd = mlp.state_dict()

        # choose a source block that exists
        src_k = None
        for i in range(128):
            if f"ip_adapter.{i}.to_k_ip.weight" in self.state:
                src_k = i
                break

        W1 = sd["proj.0.weight"]   # (hidden, in_dim)
        b1 = sd["proj.0.bias"]     # (hidden,)

        if src_k is not None:
            k_w = f"ip_adapter.{src_k}.to_k_ip.weight"   # typically (3072, 4096)
            k_b = f"ip_adapter.{src_k}.to_k_ip.bias"     # typically (3072,)
            src_w = self.state[k_w]
            src_rows, src_cols = src_w.shape  # e.g., (3072, 4096)

            # ---- cols (-> in_dim) ----
            copy_cols = min(src_cols, in_dim)
            # init target first to a good default
            nn.init.kaiming_uniform_(W1, a=math.sqrt(5))
            with torch.no_grad():
                # ---- rows (-> hidden) ----
                if src_rows >= hidden:
                    # truncate rows
                    W1[:, :copy_cols].copy_(src_w[:hidden, :copy_cols])
                    row_info = f"rows=truncate({hidden}/{src_rows})"
                else:
                    # pad rows
                    W1[:src_rows, :copy_cols].copy_(src_w[:, :copy_cols])
                    # remaining rows stay as kaiming init
                    row_info = f"rows=pad({src_rows}->{hidden})"
            col_info = f"cols={copy_cols}/{in_dim} (src_cols={src_cols})"
            print(f"[IC][HIT] seed proj.0.weight from {k_w}  {row_info}, {col_info}")

            # bias
            if k_b in self.state and self.state[k_b].dim() == 1:
                src_b = self.state[k_b]
                with torch.no_grad():
                    if src_b.shape[0] >= hidden:
                        b1.copy_(src_b[:hidden])
                        binfo = f"bias=truncate({hidden}/{src_b.shape[0]})"
                    else:
                        b1.zero_()
                        b1[:src_b.shape[0]].copy_(src_b)
                        binfo = f"bias=pad({src_b.shape[0]}->{hidden})"
                print(f"[IC][HIT] seed proj.0.bias   from {k_b}  {binfo}")
            else:
                fan_in = in_dim
                bound = 1 / max(1, math.sqrt(fan_in))
                nn.init.uniform_(b1, -bound, bound)
                print(f"[IC][MISS] {k_b} -> uniform proj.0.bias")

            seeded += 2
        else:
            # no suitable source — fall back
            nn.init.kaiming_uniform_(W1, a=math.sqrt(5))
            fan_in = in_dim
            bound = 1 / max(1, math.sqrt(fan_in))
            nn.init.uniform_(b1, -bound, bound)
            print("[IC][MISS] to_k_ip.* -> kaiming/uniform for proj.0.*")

        # Layer2: block-diagonal repeat-identity
        W2 = sd["proj.2.weight"]   # (tokens*hidden, hidden)
        b2 = sd["proj.2.bias"]     # (tokens*hidden,)
        with torch.no_grad():
            W2.zero_()
            eye = torch.eye(hidden, dtype=W2.dtype)
            for t in range(tokens):
                r0 = t * hidden
                W2[r0:r0+hidden, :].copy_(eye)
            W2.add_(torch.randn_like(W2) * 1e-3)
            b2.zero_()
        print(f"[IC][HEAD] proj.2.weight <- block-diag repeat(I_{hidden}) x {tokens} (+tiny noise)")

        mlp.load_state_dict(sd, strict=False)
        seeded += 1
        return seeded


# ---------- helpers ----------
def _safe_forward(mlp: nn.Module, x: torch.Tensor, device, dtype) -> torch.Tensor:
    try:
        mlp = mlp.to(device=device, dtype=dtype)
        return mlp(x.to(device=device, dtype=dtype))
    except RuntimeError:
        _empty_cache()
        mlp = mlp.to(device="cpu", dtype=dtype)
        y = mlp(x.to(device="cpu", dtype=dtype))
        return y.to(device=device, dtype=dtype)


# ---------- nodes ----------
class LoadICWeightsNode:
    CATEGORY = "InstantCharacterFlux"
    RETURN_TYPES = ("IC_WEIGHTS",)
    RETURN_NAMES = ("ic_weights",)
    FUNCTION = "load"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"weights_path": ("STRING", {"default": _default_ic_weights()})}}

    def load(self, weights_path: str):
        p = _abs_path(weights_path)
        if not os.path.isfile(p):
            p = _default_ic_weights()
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        return (ICWeights(p),)


class LoadSigLIPVisionNode:
    CATEGORY = "InstantCharacterFlux"
    RETURN_TYPES = ("VISION_ENCODER",)
    RETURN_NAMES = ("siglip_encoder",)
    FUNCTION = "load"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_dir": ("STRING", {"default": os.path.join("instantCharacter", "siglip-so400m-patch14-384")})}}

    def load(self, model_dir: str):
        p = _abs_path(model_dir)
        enc = VisionEncoder(_ensure_dir(p))
        return (enc,)


class LoadDINOv2VisionNode:
    CATEGORY = "InstantCharacterFlux"
    RETURN_TYPES = ("VISION_ENCODER",)
    RETURN_NAMES = ("dino_encoder",)
    FUNCTION = "load"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_dir": ("STRING", {"default": os.path.join("instantCharacter", "dinov2-giant")})}}

    def load(self, model_dir: str):
        p = _abs_path(model_dir)
        enc = VisionEncoder(_ensure_dir(p))
        return (enc,)


class EncodeRefImageICNode:
    CATEGORY = "InstantCharacterFlux"
    RETURN_TYPES = ("IC_IMAGE_TOKENS",)
    RETURN_NAMES = ("ic_image_tokens",)
    FUNCTION = "encode"
    OUTPUT_NODE = False

    _mlp_cache: dict = {}

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "ic_weights": ("IC_WEIGHTS",),
            "siglip_encoder": ("VISION_ENCODER",),
            "dino_encoder": ("VISION_ENCODER",),
        }}

    def _get_or_build(self, in_dim: int, hidden: int, n_tokens: int, icw: ICWeights, device, dtype) -> Tuple[Adapter2Layer, int, str, str]:
        key = (in_dim, hidden, n_tokens, id(icw))
        mlp = self._mlp_cache.get(key)
        loaded = 0
        mode = "explicit"
        seeded_from = "none"
        if mlp is None:
            mlp = Adapter2Layer(in_dim, hidden, n_tokens)
            for pref in ("proj_fused.", "proj_siglip.", "proj_dino."):
                loaded = icw.load_proj_into(mlp, pref)
                if loaded >= 2:
                    seeded_from = pref[:-1]
                    break
            if loaded < 2:
                mode = "reconstruct"
                seeded = icw.reconstruct_into(mlp, in_dim, n_tokens, hidden)
                seeded_from = "ip_adapter.to_k_ip" if seeded >= 2 else "kaiming"
                print(f"[InstantCharacterFlux][RECON] seeded_params={seeded}")
            self._mlp_cache[key] = mlp
        try:
            mlp = mlp.to(device=device, dtype=dtype)
        except RuntimeError:
            mlp = mlp.to(device="cpu", dtype=dtype)
        return mlp, loaded, mode, seeded_from

    def encode(self, image, ic_weights: ICWeights, siglip_encoder: VisionEncoder, dino_encoder: VisionEncoder):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        siglip_encoder = siglip_encoder.to(device) if siglip_encoder is not None else None
        dino_encoder = dino_encoder.to(device) if dino_encoder is not None else None
        assert (siglip_encoder is not None) or (dino_encoder is not None), "需至少连接一个视觉编码器"

        with torch.no_grad():
            pooled_s = siglip_encoder(image, device=device) if siglip_encoder is not None else None
            pooled_d = dino_encoder(image, device=device) if dino_encoder is not None else None
            base = pooled_s if pooled_s is not None else pooled_d
            base_dtype = base.dtype

            if pooled_s is not None and pooled_d is not None:
                concat = torch.cat([pooled_s, pooled_d], dim=-1)
                mlp, loaded, mode, seeded_from = self._get_or_build(concat.shape[-1], ic_weights.hidden, ic_weights.image_tokens, ic_weights, device, base_dtype)
                print(f"[InstantCharacterFlux][INFO] encoders: S={pooled_s.shape[-1]}, D={pooled_d.shape[-1]}; hidden={ic_weights.hidden}, tokens={ic_weights.image_tokens}; adapter_mode={mode} loaded_params={loaded}")
                tokens = _safe_forward(mlp, concat, device, base_dtype)
            elif pooled_s is not None:
                mlp, loaded, mode, seeded_from = self._get_or_build(pooled_s.shape[-1], ic_weights.hidden, ic_weights.image_tokens, ic_weights, device, base_dtype)
                print(f"[InstantCharacterFlux][INFO] encoder: S={pooled_s.shape[-1]}; hidden={ic_weights.hidden}, tokens={ic_weights.image_tokens}; adapter_mode={mode} loaded_params={loaded}")
                tokens = _safe_forward(mlp, pooled_s, device, base_dtype)
            else:
                mlp, loaded, mode, seeded_from = self._get_or_build(pooled_d.shape[-1], ic_weights.hidden, ic_weights.image_tokens, ic_weights, device, base_dtype)
                print(f"[InstantCharacterFlux][INFO] encoder: D={pooled_d.shape[-1]}; hidden={ic_weights.hidden}, tokens={ic_weights.image_tokens}; adapter_mode={mode} loaded_params={loaded}")
                tokens = _safe_forward(mlp, pooled_d, device, base_dtype)

            tokens = F.normalize(tokens, dim=-1)
            # gain 固定为 1.0，由后续强度控制节点统一管理
            # quick visibility
            print(f"[InstantCharacterFlux][DEBUG] seed_from={seeded_from}  token[0] L2-norm: {tokens[0,0].norm(p=2).item():.4f}")

            meta = {"blocks_single": ic_weights.blocks_single, "blocks_double": ic_weights.blocks_double, "hidden": ic_weights.hidden}
        print(f"[InstantCharacterFlux][INFO] tokens shape: {tuple(tokens.shape)} (T={ic_weights.image_tokens}, H={ic_weights.hidden})")
        for m in self._mlp_cache.values():
            m.to(device="cpu")
        _empty_cache()
        return ((tokens, meta),)





# ---------- register ----------
NODE_CLASS_MAPPINGS = {
    "LoadICWeights": LoadICWeightsNode,
    "LoadSigLIPVision": LoadSigLIPVisionNode,
    "LoadDINOv2Vision": LoadDINOv2VisionNode,
    "EncodeRefImageIC": EncodeRefImageICNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadICWeights": "Load IC Weights",
    "LoadSigLIPVision": "Load SigLIP Vision",
    "LoadDINOv2Vision": "Load DINOv2 Vision",
    "EncodeRefImageIC": "Encode Reference Image (InstantCharacter)",
}
