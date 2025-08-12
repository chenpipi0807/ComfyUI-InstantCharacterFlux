# ic_strength_controller.py
# Unified strength controller for InstantCharacter → FLUX t2i
#
# Usage:
#   1) 将 Encode Reference Image (InstantCharacter) 的 gain 设为 1.0
#   2) 本节点接收其输出的 (ic_image_tokens)，用 intensity 统一控制强度
#   3) 输出：
#        - ic_image_tokens_out：已按 gain_out 放大的 tokens（等价于调 gain）
#        - bridge_strength：接到 “IC → FLUX t2i-Adapter Bridge”.strength
#        - concat_scale：接到 “IC → FLUX Concat to cond”.scale
#
# 你仍可用“进阶因子”单独微调三路（gain / bridge / scale）的映射。
# 默认线性：gain_out = base_gain * intensity，其他同理。

from typing import Tuple, Dict, Any
import torch

class ICStrengthControllerNode:
    CATEGORY = "InstantCharacterFlux"
    RETURN_TYPES = ("IC_IMAGE_TOKENS", "FLOAT", "FLOAT")
    RETURN_NAMES = ("ic_image_tokens_out", "bridge_strength", "concat_scale")
    FUNCTION = "apply"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ic_image_tokens": ("IC_IMAGE_TOKENS",),
                # 统一强度滑块：建议 0.0 ~ 2.0；也可更大
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
            },
            "optional": {
                # 三路的 base 值（相当于“1 倍强度时”的标定）
                "base_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "base_bridge_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "base_concat_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01}),

                # 进阶：各路的倍率系数（非线性/更灵敏可把 k > 1，或 < 1 变钝）
                "gain_k": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.01}),
                "bridge_k": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.01}),
                "scale_k": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.01}),

                # 限幅，避免极端值导致不稳定
                "max_gain": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 10.0, "step": 0.1}),
                "max_bridge_strength": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 10.0, "step": 0.1}),
                "max_concat_scale": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 10.0, "step": 0.1}),

                # 调试打印
                "debug": ("BOOLEAN", {"default": True}),
            },
        }

    def _calc(self, base: float, inten: float, k: float, clamp_max: float) -> float:
        # 简单的可调敏感度映射： out = base * (inten ** k)
        val = float(base) * (float(inten) ** float(k))
        if clamp_max is not None:
            val = max(0.0, min(val, float(clamp_max)))
        return val

    def apply(
        self,
        ic_image_tokens,
        intensity: float,
        base_gain: float = 1.0,
        base_bridge_strength: float = 1.0,
        base_concat_scale: float = 1.0,
        gain_k: float = 1.0,
        bridge_k: float = 1.0,
        scale_k: float = 1.0,
        max_gain: float = 3.0,
        max_bridge_strength: float = 3.0,
        max_concat_scale: float = 3.0,
        debug: bool = True,
    ):
        tokens, meta = ic_image_tokens  # tokens: Tensor[B, T, H]
        assert isinstance(meta, dict), "IC Strength Controller expects (tokens, meta)."

        # 计算三路输出
        gain_out = self._calc(base_gain, intensity, gain_k, max_gain)
        bridge_strength = self._calc(base_bridge_strength, intensity, bridge_k, max_bridge_strength)
        concat_scale = self._calc(base_concat_scale, intensity, scale_k, max_concat_scale)

        # 对 tokens 等价应用“gain”
        if not torch.is_tensor(tokens):
            raise RuntimeError("ic_image_tokens[0] must be a Tensor.")
        tokens_out = tokens * gain_out  # 不原地，避免影响上游缓存

        if debug:
            h = int(tokens.shape[-1]) if tokens.ndim == 3 else -1
            print(f"[IC Strength] intensity={intensity:.3f}  -> gain={gain_out:.3f}, bridge={bridge_strength:.3f}, scale={concat_scale:.3f} (H={h})")

        return ((tokens_out, meta), bridge_strength, concat_scale)


# ---- register ----
NODE_CLASS_MAPPINGS = {
    "ICStrengthController": ICStrengthControllerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ICStrengthController": "IC Strength Controller (InstantCharacter → FLUX)",
}
