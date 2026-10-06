"""Qwen3-0.6B greedy decoding with the dynamic-persistent megakernel.

Two interchangeable backends run the same task graph:
  * "cuda"    — megakernel_dynamic/cuda  (build with setup.py there)
  * "cutedsl" — megakernel_dynamic/cutedsl (JIT-compiled by CuTeDSL on first use)

Scheduler modes: "auto" (CLC when the GPU and build support it), "clc", "atomic",
and "oneshot" (test only: CLC's launch pattern without work stealing).

Weight formats (weight-only; activations stay fp16), matching CUTLASS's
block-scaled types on Blackwell:
  * "fp16"
  * "fp8" — MXF8: e4m3 values, one e8m0 (power-of-two) scale per 32 along K
  * "fp4" — NVF4: e2m1 values (2 per byte, low nibble first), one e4m3 scale per 16
            along K, plus one fp32 scale per tensor
Norms and the embedding lookup stay fp16; the (tied) LM head gets its own
quantized copy.
"""

import math
import os
import struct
import sys
from collections import namedtuple

import torch

HERE = os.path.dirname(os.path.abspath(__file__))

NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
MAX_SEQ_LEN = 2048
ROPE_THETA = 1_000_000.0
SCHED_MODES = {"auto": 0, "atomic": 1, "clc": 2, "oneshot": 3, "static": 4}
SCHED_NAMES = {1: "atomic", 2: "clc", 3: "oneshot", 4: "static"}
WEIGHT_FORMATS = {"fp16": 0, "fp8": 1, "fp4": 2}

# One projection matrix: data [rows, K] (fp16 | e4m3 bytes | packed e2m1),
# scale [rows, K/block] bytes or None, per-tensor gscale.
QMat = namedtuple("QMat", "data scale gscale")

_NORMS = ("input_layernorm", "q_norm", "k_norm", "post_attention_layernorm")
_MATS = ("q", "k", "v", "o", "gate", "up", "down")
_HF_KEYS = {
    "input_layernorm": "input_layernorm.weight",
    "q_norm": "self_attn.q_norm.weight",
    "k_norm": "self_attn.k_norm.weight",
    "post_attention_layernorm": "post_attention_layernorm.weight",
    "q": "self_attn.q_proj.weight",
    "k": "self_attn.k_proj.weight",
    "v": "self_attn.v_proj.weight",
    "o": "self_attn.o_proj.weight",
    "gate": "mlp.gate_proj.weight",
    "up": "mlp.up_proj.weight",
    "down": "mlp.down_proj.weight",
}


def rope_tables(max_seq: int = MAX_SEQ_LEN):
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    freqs = torch.outer(torch.arange(max_seq, dtype=torch.float32), inv_freq)
    cos = torch.cos(freqs).repeat(1, 2).to(torch.float16).cuda().contiguous()
    sin = torch.sin(freqs).repeat(1, 2).to(torch.float16).cuda().contiguous()
    return cos, sin


def weights_from_model(model) -> dict:
    """fp16 weights of a loaded HF Qwen3 model (storage is shared with the model)."""
    state = model.state_dict()
    layers = []
    for i in range(NUM_LAYERS):
        def get(name):
            return state[f"model.layers.{i}.{_HF_KEYS[name]}"].contiguous()

        layer = {n: get(n) for n in _NORMS}
        layer.update({n: QMat(get(n), None, 1.0) for n in _MATS})
        layers.append(layer)
    embed = state["model.embed_tokens.weight"].contiguous()
    cos, sin = rope_tables()
    return dict(format="fp16", embed=embed, layers=layers, final_norm=state["model.norm.weight"].contiguous(),
                lm_head=QMat(state.get("lm_head.weight", embed).contiguous(), None, 1.0), cos=cos, sin=sin)


def load_weights(model_name: str = "Qwen/Qwen3-0.6B"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return weights_from_model(model), tokenizer, model


# quantization
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_ROW_CHUNK = 8192   # rows quantized at a time (bounds fp32 temporaries for the LM head)


def _quantize_mxfp8_rows(w):
    rows, k = w.shape
    wb = w.float().reshape(rows, k // 32, 32)
    amax = wb.abs().amax(-1, keepdim=True)
    # Smallest power of two that brings the block into e4m3 range (|v| <= 448).
    e = torch.ceil(torch.log2(amax.clamp_min(1e-30) / 448.0)).clamp(-126, 127)
    q = (wb / torch.exp2(e)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.reshape(rows, k).view(torch.uint8), (e + 127).to(torch.uint8).reshape(rows, k // 32)


def _quantize_nvfp4_rows(w, gscale):
    rows, k = w.shape
    wb = w.float().reshape(rows, k // 16, 16)
    bs = (wb.abs().amax(-1, keepdim=True) / (6.0 * gscale)).clamp(max=448.0).to(torch.float8_e4m3fn)
    step = bs.float() * gscale
    x = torch.where(step > 0, wb / step, torch.zeros_like(wb))
    mids = torch.tensor(_E2M1_MIDPOINTS, device=w.device)
    code = torch.bucketize(x.abs().clamp(max=6.0), mids).to(torch.uint8)
    code |= ((x < 0) & (code > 0)).to(torch.uint8) << 3
    code = code.reshape(rows, k)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed.contiguous(), bs.view(torch.uint8).reshape(rows, k // 16)


def quantize_matrix(w: torch.Tensor, fmt: str) -> QMat:
    if fmt == "fp16":
        return QMat(w, None, 1.0)
    rows = w.shape[0]
    if fmt == "fp8":
        parts = [_quantize_mxfp8_rows(w[r:r + _ROW_CHUNK]) for r in range(0, rows, _ROW_CHUNK)]
        gscale = 1.0
    elif fmt == "fp4":
        gscale = float(w.abs().max()) / (448.0 * 6.0) or 1.0   # |max| is exact in fp16, no fp32 copy
        parts = [_quantize_nvfp4_rows(w[r:r + _ROW_CHUNK], gscale) for r in range(0, rows, _ROW_CHUNK)]
    else:
        raise ValueError(f"unknown weight format {fmt!r}")
    return QMat(torch.cat([p[0] for p in parts]).contiguous(), torch.cat([p[1] for p in parts]).contiguous(), gscale)


def dequantize_matrix(m: QMat, fmt: str) -> torch.Tensor:
    """fp32 values the kernel effectively multiplies with (used for reference checks)."""
    if fmt == "fp16":
        return m.data.float()
    rows = m.data.shape[0]
    if fmt == "fp8":
        k = m.data.shape[1]
        q = m.data.view(torch.float8_e4m3fn).float().reshape(rows, k // 32, 32)
        return (q * torch.exp2(m.scale.float() - 127.0)[..., None]).reshape(rows, k)
    k = 2 * m.data.shape[1]
    codes = torch.stack([m.data & 0xF, m.data >> 4], dim=-1).reshape(rows, k).long()
    grid = torch.tensor(_E2M1_VALUES, device=m.data.device)
    vals = grid[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    bs = m.scale.view(torch.float8_e4m3fn).float()[..., None]
    return (vals.reshape(rows, k // 16, 16) * bs * m.gscale).reshape(rows, k)


def quantize_weights(weights: dict, fmt: str) -> dict:
    """Quantize every projection matrix and the LM head; norms/embedding stay fp16."""
    if weights["format"] != "fp16":
        raise ValueError("quantize from fp16 weights")
    if fmt == "fp16":
        return weights
    layers = []
    for layer in weights["layers"]:
        q = {n: layer[n] for n in _NORMS}
        q.update({n: quantize_matrix(layer[n].data, fmt) for n in _MATS})
        layers.append(q)
    out = dict(weights, format=fmt, layers=layers, lm_head=quantize_matrix(weights["lm_head"].data, fmt))
    torch.cuda.empty_cache()
    return out


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def pack_layer_table(layers) -> torch.Tensor:
    """[28 x 25] int64 device table matching QwenDpsLayerWeights (cuda/qwen_dps.h)."""
    slots = []
    for layer in layers:
        for n in _NORMS:
            slots.append(layer[n].data_ptr())
        for n in _MATS:
            m = layer[n]
            for t in (m.data, m.scale):
                assert t is None or t.data_ptr() % 16 == 0, "weights must be 16-byte aligned"
            slots += [m.data.data_ptr(), 0 if m.scale is None else m.scale.data_ptr(), _f32_bits(m.gscale)]
    return torch.tensor(slots, dtype=torch.int64, device="cuda")


def _cuda_ext(trace: bool = False):
    path = os.path.join(HERE, "cuda")
    if path not in sys.path:
        sys.path.insert(0, path)
    if trace:
        import qwen_dps_trace_C

        return qwen_dps_trace_C
    import qwen_dps_C

    return qwen_dps_C


# QwenDpsTraceRecord (cuda/qwen_dps.h): %globaltimer ns per tile, plus SM and CTA.
TRACE_FIELDS = [("claim", "<u8"), ("start", "<u8"), ("ready", "<u8"), ("end", "<u8"),
                ("prep_ns", "<u4"), ("wwait_ns", "<u4"), ("sm", "<u4"), ("cta", "<u4")]
TRACE_RECORD_BYTES = 48


def first_ticket(step: int, n_pre: int, tiles_per_step: int, lm_tiles: int) -> int:
    """Ticket of the first tile of launch-relative token `step`. Tickets run
    [n_pre prefill steps x tiles_per_step] [decode steps x (tiles_per_step + lm_tiles)]."""
    return step * tiles_per_step + max(0, step - n_pre) * lm_tiles


class DpsDecoder:
    """Stateless-per-request decoder: every generate() call is one kernel launch."""

    def __init__(self, weights: dict, tokenizer, backend: str = "cuda", sched: str = "auto",
                 weight_format: str = None, max_seq: int = MAX_SEQ_LEN, trace: bool = False):
        if sched not in SCHED_MODES:
            raise ValueError(f"sched must be one of {list(SCHED_MODES)}")
        weight_format = weight_format or weights["format"]
        if weight_format not in WEIGHT_FORMATS:
            raise ValueError(f"weight_format must be one of {list(WEIGHT_FORMATS)}")
        if weights["format"] != weight_format:
            weights = quantize_weights(weights, weight_format)
        self.tokenizer = tokenizer
        self.backend = backend
        self.sched = sched
        self.weight_format = weight_format
        self.w = weights
        self.layer_table = pack_layer_table(weights["layers"])
        self.attn_scale = 1.0 / math.sqrt(HEAD_DIM)
        self.k_cache = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, max_seq, HEAD_DIM,
                                   dtype=torch.float16, device="cuda")
        self.v_cache = torch.zeros_like(self.k_cache)
        self.last_launch = None   # (scheduler mode used, grid CTAs)
        self.last_trace = None    # (first ticket, uint8 [n, 48] records) after a traced generate
        self._no_scale = torch.empty(0, dtype=torch.uint8, device="cuda")

        if trace and backend != "cuda":
            raise ValueError("tracing needs the CUDA backend")
        if backend == "cuda":
            self.ext = _cuda_ext(trace)
            self.workspace = torch.empty(self.ext.workspace_bytes(), dtype=torch.uint8, device="cuda")
        elif backend == "cutedsl":
            sys.path.insert(0, os.path.join(HERE, "cutedsl"))
            from qwen_dps_cutedsl import CuteDslMegakernel

            self.ext = CuteDslMegakernel(weights, self.layer_table, self.k_cache, self.v_cache, weight_format)
            self.workspace = None
        else:
            raise ValueError("backend must be 'cuda' or 'cutedsl'")

    def info(self) -> dict:
        if self.backend == "cuda":
            return self.ext.info(WEIGHT_FORMATS[self.weight_format])
        return self.ext.info()

    def generate_ids(self, prompt_ids, max_new: int, eos_token_id=None, start_pos: int = 0, trace_steps=None):
        """Greedy-decode up to max_new tokens. Returns generated ids (EOS excluded).

        trace_steps=(first, last): with trace=True, record every tile of launch-relative
        token steps first..last-1 (prefill steps are 0..n_prompt-2) into self.last_trace."""
        n_prompt = len(prompt_ids)
        eos = -1 if eos_token_id is None else int(eos_token_id)
        tokens = torch.empty(n_prompt + max_new, dtype=torch.int32, device="cuda")
        tokens[:n_prompt].copy_(torch.tensor(prompt_ids, dtype=torch.int32))
        out = torch.full((max_new,), -1, dtype=torch.int32, device="cuda")

        if self.backend == "cuda":
            trace, t0 = self._no_scale, 0
            if trace_steps is not None:
                info = self.info()
                if not info["trace_build"]:
                    raise RuntimeError("tracing needs DPS_TRACE=1 python setup.py build_ext --inplace")
                n_pre, s0, s1 = n_prompt - 1, trace_steps[0], min(trace_steps[1], n_prompt - 1 + max_new)
                t0 = first_ticket(s0, n_pre, info["tiles_per_step"], info["lm_tiles"])
                t1 = first_ticket(s1, n_pre, info["tiles_per_step"], info["lm_tiles"])
                trace = torch.zeros(max(t1 - t0, 0), TRACE_RECORD_BYTES, dtype=torch.uint8, device="cuda")
            lm = self.w["lm_head"]
            self.last_launch = self.ext.generate(
                tokens, n_prompt, max_new, start_pos, eos, SCHED_MODES[self.sched],
                WEIGHT_FORMATS[self.weight_format], self.w["embed"], self.layer_table, self.w["final_norm"],
                lm.data, self._no_scale if lm.scale is None else lm.scale, lm.gscale,
                self.w["cos"], self.w["sin"], self.k_cache, self.v_cache, self.workspace, out, self.attn_scale,
                trace, t0)
            if trace_steps is not None:
                self.last_trace = (t0, trace.cpu())
        else:
            self.last_launch = self.ext.generate(tokens, n_prompt, max_new, start_pos, eos,
                                                 SCHED_MODES[self.sched], out)

        ids = []
        for t in out.cpu().tolist():
            if t == -1 or t == eos:
                break
            ids.append(t)
        return ids

    def generate(self, prompt: str, max_tokens: int = 100):
        """Same contract as Model/Qwen06B_architecture.Decoder.generate."""
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        out = self.generate_ids(ids, max_tokens, self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out, skip_special_tokens=True), len(ids), len(out)
