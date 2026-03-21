"""Weight loading and high-level decode API for Qwen3-0.6B."""

import math
import torch
import os

try:
    import qwen_megakernel_C
except ImportError:
    qwen_megakernel_C = None

EXTENSION_ABI_VERSION = 2

NUM_LAYERS      = 28
NUM_KV_HEADS    = 8
HEAD_DIM        = 128
HIDDEN_SIZE     = 1024
INTERMEDIATE_SIZE = 3072
Q_SIZE          = 16 * HEAD_DIM   # 2048
KV_SIZE         = 8  * HEAD_DIM   # 1024
MAX_SEQ_LEN     = 2048
VOCAB_SIZE      = 151936

# FIX A: match the compile-time constant in megakernel_5090.cu so buffer
#         sizes are derived from a single source of truth.
LDG_LM_NUM_BLOCKS = 680

ROPE_THETA = 1_000_000.0


def _require_megakernel_op(op_name: str):
    """Return an op from the extension module or torch.ops namespace."""
    if qwen_megakernel_C is not None and hasattr(qwen_megakernel_C, op_name):
        return getattr(qwen_megakernel_C, op_name)

    namespace = getattr(torch.ops, "qwen_megakernel_C", None)
    if namespace is not None and hasattr(namespace, op_name):
        return getattr(namespace, op_name)

    available_ops = []
    if qwen_megakernel_C is not None:
        available_ops.extend(
            op for op in ("decode", "generate_nosync", "prefill")
            if hasattr(qwen_megakernel_C, op)
        )
    if namespace is not None:
        available_ops.extend(
            op for op in ("decode", "generate_nosync", "prefill")
            if hasattr(namespace, op)
        )

    available_ops_text = ", ".join(sorted(set(available_ops))) or "none"
    raise RuntimeError(
        f"qwen_megakernel_C op '{op_name}' is unavailable. "
        "The C++/CUDA extension is not loaded for this PyTorch build. "
        "Rebuild/install the megakernel extension against your current "
        f"torch version ({torch.__version__}). "
        f"Available extension ops: {available_ops_text}."
    )


def _assert_extension_compatibility() -> None:
    """Fail fast on stale extension builds that can crash with CUDA misalignment."""
    if qwen_megakernel_C is None:
        return

    ext_abi = None
    if hasattr(qwen_megakernel_C, "abi_version"):
        try:
            ext_abi = int(qwen_megakernel_C.abi_version())
        except Exception:
            ext_abi = None

    if ext_abi != EXTENSION_ABI_VERSION:
        built_torch = "unknown"
        if hasattr(qwen_megakernel_C, "built_torch_version"):
            try:
                built_torch = str(qwen_megakernel_C.built_torch_version())
            except Exception:
                built_torch = "unknown"

        raise RuntimeError(
            "Incompatible qwen_megakernel_C extension binary detected. "
            f"Expected ABI {EXTENSION_ABI_VERSION}, got {ext_abi}. "
            "This usually means Python code and CUDA extension are out-of-sync "
            "(e.g. pointer packing changed) and can trigger CUDA 'misaligned address'. "
            f"Rebuild/reinstall qwen_megakernel_C for torch {torch.__version__} "
            f"(extension was built against {built_torch})."
        )


_assert_extension_compatibility()
_decode = _require_megakernel_op("decode")


def load_weights(model_name="Qwen/Qwen3-0.6B", verbose: bool = True):
    """Load Qwen3-0.6B weights from HuggingFace into GPU tensors."""
    if not verbose:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    if not verbose:
        hf_logging.set_verbosity_error()
        try:
            hf_logging.disable_progress_bar()
        except AttributeError:
            pass
        try:
            from huggingface_hub import logging as hf_hub_logging
            hf_hub_logging.set_verbosity_error()
        except Exception:
            pass

    if verbose:
        print(f"Loading {model_name}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    state = model.state_dict()

    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
    freqs     = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.float16).cuda().contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.float16).cuda().contiguous()

    layer_weights = []
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}."
        layer_weights.extend([
            state[p + "input_layernorm.weight"].contiguous(),
            state[p + "self_attn.q_proj.weight"].contiguous(),
            state[p + "self_attn.k_proj.weight"].contiguous(),
            state[p + "self_attn.v_proj.weight"].contiguous(),
            state[p + "self_attn.q_norm.weight"].contiguous(),
            state[p + "self_attn.k_norm.weight"].contiguous(),
            state[p + "self_attn.o_proj.weight"].contiguous(),
            state[p + "post_attention_layernorm.weight"].contiguous(),
            state[p + "mlp.gate_proj.weight"].contiguous(),
            state[p + "mlp.up_proj.weight"].contiguous(),
            state[p + "mlp.down_proj.weight"].contiguous(),
        ])

    embed_weight = state["model.embed_tokens.weight"].contiguous()
    weights = dict(
        embed_weight=embed_weight,
        layer_weights=layer_weights,
        final_norm_weight=state["model.norm.weight"].contiguous(),
        lm_head_weight=embed_weight,   # tied embeddings
        cos_table=cos_table,
        sin_table=sin_table,
    )

    del model
    torch.cuda.empty_cache()
    return weights, tokenizer


def _pack_layer_weights(layer_weights: list) -> torch.Tensor:
    N        = 11
    n_layers = len(layer_weights) // N
    assert len(layer_weights) == n_layers * N

    all_ptrs = []
    for li in range(n_layers):
        base = li * N
        for j in range(N):
            all_ptrs.append(layer_weights[base + j].data_ptr())
        all_ptrs.append(0)   # padding → 12 pointers × 8 bytes = 96 bytes, 16-byte aligned

    t = torch.zeros(len(all_ptrs) + 2, dtype=torch.int64, device="cuda")
    base_ptr     = t.data_ptr()
    offset       = (16 - (base_ptr % 16)) % 16
    offset_elems = offset // 8
    t_aligned    = t[offset_elems : offset_elems + len(all_ptrs)]
    t_aligned.copy_(torch.tensor(all_ptrs, dtype=torch.int64))
    assert t_aligned.data_ptr() % 16 == 0, "Alignment failed!"
    return t_aligned


class Decoder:
    """Stateful decoder wrapping the Qwen megakernel ops."""

    def __init__(
        self,
        weights=None,
        tokenizer=None,
        model_name="Qwen/Qwen3-0.6B",
        verbose: bool = True,
    ):
        if weights is None:
            weights, tokenizer = load_weights(model_name, verbose=verbose)

        self.tokenizer              = tokenizer
        self._position              = 0
        self._weights               = weights

        self._embed_weight          = weights["embed_weight"]
        self._final_norm_weight     = weights["final_norm_weight"]
        self._lm_head_weight        = weights["lm_head_weight"]
        self._cos_table             = weights["cos_table"]
        self._sin_table             = weights["sin_table"]
        self._layer_weights_packed  = _pack_layer_weights(weights["layer_weights"])
        self._check_weight_alignment(weights["layer_weights"])
        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)

        # KV cache — allocate once; partial reset via high-water mark avoids
        # zeroing the full 235 MB buffer every request.
        self._k_cache       = torch.zeros(
            NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.float16, device="cuda",
        )
        self._v_cache       = torch.zeros_like(self._k_cache)
        self._kv_high_water = 0

        # Scratch buffers — sizes derived from model constants above.
        f16 = dict(dtype=torch.float16, device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")
        i32 = dict(dtype=torch.int32,   device="cuda")

        self._hidden    = torch.empty(HIDDEN_SIZE,        **f16)

        # g_activations is indexed up to HIDDEN_SIZE=1024, not Q_SIZE=2048. Previous allocation (Q_SIZE) was a 2× overallocation.
        self._act       = torch.empty(HIDDEN_SIZE,        **f32)

        # g_residual is accepted by the kernel for ABI compatibility but is no
        # longer read or written — the fixed kernel reads the half* residual
        # directly from hidden_buffer.  Kept here so qwen_ops.cpp compiles
        # without changes.
        self._res       = torch.empty(HIDDEN_SIZE,        **f32)

        self._q         = torch.empty(Q_SIZE,             **f16)
        self._k         = torch.empty(KV_SIZE,            **f16)
        self._v         = torch.empty(KV_SIZE,            **f16)
        self._attn_out  = torch.empty(Q_SIZE,             **f16)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE,  **f32)
        self._norm_out  = torch.empty(HIDDEN_SIZE,        **f32)

        # size LM-head reduction buffers from the compile-time constant, not a magic number.  LDG_LM_NUM_BLOCKS=680 in megakernel_5090.cu; 4096 was a 6× overallocation.
        self._fmax_vals = torch.empty(LDG_LM_NUM_BLOCKS,  **f32)
        self._fmax_idxs = torch.empty(LDG_LM_NUM_BLOCKS,  **i32)

        self._out_token = torch.empty(1,                  **i32)

        # Pinned output log: kernel writes token IDs on-device; single DtoH
        # copy at the end of generation.
        self._output_log_cpu = torch.empty(MAX_SEQ_LEN, dtype=torch.int32).pin_memory()
        self._output_log     = self._output_log_cpu.cuda()

        self._gen = _require_megakernel_op("generate_nosync")
        try:
            self._prefill_op = _require_megakernel_op("prefill")
        except RuntimeError:
            self._prefill_op = None

    # Argument packers

    def _prefill_args(self, token_ids_tensor):
        """Pack arguments for the batched prefill op."""
        return (
            token_ids_tensor,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._fmax_vals,
            self._fmax_idxs,
            NUM_LAYERS,
            self._position,   # start_position
            MAX_SEQ_LEN,
            self._attn_scale,
        )

    def _gen_args(self, first_token, num_steps, position, output_log=None):
        """Pack arguments for generate_nosync."""
        if output_log is None:
            output_log = self._output_log
        return (
            first_token, num_steps,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._fmax_vals,
            self._fmax_idxs,
            output_log,
            NUM_LAYERS,
            position,
            MAX_SEQ_LEN,
            self._attn_scale,
            self.tokenizer.eos_token_id,
        )

    # Weight alignment check

    def _check_weight_alignment(self, layer_weights_list):
        N     = 11
        names = [
            "input_layernorm", "q_proj", "k_proj", "v_proj",
            "q_norm", "k_norm", "o_proj", "post_attn_norm",
            "gate_proj", "up_proj", "down_proj",
        ]
        bad = []
        for li in range(NUM_LAYERS):
            for j in range(N):
                ptr = layer_weights_list[li * N + j].data_ptr()
                if ptr % 16 != 0:
                    bad.append(f"  Layer {li} [{names[j]}]: ptr={ptr:#x}, rem={ptr%16}")
        if bad:
            print("MISALIGNED WEIGHT TENSORS:")
            for b in bad:
                print(b)
        else:
            print("All weight pointers 16-byte aligned OK")

    # Single-step decode (used for testing / interactive use)

    def step(self, input_token_id: int) -> int:
        result_tensor = _decode(
            input_token_id,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._fmax_vals,
            self._fmax_idxs,
            NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        torch.cuda.synchronize()
        self._position += 1
        return int(result_tensor.item())

    # KV cache reset

    def reset(self):
        self._position = 0
        if self._kv_high_water > 0:
            hw = self._kv_high_water
            self._k_cache[:, :, :hw, :].zero_()
            self._v_cache[:, :, :hw, :].zero_()
        self._kv_high_water = 0

    @property
    def position(self) -> int:
        return self._position

    # Full generation 

    def generate(self, prompt: str, max_tokens: int = 100) -> tuple:
        ids      = self.tokenizer.encode(prompt, add_special_tokens=False)
        n_prompt = len(ids)

        self.reset()

        prefill_ids = ids[:-1]

        # Prefill: build KV cache for all prompt tokens except the last.
        # launch_ldg_prefill processes all tokens in one C call with one sync.
        if prefill_ids:
            if self._prefill_op is not None:
                ids_tensor = torch.tensor(prefill_ids, dtype=torch.int32)
                self._prefill_op(*self._prefill_args(ids_tensor))
            else:
                # Fallback: use generate_nosync (correct with fixed kernel)
                self._gen(*self._gen_args(
                    prefill_ids[0], len(prefill_ids), self._position
                ))
            self._position += len(prefill_ids)

        # Decode: queue all max_tokens steps on-device with NO per-step CPU sync.
        # The fixed kernel's ldg_decode_body checks d_eos_flag at entry and
        # returns immediately after EOS — subsequent steps are near-zero cost.
        # ldg_update_step also guards against writing garbage tokens after EOS.
        # One cudaStreamSynchronize fires at the end of launch_ldg_generate_nosync.
        cur_token  = ids[-1]
        output_log = self._output_log[:max_tokens]
        output_log.fill_(-1)   # sentinel: positions not written remain -1

        self._gen(*self._gen_args(cur_token, max_tokens, self._position, output_log))

        # Single DtoH copy — stream is already idle after generate_nosync's
        # internal cudaStreamSynchronize.
        tokens_cpu = output_log.cpu().tolist()

        # Trim at EOS or sentinel
        out = []
        for tok in tokens_cpu:
            if tok == -1 or tok == self.tokenizer.eos_token_id:
                break
            out.append(tok)

        # FIX B: advance position by the number of tokens actually generated,not max_tokens. It matters for stateful multi-turn use where reset() is not called between turns.
        self._position += len(out)

        # Track KV high-water mark for partial reset on next request.
        self._kv_high_water = n_prompt + len(out)

        text = self.tokenizer.decode(out, skip_special_tokens=True)
        return text, n_prompt, len(out)


def generate(prompt: str, max_tokens: int = 50, verbose: bool = True) -> str:
    """One-shot convenience: load model, generate, return text."""
    text, _, _ = Decoder(verbose=verbose).generate(prompt, max_tokens)
    return text