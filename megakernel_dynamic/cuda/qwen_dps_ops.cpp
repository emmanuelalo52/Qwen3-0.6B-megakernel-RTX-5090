// qwen_dps_ops.cpp — PyTorch bindings for the dynamic-persistent Qwen megakernel.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include "qwen_dps.h"

namespace {

constexpr int kAbiVersion = 4;

void check_cuda(const torch::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_aligned(const torch::Tensor &t, const char *name) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, " must be 16-byte aligned");
}

int64_t workspace_bytes() { return static_cast<int64_t>(qwen_dps_workspace_bytes()); }

py::dict info(int64_t weight_format) {
    QwenDpsInfo i{};
    const cudaError_t err = qwen_dps_info(static_cast<int>(weight_format), &i);
    TORCH_CHECK(err == cudaSuccess, "qwen_dps_info failed: ", cudaGetErrorString(err));
    py::dict d;
    d["num_sms"] = i.num_sms;
    d["clc_supported"] = static_cast<bool>(i.clc_supported);
    d["real_mbarrier"] = static_cast<bool>(i.real_mbarrier);
    d["block_threads"] = i.block_threads;
    d["smem_bytes"] = i.smem_bytes;
    d["stage_bytes"] = i.stage_bytes;
    d["weight_stages"] = i.weight_stages;
    d["sched_stages"] = i.sched_stages;
    d["ctas_per_sm"] = i.ctas_per_sm;
    d["tiles_per_step"] = i.tiles_per_step;
    d["lm_tiles"] = i.lm_tiles;
    d["max_seq_supported"] = i.max_seq_supported;
    d["trace_build"] = static_cast<bool>(i.trace_build);
    py::list phase_tiles;
    for (int t : i.phase_tiles) phase_tiles.append(t);
    d["phase_tiles"] = phase_tiles;
    return d;
}

// Runs prompt prefill + max_new greedy decode steps in a single kernel launch.
// tokens[:n_prompt] holds the prompt; generated ids land in output_log (and are
// fed back through tokens[n_prompt:]). layer_table is [28 x 25] int64 (see
// QwenDpsLayerWeights). lm_scale may be empty for fp16 weights. trace is empty, or a
// uint8 [n, 48] buffer of QwenDpsTraceRecord for tickets trace_first .. trace_first + n
// (DPS_TRACE builds only).
// Returns (scheduler mode used, grid CTAs).
std::tuple<int64_t, int64_t> generate(
    torch::Tensor tokens, int64_t n_prompt, int64_t max_new, int64_t start_pos,
    int64_t eos_token, int64_t sched_mode, int64_t weight_format,
    torch::Tensor embed, torch::Tensor layer_table, torch::Tensor final_norm,
    torch::Tensor lm_head, torch::Tensor lm_scale, double lm_gscale,
    torch::Tensor cos_table, torch::Tensor sin_table,
    torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor workspace,
    torch::Tensor output_log, double attn_scale, torch::Tensor trace, int64_t trace_first) {
    check_cuda(tokens, "tokens");
    check_cuda(output_log, "output_log");
    check_cuda(workspace, "workspace");
    check_cuda(k_cache, "k_cache");
    check_cuda(v_cache, "v_cache");
    check_cuda(layer_table, "layer_table");
    check_cuda(lm_head, "lm_head");
    TORCH_CHECK(tokens.dtype() == torch::kInt32 && output_log.dtype() == torch::kInt32,
                "tokens and output_log must be int32");
    TORCH_CHECK(tokens.numel() >= n_prompt + max_new, "tokens must hold n_prompt + max_new ids");
    TORCH_CHECK(output_log.numel() >= max_new, "output_log must hold max_new ids");
    TORCH_CHECK(workspace.numel() * workspace.element_size() >= workspace_bytes(), "workspace too small");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must be [layers, kv_heads, max_seq, head_dim]");
    TORCH_CHECK(layer_table.dtype() == torch::kInt64 &&
                layer_table.numel() * 8 == 28 * static_cast<int64_t>(sizeof(QwenDpsLayerWeights)),
                "layer_table must be 28 x 25 int64");
    TORCH_CHECK(weight_format == QWEN_DPS_FP16 || lm_scale.numel() > 0, "quantized lm_head needs scales");
    if (trace.numel() > 0) {
        check_cuda(trace, "trace");
        TORCH_CHECK(trace.dtype() == torch::kUInt8 && trace.numel() % sizeof(QwenDpsTraceRecord) == 0,
                    "trace must be a uint8 buffer of 48-byte records");
    }
    for (const auto &[t, name] : {std::pair{embed, "embed"}, {lm_head, "lm_head"}, {workspace, "workspace"},
                                  {layer_table, "layer_table"}, {k_cache, "k_cache"}, {v_cache, "v_cache"}})
        check_aligned(t, name);

    QwenDpsLaunch a{};
    a.embed          = embed.data_ptr();
    a.layers         = reinterpret_cast<const QwenDpsLayerWeights *>(layer_table.data_ptr());
    a.final_norm     = final_norm.data_ptr();
    a.lm_head.data   = lm_head.data_ptr();
    a.lm_head.scale  = lm_scale.numel() > 0 ? lm_scale.data_ptr() : nullptr;
    a.lm_head.gscale = static_cast<float>(lm_gscale);
    a.cos_table      = cos_table.data_ptr();
    a.sin_table      = sin_table.data_ptr();
    a.k_cache        = k_cache.data_ptr();
    a.v_cache        = v_cache.data_ptr();
    a.workspace      = workspace.data_ptr();
    a.tokens         = tokens.data_ptr<int>();
    a.output_log     = output_log.data_ptr<int>();
    a.n_prompt       = static_cast<int>(n_prompt);
    a.max_new        = static_cast<int>(max_new);
    a.start_pos      = static_cast<int>(start_pos);
    a.max_seq        = static_cast<int>(k_cache.size(2));
    a.eos_token      = static_cast<int>(eos_token);
    a.sched_mode     = static_cast<int>(sched_mode);
    a.weight_format  = static_cast<int>(weight_format);
    a.attn_scale     = static_cast<float>(attn_scale);
    if (trace.numel() > 0) {
        a.trace       = reinterpret_cast<QwenDpsTraceRecord *>(trace.data_ptr());
        a.trace_first = static_cast<int>(trace_first);
        a.trace_count = static_cast<int>(trace.numel() / sizeof(QwenDpsTraceRecord));
    }

    int used_mode = 0;
    long long grid = 0;
    const cudaError_t err = qwen_dps_launch(&a, c10::cuda::getCurrentCUDAStream(), &used_mode, &grid);
    TORCH_CHECK(err == cudaSuccess, "qwen_dps_launch failed: ", cudaGetErrorString(err));
    return {used_mode, grid};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("abi_version", [] { return kAbiVersion; });
    m.def("workspace_bytes", &workspace_bytes, "Bytes of scratch the kernel needs");
    m.def("info", &info, "Device / build information for one weight format", py::arg("weight_format") = 0);
    m.def("generate", &generate, "Prefill + greedy decode in one dynamic-persistent launch");
}
