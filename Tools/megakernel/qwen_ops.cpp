// qwen_ops.cpp - PyTorch C++ Extension for Qwen Megakernel
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <c10/cuda/CUDAStream.h>
struct LDGLayerWeight;

// Forward declarations of C functions from megakernel.cu
extern "C" void launch_ldg_decode_direct(
    int input_token_id, int *output_token_id, const void *embed_weight,
    const LDGLayerWeight *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs, int num_layers, int position, int max_seq_len,
    float attn_scale, cudaStream_t stream);

extern "C" void launch_ldg_generate_nosync(
    int first_token_id, int num_steps, const void *embed_weight,
    const LDGLayerWeight *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs,
    int *output_log,
    int num_layers, int start_position,
    int max_seq_len, float attn_scale, int eos_token_id, cudaStream_t stream);

extern "C" void launch_ldg_prefill(
    const int *token_ids, int num_tokens, const void *embed_weight,
    const LDGLayerWeight *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs, int num_layers, int start_position,
    int max_seq_len, float attn_scale, cudaStream_t stream);

// ABI Version 2: Matches 12-pointer (96-byte) LDGLayerWeight alignment — RTX 5090 (sm_120)
int abi_version() {
    return 2;
}

torch::Tensor decode(
    int input_token_id,
    torch::Tensor embed_weight,
    torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight,
    torch::Tensor lm_head_weight,
    torch::Tensor cos_table,
    torch::Tensor sin_table,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor hidden,
    torch::Tensor act,
    torch::Tensor res,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor attn_out,
    torch::Tensor mlp_inter,
    torch::Tensor norm_out,
    torch::Tensor bmax_vals,
    torch::Tensor bmax_idxs,
    int num_layers,
    int position,
    int max_seq_len,
    float attn_scale
) {
    auto output_token_id = torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(layer_weights_packed.data_ptr());
    TORCH_CHECK(ptr_val % 16 == 0,
        "RTX 5090 Alignment Error: layer_weights_packed must be 16-byte aligned. "
        "Current address ends in: ", ptr_val % 16);

    launch_ldg_decode_direct(
        input_token_id,
        output_token_id.data_ptr<int>(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LDGLayerWeight*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(),
        lm_head_weight.data_ptr(),
        cos_table.data_ptr(),
        sin_table.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        hidden.data_ptr(),
        act.data_ptr(),
        res.data_ptr(),
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        attn_out.data_ptr(),
        mlp_inter.data_ptr(),
        norm_out.data_ptr(),
        bmax_vals.data_ptr(),
        bmax_idxs.data_ptr(),
        num_layers,
        position,
        max_seq_len,
        attn_scale,
        stream
    );

    return output_token_id;
}

torch::Tensor generate_nosync(
    int first_token_id,
    int num_steps,
    torch::Tensor embed_weight,
    torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight,
    torch::Tensor lm_head_weight,
    torch::Tensor cos_table,
    torch::Tensor sin_table,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor hidden,
    torch::Tensor act,
    torch::Tensor res,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor attn_out,
    torch::Tensor mlp_inter,
    torch::Tensor norm_out,
    torch::Tensor bmax_vals,
    torch::Tensor bmax_idxs,
    torch::Tensor output_log,
    int num_layers,
    int start_position,
    int max_seq_len,
    float attn_scale,
    int eos_token_id = 151645
) {
    // Validate output_log so misuse produces a clear error, not a CUDA crash.
    TORCH_CHECK(output_log.dtype()    == torch::kInt32,
        "generate_nosync: output_log must be dtype int32, got ", output_log.dtype());
    TORCH_CHECK(output_log.device().is_cuda(),
        "generate_nosync: output_log must be on CUDA");
    TORCH_CHECK(output_log.is_contiguous(),
        "generate_nosync: output_log must be contiguous");
    TORCH_CHECK(output_log.numel() >= num_steps,
        "generate_nosync: output_log.numel() (", output_log.numel(),
        ") < num_steps (", num_steps, ")");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // RTX 5090 ALIGNMENT GUARD
    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(layer_weights_packed.data_ptr());
    TORCH_CHECK(ptr_val % 16 == 0,
        "RTX 5090 Alignment Error: layer_weights_packed must be 16-byte aligned.");

    launch_ldg_generate_nosync(
        first_token_id,
        num_steps,
        embed_weight.data_ptr(),
        reinterpret_cast<const LDGLayerWeight*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(),
        lm_head_weight.data_ptr(),
        cos_table.data_ptr(),
        sin_table.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        hidden.data_ptr(),
        act.data_ptr(),
        res.data_ptr(),
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        attn_out.data_ptr(),
        mlp_inter.data_ptr(),
        norm_out.data_ptr(),
        bmax_vals.data_ptr(),
        bmax_idxs.data_ptr(),
        output_log.data_ptr<int>(), 
        num_layers,
        start_position,
        max_seq_len,
        attn_scale,
        eos_token_id,
        stream
    );

    return output_log;  // return the same tensor the caller passed in
}

void prefill(
    torch::Tensor token_ids,          // CPU int32 [num_tokens]
    torch::Tensor embed_weight,
    torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight,
    torch::Tensor lm_head_weight,
    torch::Tensor cos_table,
    torch::Tensor sin_table,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor hidden,
    torch::Tensor act,
    torch::Tensor res,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor attn_out,
    torch::Tensor mlp_inter,
    torch::Tensor norm_out,
    torch::Tensor bmax_vals,
    torch::Tensor bmax_idxs,
    int num_layers,
    int start_position,
    int max_seq_len,
    float attn_scale
) {
    TORCH_CHECK(token_ids.dtype() == torch::kInt32, "token_ids must be int32");
    TORCH_CHECK(token_ids.device().is_cpu(),         "token_ids must be on CPU");
    TORCH_CHECK(token_ids.is_contiguous(),           "token_ids must be contiguous");

    int num_tokens = (int)token_ids.size(0);
    if (num_tokens == 0) return;

    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(layer_weights_packed.data_ptr());
    TORCH_CHECK(ptr_val % 16 == 0,
        "RTX 5090 Alignment Error: layer_weights_packed must be 16-byte aligned.");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    launch_ldg_prefill(
        token_ids.data_ptr<int>(),
        num_tokens,
        embed_weight.data_ptr(),
        reinterpret_cast<const LDGLayerWeight*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(),
        lm_head_weight.data_ptr(),
        cos_table.data_ptr(),
        sin_table.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        hidden.data_ptr(),
        act.data_ptr(),
        res.data_ptr(),
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        attn_out.data_ptr(),
        mlp_inter.data_ptr(),
        norm_out.data_ptr(),
        bmax_vals.data_ptr(),
        bmax_idxs.data_ptr(),
        num_layers,
        start_position,
        max_seq_len,
        attn_scale,
        stream
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode",          &decode,          "Qwen3 single token decode");
    m.def("generate_nosync", &generate_nosync, "Qwen3 batched generation (caller-supplied output_log)");
    m.def("abi_version",     &abi_version,     "Get the ABI version of the extension");
    m.def("prefill",         &prefill,         "Qwen3 batched prefill — fills KV cache for all prompt tokens");
}