#ifndef SWIGLU_FUSED_CUH
#define SWIGLU_FUSED_CUH

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declaration - returns a vector containing {output, gate_cache, up_cache}
std::vector<torch::Tensor> swiglu_forward_cuda(
    torch::Tensor x,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor b_gate,
    torch::Tensor b_up
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> swiglu_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor gate_cache,
    torch::Tensor up_cache
);

// Fused SwiGLU + Down projection
torch::Tensor swiglu_down_forward_cuda(
    torch::Tensor x,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor w_down,
    torch::Tensor b_gate,
    torch::Tensor b_up,
    torch::Tensor b_down
);

#endif