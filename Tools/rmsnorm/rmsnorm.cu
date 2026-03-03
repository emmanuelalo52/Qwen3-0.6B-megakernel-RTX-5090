#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include "rmsnorm.cuh"

// Fused Forward Pass WITH residual
std::vector<torch::Tensor> rmsnorm_forward_fused(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor residual,
    float eps) 
{
    const int N = input.numel() / input.size(-1); 
    const int C = input.size(-1);
    
    auto output = torch::empty_like(input);
    auto rms = torch::empty({N}, torch::dtype(torch::kFloat32).device(input.device()));
    
    constexpr int BLOCK_SIZE = 512;
    size_t shared_mem = sizeof(float) * ((BLOCK_SIZE + 31) / 32);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "rmsnorm_forward_fused",
        ([&] {
            kernels::rmsnorm_kernel_fused<scalar_t, BLOCK_SIZE, true><<<N, BLOCK_SIZE, shared_mem>>>(
                output.data_ptr<scalar_t>(),
                rms.data_ptr<float>(), 
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                residual.data_ptr<scalar_t>(),
                N, C, eps
            );
        })
    );

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in rmsnorm_forward_fused: ") + cudaGetErrorString(err));
    }

    return {output, rms};
}

// Forward Pass WITHOUT residual (for final norm, etc.)
std::vector<torch::Tensor> rmsnorm_forward_simple(
    torch::Tensor input, 
    torch::Tensor weight, 
    float eps) 
{
    const int N = input.numel() / input.size(-1); 
    const int C = input.size(-1);
    
    auto output = torch::empty_like(input);
    auto rms = torch::empty({N}, torch::dtype(torch::kFloat32).device(input.device()));
    
    constexpr int BLOCK_SIZE = 512;
    size_t shared_mem = sizeof(float) * ((BLOCK_SIZE + 31) / 32);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "rmsnorm_forward_simple",
        ([&] {
            kernels::rmsnorm_kernel_simple<scalar_t, BLOCK_SIZE, true><<<N, BLOCK_SIZE, shared_mem>>>(
                output.data_ptr<scalar_t>(),
                rms.data_ptr<float>(), 
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                N, C, eps
            );
        })
    );

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in rmsnorm_forward_simple: ") + cudaGetErrorString(err));
    }

    return {output, rms};
}

// Backward Pass
std::vector<torch::Tensor> rmsnorm_backward(
    torch::Tensor grad_out, 
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor rms) 
{
    const int N = input.numel() / input.size(-1);
    const int C = input.size(-1);
    
    auto d_input = torch::empty_like(input);
    auto d_weight = torch::zeros({C}, torch::dtype(torch::kFloat32).device(input.device()));

    constexpr int BLOCK_SIZE = 512;
    size_t shared_mem = sizeof(float) * ((BLOCK_SIZE + 31) / 32);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "rmsnorm_backward",
        ([&] {
            kernels::rmsnorm_backward_kernel<scalar_t, BLOCK_SIZE><<<N, BLOCK_SIZE, shared_mem>>>(
                d_input.data_ptr<scalar_t>(),
                d_weight.data_ptr<float>(),
                grad_out.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                rms.data_ptr<float>(),
                N, C
            );
        })
    );

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in rmsnorm_backward: ") + cudaGetErrorString(err));
    }

    return {d_input, d_weight.to(input.scalar_type())};
}

// Python wrapper that dispatches to the right version
std::vector<torch::Tensor> rmsnorm_forward(
    torch::Tensor input, 
    torch::Tensor weight, 
    float eps,
    torch::Tensor residual = torch::Tensor()) 
{
    if (residual.defined() && residual.numel() > 0) {
        return rmsnorm_forward_fused(input, weight, residual, eps);
    } else {
        return rmsnorm_forward_simple(input, weight, eps);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rmsnorm_forward, "RMSNorm Forward (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("eps"),
          py::arg("residual") = torch::Tensor());
    m.def("backward", &rmsnorm_backward, "RMSNorm Backward (CUDA)");
}
