#include "swiglu.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// ============================================================================
// ATOMIC ADD HELPERS FOR PYTORCH TYPES
// ============================================================================

// Generic atomic add wrapper (works for float, double, int, etc.)
template<typename T>
__device__ __forceinline__ void atomic_add_helper(T* address, T val) {
    atomicAdd(address, val);
}

// Specialization for c10::Half -> __half conversion
template<>
__device__ __forceinline__ void atomic_add_helper<c10::Half>(c10::Half* address, c10::Half val) {
    atomicAdd(reinterpret_cast<__half*>(address), *reinterpret_cast<__half*>(&val));
}

// Specialization for c10::BFloat16 -> __nv_bfloat16 conversion (Ampere+ only)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template<>
__device__ __forceinline__ void atomic_add_helper<c10::BFloat16>(c10::BFloat16* address, c10::BFloat16 val) {
    atomicAdd(reinterpret_cast<__nv_bfloat16*>(address), *reinterpret_cast<__nv_bfloat16*>(&val));
}
#else
// For older GPUs (sm_75 and below), BFloat16 atomicAdd not supported
// Fall back to atomicCAS loop
template<>
__device__ __forceinline__ void atomic_add_helper<c10::BFloat16>(c10::BFloat16* address, c10::BFloat16 val) {
    // Convert to float for atomic operation (loses some precision but works)
    auto* address_as_uint = reinterpret_cast<unsigned int*>(address);
    unsigned int old = *address_as_uint;
    unsigned int assumed;
    
    do {
        assumed = old;
        c10::BFloat16 old_val = *reinterpret_cast<c10::BFloat16*>(&assumed);
        float new_val_f = static_cast<float>(old_val) + static_cast<float>(val);
        c10::BFloat16 new_val = static_cast<c10::BFloat16>(new_val_f);
        old = atomicCAS(address_as_uint, assumed, *reinterpret_cast<unsigned int*>(&new_val));
    } while (assumed != old);
}
#endif

// ============================================================================
// SILU ACTIVATION AND DERIVATIVES
// ============================================================================

// SiLU activation: x * sigmoid(x)
template<typename T>
__device__ __forceinline__ T silu(T x) {
    return x / (static_cast<T>(1.0) + expf(-x));
}

// Specialized SiLU for half precision
__device__ __forceinline__ __half silu(__half x) {
    return __hmul(x, hrcp(__hadd(__float2half(1.0f), hexp(__hneg(x)))));
}

// Specialized SiLU for bfloat16 (Ampere+)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ __nv_bfloat16 silu(__nv_bfloat16 x) {
    float x_f = __bfloat162float(x);
    float result = x_f / (1.0f + expf(-x_f));
    return __float2bfloat16(result);
}
#endif

// SiLU derivative: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
template<typename T>
__device__ __forceinline__ T silu_grad(T x, T grad_output) {
    T sigmoid_x = static_cast<T>(1.0) / (static_cast<T>(1.0) + expf(-x));
    return grad_output * sigmoid_x * (static_cast<T>(1.0) + x * (static_cast<T>(1.0) - sigmoid_x));
}

// ============================================================================
// FORWARD KERNELS
// ============================================================================

/**
 * Fused SwiGLU Forward Kernel
 */
template<typename T>
__global__ void swiglu_forward_kernel(
    const T* __restrict__ x,
    const T* __restrict__ w_gate,
    const T* __restrict__ w_up,
    const T* __restrict__ b_gate,
    const T* __restrict__ b_up,
    T* __restrict__ output,
    T* __restrict__ gate_cache,
    T* __restrict__ up_cache,
    int batch_size,
    int seq_len,
    int hidden_size,
    int intermediate_size
) {
    int total_elements = batch_size * seq_len * intermediate_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        int inter_idx = idx % intermediate_size;
        int seq_idx = (idx / intermediate_size) % seq_len;
        int batch_idx = idx / (seq_len * intermediate_size);
        
        int x_offset = (batch_idx * seq_len + seq_idx) * hidden_size;
        
        T gate_val = b_gate ? b_gate[inter_idx] : static_cast<T>(0.0);
        T up_val = b_up ? b_up[inter_idx] : static_cast<T>(0.0);
        
        #pragma unroll 4
        for (int h = 0; h < hidden_size; h++) {
            T x_val = x[x_offset + h];
            gate_val += x_val * w_gate[h * intermediate_size + inter_idx];
            up_val += x_val * w_up[h * intermediate_size + inter_idx];
        }
        
        T gate_activated = silu(gate_val);
        T result = gate_activated * up_val;
        
        output[idx] = result;
        if (gate_cache) gate_cache[idx] = gate_val;
        if (up_cache) up_cache[idx] = up_val;
    }
}

/**
 * Optimized SwiGLU Forward Kernel using Shared Memory
 */
template<typename T, int TILE_SIZE = 32>
__global__ void swiglu_forward_optimized_kernel(
    const T* __restrict__ x,
    const T* __restrict__ w_gate,
    const T* __restrict__ w_up,
    const T* __restrict__ b_gate,
    const T* __restrict__ b_up,
    T* __restrict__ output,
    T* __restrict__ gate_cache,
    T* __restrict__ up_cache,
    int batch_size,
    int seq_len,
    int hidden_size,
    int intermediate_size
) {
    __shared__ T s_x[TILE_SIZE];
    __shared__ T s_w_gate[TILE_SIZE];
    __shared__ T s_w_up[TILE_SIZE];
    
    int batch_idx = blockIdx.z;
    int seq_idx = blockIdx.y;
    int inter_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || inter_idx >= intermediate_size) {
        return;
    }
    
    int x_offset = (batch_idx * seq_len + seq_idx) * hidden_size;
    int out_idx = (batch_idx * seq_len + seq_idx) * intermediate_size + inter_idx;
    
    T gate_val = b_gate ? b_gate[inter_idx] : static_cast<T>(0.0);
    T up_val = b_up ? b_up[inter_idx] : static_cast<T>(0.0);
    
    for (int tile = 0; tile < (hidden_size + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        int h_idx = tile * TILE_SIZE + threadIdx.x;
        
        if (h_idx < hidden_size && threadIdx.x < TILE_SIZE) {
            s_x[threadIdx.x] = x[x_offset + h_idx];
        } else {
            s_x[threadIdx.x] = static_cast<T>(0.0);
        }
        
        if (h_idx < hidden_size && inter_idx < intermediate_size) {
            s_w_gate[threadIdx.x] = w_gate[h_idx * intermediate_size + inter_idx];
            s_w_up[threadIdx.x] = w_up[h_idx * intermediate_size + inter_idx];
        } else {
            s_w_gate[threadIdx.x] = static_cast<T>(0.0);
            s_w_up[threadIdx.x] = static_cast<T>(0.0);
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            if (tile * TILE_SIZE + i < hidden_size) {
                gate_val += s_x[i] * s_w_gate[i];
                up_val += s_x[i] * s_w_up[i];
            }
        }
        
        __syncthreads();
    }
    
    T gate_activated = silu(gate_val);
    T result = gate_activated * up_val;
    
    output[out_idx] = result;
    if (gate_cache) gate_cache[out_idx] = gate_val;
    if (up_cache) up_cache[out_idx] = up_val;
}

// ============================================================================
// BACKWARD KERNEL
// ============================================================================

/**
 * Fused SwiGLU Backward Kernel
 */
template<typename T>
__global__ void swiglu_backward_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ x,
    const T* __restrict__ w_gate,
    const T* __restrict__ w_up,
    const T* __restrict__ gate_cache,
    const T* __restrict__ up_cache,
    T* __restrict__ grad_x,
    T* __restrict__ grad_w_gate,
    T* __restrict__ grad_w_up,
    int batch_size,
    int seq_len,
    int hidden_size,
    int intermediate_size
) {
    int total_elements = batch_size * seq_len * intermediate_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        int inter_idx = idx % intermediate_size;
        int seq_idx = (idx / intermediate_size) % seq_len;
        int batch_idx = idx / (seq_len * intermediate_size);
        int x_offset = (batch_idx * seq_len + seq_idx) * hidden_size;
        
        T grad_out = grad_output[idx];
        T gate_val = gate_cache[idx];
        T up_val = up_cache[idx];
        
        T gate_activated = silu(gate_val);
        T grad_gate = silu_grad(gate_val, grad_out * up_val);
        T grad_up = grad_out * gate_activated;
        
        for (int h = 0; h < hidden_size; h++) {
            T x_val = x[x_offset + h];
            T g_x = grad_gate * w_gate[h * intermediate_size + inter_idx] +
                   grad_up * w_up[h * intermediate_size + inter_idx];
            
            // Use our helper function that handles type conversion
            atomic_add_helper(&grad_x[x_offset + h], g_x);
            atomic_add_helper(&grad_w_gate[h * intermediate_size + inter_idx], grad_gate * x_val);
            atomic_add_helper(&grad_w_up[h * intermediate_size + inter_idx], grad_up * x_val);
        }
    }
}

// ============================================================================
// FUSED FORWARD + DOWN PROJECTION KERNEL
// ============================================================================

/**
 * Fused SwiGLU + Down Projection Forward Kernel
 */
template<typename T>
__global__ void swiglu_down_forward_kernel(
    const T* __restrict__ x,
    const T* __restrict__ w_gate,
    const T* __restrict__ w_up,
    const T* __restrict__ w_down,
    const T* __restrict__ b_gate,
    const T* __restrict__ b_up,
    const T* __restrict__ b_down,
    T* __restrict__ output,
    int batch_size,
    int seq_len,
    int hidden_size,
    int intermediate_size
) {
    int total_elements = batch_size * seq_len * hidden_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        int h_out = idx % hidden_size;
        int seq_idx = (idx / hidden_size) % seq_len;
        int batch_idx = idx / (seq_len * hidden_size);
        
        int x_offset = (batch_idx * seq_len + seq_idx) * hidden_size;
        
        T result = b_down ? b_down[h_out] : static_cast<T>(0.0);
        
        for (int inter = 0; inter < intermediate_size; inter++) {
            T gate_val = b_gate ? b_gate[inter] : static_cast<T>(0.0);
            T up_val = b_up ? b_up[inter] : static_cast<T>(0.0);
            
            #pragma unroll 4
            for (int h = 0; h < hidden_size; h++) {
                T x_val = x[x_offset + h];
                gate_val += x_val * w_gate[h * intermediate_size + inter];
                up_val += x_val * w_up[h * intermediate_size + inter];
            }
            
            T swiglu_out = silu(gate_val) * up_val;
            result += swiglu_out * w_down[inter * hidden_size + h_out];
        }
        
        output[idx] = result;
    }
}

// ============================================================================
// HOST FUNCTIONS
// ============================================================================

std::vector<torch::Tensor> swiglu_forward_cuda(
    torch::Tensor x,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor b_gate,
    torch::Tensor b_up
) {
    auto batch_size = x.size(0);
    auto seq_len = x.size(1);
    auto hidden_size = x.size(2);
    auto intermediate_size = w_gate.size(1);
    
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, seq_len, intermediate_size}, options);
    auto gate_cache = torch::empty({batch_size, seq_len, intermediate_size}, options);
    auto up_cache = torch::empty({batch_size, seq_len, intermediate_size}, options);
    
    int total_elements = batch_size * seq_len * intermediate_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "swiglu_forward",
        ([&] {
            swiglu_forward_kernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                w_gate.data_ptr<scalar_t>(),
                w_up.data_ptr<scalar_t>(),
                b_gate.defined() ? b_gate.data_ptr<scalar_t>() : nullptr,
                b_up.defined() ? b_up.data_ptr<scalar_t>() : nullptr,
                output.data_ptr<scalar_t>(),
                gate_cache.data_ptr<scalar_t>(),
                up_cache.data_ptr<scalar_t>(),
                batch_size, seq_len, hidden_size, intermediate_size
            );
        })
    );
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in swiglu_forward: ") + cudaGetErrorString(err));
    }
    
    return {output, gate_cache, up_cache};
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> swiglu_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor gate_cache,
    torch::Tensor up_cache
) {
    auto batch_size = x.size(0);
    auto seq_len = x.size(1);
    auto hidden_size = x.size(2);
    auto intermediate_size = w_gate.size(1);
    
    auto grad_x = torch::zeros_like(x);
    auto grad_w_gate = torch::zeros_like(w_gate);
    auto grad_w_up = torch::zeros_like(w_up);
    
    int total_elements = batch_size * seq_len * intermediate_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "swiglu_backward",
        ([&] {
            swiglu_backward_kernel<scalar_t><<<blocks, threads>>>(
                grad_output.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(),
                w_gate.data_ptr<scalar_t>(),
                w_up.data_ptr<scalar_t>(),
                gate_cache.data_ptr<scalar_t>(),
                up_cache.data_ptr<scalar_t>(),
                grad_x.data_ptr<scalar_t>(),
                grad_w_gate.data_ptr<scalar_t>(),
                grad_w_up.data_ptr<scalar_t>(),
                batch_size, seq_len, hidden_size, intermediate_size
            );
        })
    );
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in swiglu_backward: ") + cudaGetErrorString(err));
    }
    
    return std::make_tuple(grad_x, grad_w_gate, grad_w_up);
}


torch::Tensor swiglu_down_forward_cuda(
    torch::Tensor x,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor w_down,
    torch::Tensor b_gate,
    torch::Tensor b_up,
    torch::Tensor b_down
) {
    auto batch_size = x.size(0);
    auto seq_len = x.size(1);
    auto hidden_size = x.size(2);
    auto intermediate_size = w_gate.size(1);
    
    auto output = torch::empty({batch_size, seq_len, hidden_size}, 
                               torch::TensorOptions()
                                   .dtype(x.dtype())
                                   .device(x.device()));
    
    int total_elements = batch_size * seq_len * hidden_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    const void* b_gate_ptr = b_gate.defined() ? b_gate.data_ptr() : nullptr;
    const void* b_up_ptr = b_up.defined() ? b_up.data_ptr() : nullptr;
    const void* b_down_ptr = b_down.defined() ? b_down.data_ptr() : nullptr;
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "swiglu_down_forward_cuda",
        ([&] {
            swiglu_down_forward_kernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                w_gate.data_ptr<scalar_t>(),
                w_up.data_ptr<scalar_t>(),
                w_down.data_ptr<scalar_t>(),
                static_cast<const scalar_t*>(b_gate_ptr),
                static_cast<const scalar_t*>(b_up_ptr),
                static_cast<const scalar_t*>(b_down_ptr),
                output.data_ptr<scalar_t>(),
                batch_size, seq_len, hidden_size, intermediate_size
            );
        })
    );
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in swiglu_down_forward: ") + cudaGetErrorString(err));
    }
    
    return output;
}
