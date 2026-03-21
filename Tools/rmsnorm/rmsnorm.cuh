#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace kernels {
    constexpr int VEC_SIZE = 4;

    /**
     * Fused Add + RMSNorm Kernel
     * Performs: residual = input + residual, then output = RMSNorm(residual)
     */
    template<typename T, int BLOCK_SIZE, bool OUTPUT_RMS = false>
    __global__ void rmsnorm_kernel_fused(
        T* __restrict__ out, 
        float* __restrict__ rms_out, 
        const T* __restrict__ input, 
        const T* __restrict__ weight,
        T* __restrict__ residual,
        int N, int C, float eps) 
    {
        cg::thread_block block = cg::this_thread_block();
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

        int row = blockIdx.x;
        if (row >= N) return;

        // Pointer arithmetic for the current row
        const T* x_ptr = input + row * C;
        T* res_ptr = residual + row * C;
        T* o_ptr = out + row * C;

        __shared__ float shared_data[BLOCK_SIZE / 32];
        __shared__ float rms_inv_shared;

        float thread_sum_sq = 0.0f;
        int C_vec = C / VEC_SIZE;

        const uint2* x_vec = reinterpret_cast<const uint2*>(x_ptr);
        uint2* res_vec = reinterpret_cast<uint2*>(res_ptr);

        // STEP 1: Fused Addition + Sum of Squares
        for (int i = threadIdx.x; i < C_vec; i += BLOCK_SIZE) {
            uint2 v_in = x_vec[i];
            uint2 v_res = res_vec[i];
            
            half2* h2_in = reinterpret_cast<half2*>(&v_in);
            half2* h2_res = reinterpret_cast<half2*>(&v_res);

            // Vectorized addition: residual = input + residual
            h2_res[0] = __hadd2(h2_in[0], h2_res[0]);
            h2_res[1] = __hadd2(h2_in[1], h2_res[1]);

            // Write back updated residual
            res_vec[i] = v_res; 

            // Calculate sum of squares using the NEW residual values
            float2 f0 = __half22float2(h2_res[0]);
            float2 f1 = __half22float2(h2_res[1]);
            thread_sum_sq += f0.x * f0.x + f0.y * f0.y + f1.x * f1.x + f1.y * f1.y;
        }

        // Handle remainder for non-multiple of 4 widths
        for (int i = C_vec * VEC_SIZE + threadIdx.x; i < C; i += BLOCK_SIZE) {
            float val = static_cast<float>(x_ptr[i]) + static_cast<float>(res_ptr[i]);
            res_ptr[i] = static_cast<T>(val);
            thread_sum_sq += val * val;
        }

        // STEP 2: Block Reduction to compute RMS
        float warp_sum = cg::reduce(warp, thread_sum_sq, cg::plus<float>());
        if (warp.thread_rank() == 0) shared_data[threadIdx.x / 32] = warp_sum;
        block.sync();

        float block_sum = 0.0f;
        if (threadIdx.x < (BLOCK_SIZE / 32)) block_sum = shared_data[threadIdx.x];
        block_sum = cg::reduce(warp, block_sum, cg::plus<float>());

        if (threadIdx.x == 0) {
            float inv_rms = rsqrtf(block_sum / static_cast<float>(C) + eps);
            rms_inv_shared = inv_rms;
            if constexpr (OUTPUT_RMS) rms_out[row] = 1.0f / inv_rms;
        }
        block.sync();

        // STEP 3: Normalization
        float rms_inv = rms_inv_shared;
        uint2* o_vec = reinterpret_cast<uint2*>(o_ptr);
        const uint2* w_vec = reinterpret_cast<const uint2*>(weight);

        for (int i = threadIdx.x; i < C_vec; i += BLOCK_SIZE) {
            uint2 v_sum = res_vec[i]; // Read the already updated residual
            uint2 v_w = w_vec[i];
            
            half2* h2_sum = reinterpret_cast<half2*>(&v_sum);
            half2* h2_w = reinterpret_cast<half2*>(&v_w);

            #pragma unroll
            for (int j = 0; j < 2; j++) {
                float2 f_s = __half22float2(h2_sum[j]);
                float2 f_w = __half22float2(h2_w[j]);
                f_s.x *= (rms_inv * f_w.x);
                f_s.y *= (rms_inv * f_w.y);
                h2_sum[j] = __float22half2_rn(f_s);
            }
            o_vec[i] = v_sum;
        }
        
        // Handle remainder
        for (int i = C_vec * VEC_SIZE + threadIdx.x; i < C; i += BLOCK_SIZE) {
            float val = static_cast<float>(res_ptr[i]);
            float w = static_cast<float>(weight[i]);
            o_ptr[i] = static_cast<T>(val * rms_inv * w);
        }
    }

    /**
     * Simple RMSNorm Kernel (no residual addition)
     * Performs: output = RMSNorm(input)
     */
    template<typename T, int BLOCK_SIZE, bool OUTPUT_RMS = false>
    __global__ void rmsnorm_kernel_simple(
        T* __restrict__ out, 
        float* __restrict__ rms_out, 
        const T* __restrict__ input, 
        const T* __restrict__ weight,
        int N, int C, float eps) 
    {
        cg::thread_block block = cg::this_thread_block();
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

        int row = blockIdx.x;
        if (row >= N) return;

        const T* x_ptr = input + row * C;
        T* o_ptr = out + row * C;

        __shared__ float shared_data[BLOCK_SIZE / 32];
        __shared__ float rms_inv_shared;

        float thread_sum_sq = 0.0f;

        // STEP 1: Sum of Squares
        for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
            float val = static_cast<float>(x_ptr[i]);
            thread_sum_sq += val * val;
        }

        // STEP 2: Block Reduction
        float warp_sum = cg::reduce(warp, thread_sum_sq, cg::plus<float>());
        if (warp.thread_rank() == 0) shared_data[threadIdx.x / 32] = warp_sum;
        block.sync();

        float block_sum = 0.0f;
        if (threadIdx.x < (BLOCK_SIZE / 32)) block_sum = shared_data[threadIdx.x];
        block_sum = cg::reduce(warp, block_sum, cg::plus<float>());

        if (threadIdx.x == 0) {
            float inv_rms = rsqrtf(block_sum / static_cast<float>(C) + eps);
            rms_inv_shared = inv_rms;
            if constexpr (OUTPUT_RMS) rms_out[row] = 1.0f / inv_rms;
        }
        block.sync();

        // STEP 3: Normalization
        float rms_inv = rms_inv_shared;
        for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
            float val = static_cast<float>(x_ptr[i]);
            float w = static_cast<float>(weight[i]);
            o_ptr[i] = static_cast<T>(val * rms_inv * w);
        }
    }

    /**
     * RMSNorm Backward Kernel
     */
    template<typename T, int BLOCK_SIZE>
    __global__ void rmsnorm_backward_kernel(
        T* __restrict__ d_inp, 
        float* __restrict__ d_weight, 
        const T* __restrict__ grad, 
        const T* __restrict__ inp, 
        const T* __restrict__ weight, 
        const float* __restrict__ rms, 
        int N, int C) 
    {
        int idx = blockIdx.x;
        if (idx >= N) return;

        cg::thread_block block = cg::this_thread_block();
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

        const T* x = inp + idx * C;
        const T* g = grad + idx * C;
        T* dx = d_inp + idx * C;
        
        float r_inv = 1.0f / (rms[idx] + 1e-6f);
        __shared__ float shared_data[BLOCK_SIZE / 32];
        __shared__ float correction_shared;

        float thread_dot = 0.0f;
        
        // Compute dot product and weight gradient
        for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
            float g_val = static_cast<float>(g[i]);
            float x_val = static_cast<float>(x[i]);
            float w_val = static_cast<float>(weight[i]);
            
            thread_dot += g_val * w_val * x_val;
            atomicAdd(&d_weight[i], g_val * x_val * r_inv);
        }

        // Reduce dot product across block
        float warp_dot = cg::reduce(warp, thread_dot, cg::plus<float>());
        if (warp.thread_rank() == 0) shared_data[threadIdx.x / 32] = warp_dot;
        block.sync();
        
        float block_dot = 0.0f;
        if (threadIdx.x < (BLOCK_SIZE / 32)) block_dot = shared_data[threadIdx.x];
        block_dot = cg::reduce(warp, block_dot, cg::plus<float>());
        
        if (threadIdx.x == 0) {
            correction_shared = block_dot / (static_cast<float>(C) * rms[idx] * rms[idx]);
        }
        block.sync();

        // Compute input gradient
        float correction = correction_shared;
        for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
            float g_val = static_cast<float>(g[i]);
            float x_val = static_cast<float>(x[i]);
            float w_val = static_cast<float>(weight[i]);
            
            dx[i] = static_cast<T>(r_inv * (g_val * w_val - x_val * correction));
        }
    }
}
