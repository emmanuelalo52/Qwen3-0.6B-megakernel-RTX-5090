#include <torch/extension.h>
#include "swiglu.cuh"

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Forward: Returns {output, gate_cache, up_cache}
    m.def("forward", &swiglu_forward_cuda, "Fused SwiGLU forward (CUDA)",
          py::arg("x"),
          py::arg("w_gate"),
          py::arg("w_up"),
          py::arg("b_gate") = torch::Tensor(), 
          py::arg("b_up") = torch::Tensor());
    
    // Backward: Computes gradients for x, w_gate, w_up, and biases
    m.def("backward", &swiglu_backward_cuda, "Fused SwiGLU backward (CUDA)",
          py::arg("grad_output"),
          py::arg("x"),
          py::arg("w_gate"),
          py::arg("w_up"),
          py::arg("gate_cache"),
          py::arg("up_cache"));
    
    // Fused Forward + Down: More efficient for inference/inference-only steps
    m.def("forward_down", &swiglu_down_forward_cuda, "Fused SwiGLU + Down projection forward (CUDA)",
          py::arg("x"),
          py::arg("w_gate"),
          py::arg("w_up"),
          py::arg("w_down"),
          py::arg("b_gate") = torch::Tensor(),
          py::arg("b_up") = torch::Tensor(),
          py::arg("b_down") = torch::Tensor());
}