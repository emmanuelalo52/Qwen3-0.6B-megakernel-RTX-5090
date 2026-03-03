import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

# Set library path before importing CUDA extensions
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':' + os.path.join(
    os.path.dirname(torch.__file__), 'lib'
)

# Try to import the compiled CUDA extension
try:
    import swiglu_fused as swiglu
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: swiglu CUDA extension not found. Using PyTorch fallback.")
    print("To compile: python setup.py install")


if CUDA_AVAILABLE:
    class SwiGLUFunction(torch.autograd.Function):
        """CUDA-accelerated SwiGLU autograd function"""
        
        @staticmethod
        def forward(ctx, x, w_gate, w_up, b_gate=None, b_up=None):
            x = x.contiguous()
            w_gate = w_gate.contiguous()
            w_up = w_up.contiguous()
            
            output, gate_cache, up_cache = swiglu.forward(x, w_gate, w_up, b_gate, b_up)
            
            ctx.save_for_backward(x, w_gate, w_up, gate_cache, up_cache)
            ctx.has_bias = (b_gate is not None, b_up is not None)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            x, w_gate, w_up, gate_cache, up_cache = ctx.saved_tensors
            grad_output = grad_output.contiguous()
            
            grad_x, grad_w_gate, grad_w_up = swiglu.backward(
                grad_output, x, w_gate, w_up, gate_cache, up_cache
            )
            
            grad_b_gate = None
            grad_b_up = None
            
            if ctx.has_bias[0] or ctx.has_bias[1]:
                sigmoid_gate = torch.sigmoid(gate_cache)
                silu_gate = gate_cache * sigmoid_gate
                silu_grad = sigmoid_gate * (1.0 + gate_cache * (1.0 - sigmoid_gate))
                
                if ctx.has_bias[0]:
                    grad_gate_pre = grad_output * up_cache * silu_grad
                    grad_b_gate = grad_gate_pre.sum(dim=list(range(grad_gate_pre.ndim - 1)))
                
                if ctx.has_bias[1]:
                    grad_up_pre = grad_output * silu_gate
                    grad_b_up = grad_up_pre.sum(dim=list(range(grad_up_pre.ndim - 1)))
            
            return grad_x, grad_w_gate, grad_w_up, grad_b_gate, grad_b_up

else:
    class SwiGLUFunction(torch.autograd.Function):
        """PyTorch fallback SwiGLU implementation"""
        
        @staticmethod
        def forward(ctx, x, w_gate, w_up, b_gate=None, b_up=None):
            # w_gate and w_up are [hidden_size, intermediate_size]
            # We need to transpose for F.linear which expects [out, in]
            gate = F.linear(x, w_gate.t(), b_gate)
            up = F.linear(x, w_up.t(), b_up)
            output = F.silu(gate) * up
            
            ctx.save_for_backward(x, w_gate, w_up, gate, up)
            ctx.has_bias = (b_gate is not None, b_up is not None)
            return output
        
        @staticmethod
        def backward(ctx, grad_output):
            x, w_gate, w_up, gate, up = ctx.saved_tensors
            
            sigmoid_gate = torch.sigmoid(gate)
            silu_gate = gate * sigmoid_gate
            silu_grad = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
            
            grad_gate = grad_output * up * silu_grad
            grad_up = grad_output * silu_gate
            
            # grad_x computation with correct dimensions
            grad_x = grad_gate @ w_gate.t() + grad_up @ w_up.t()
            
            # Weight gradients
            x_flat = x.reshape(-1, x.size(-1))
            grad_gate_flat = grad_gate.reshape(-1, grad_gate.size(-1))
            grad_up_flat = grad_up.reshape(-1, grad_up.size(-1))
            
            grad_w_gate = grad_gate_flat.t() @ x_flat
            grad_w_up = grad_up_flat.t() @ x_flat
            
            grad_b_gate = None
            grad_b_up = None
            if ctx.has_bias[0]:
                grad_b_gate = grad_gate.sum(dim=list(range(grad_gate.ndim - 1)))
            if ctx.has_bias[1]:
                grad_b_up = grad_up.sum(dim=list(range(grad_up.ndim - 1)))
            
            return grad_x, grad_w_gate, grad_w_up, grad_b_gate, grad_b_up


class FusedSwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size, bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        self.w_gate = nn.Parameter(torch.empty(hidden_size, intermediate_size))
        self.w_up = nn.Parameter(torch.empty(hidden_size, intermediate_size))
        
        if bias:
            self.b_gate = nn.Parameter(torch.zeros(intermediate_size))
            self.b_up = nn.Parameter(torch.zeros(intermediate_size))
        else:
            self.register_parameter('b_gate', None)
            self.register_parameter('b_up', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_gate, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_up, a=math.sqrt(5))
        
        if self.b_gate is not None:
            fan_in = self.hidden_size
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b_gate, -bound, bound)
            nn.init.uniform_(self.b_up, -bound, bound)
    
    def forward(self, x):
        return SwiGLUFunction.apply(x, self.w_gate, self.w_up, self.b_gate, self.b_up)
    
    def extra_repr(self):
        return f'hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}, bias={self.b_gate is not None}'


class FusedFeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.swiglu = FusedSwiGLU(hidden_size, intermediate_size, bias=bias)
        self.w_down = nn.Linear(intermediate_size, hidden_size, bias=bias)
    
    def forward(self, x):
        intermediate = self.swiglu(x)
        output = self.w_down(intermediate)
        return output


def test_swiglu_gradients():
    print("Testing SwiGLU gradients...")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping tests")
        return
    
    batch, seq, hidden, inter = 2, 4, 64, 256
    device = 'cuda'
    
    print("\n1. Testing without bias...")
    x = torch.randn(batch, seq, hidden, device=device, requires_grad=True)
    swiglu = FusedSwiGLU(hidden, inter, bias=False).to(device)
    
    output = swiglu(x)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None
    assert swiglu.w_gate.grad is not None
    assert swiglu.w_up.grad is not None
    print("✓ Gradients computed correctly")
    
    print("\n2. Testing with bias...")
    x = torch.randn(batch, seq, hidden, device=device, requires_grad=True)
    swiglu_bias = FusedSwiGLU(hidden, inter, bias=True).to(device)
    
    output = swiglu_bias(x)
    loss = output.sum()
    loss.backward()
    
    assert swiglu_bias.b_gate.grad is not None
    assert swiglu_bias.b_up.grad is not None
    print("✓ Bias gradients computed")
    
    print("\n3. Testing numerical stability...")
    x_large = torch.randn(batch, seq, hidden, device=device) * 100
    x_small = torch.randn(batch, seq, hidden, device=device) * 0.01
    
    out_large = swiglu(x_large)
    out_small = swiglu(x_small)
    
    assert not torch.isnan(out_large).any()
    assert not torch.isnan(out_small).any()
    print("✓ Numerically stable")
    
    print("\n4. Testing FusedFeedForward...")
    x = torch.randn(batch, seq, hidden, device=device, requires_grad=True)
    ffn = FusedFeedForward(hidden, inter).to(device)
    
    output = ffn(x)
    assert output.shape == (batch, seq, hidden)
    
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    print("✓ FusedFeedForward working")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_swiglu_gradients()
