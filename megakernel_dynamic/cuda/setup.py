"""
Builds the qwen_dps_C extension (dynamic-persistent Qwen3-0.6B megakernel).

    cd megakernel_dynamic/cuda
    python setup.py build_ext --inplace                  # B200 (sm_100a), the default
    DPS_ARCH=120a python setup.py build_ext --inplace    # RTX 5090
    DPS_ARCH=75   python setup.py build_ext --inplace    # older GPUs: atomic scheduler, emulated TMA

DPS_ARCH picks one target. The weight ring gets DPS_RING_BYTES of shared memory
(default: what that GPU allows per block, minus ~30 KB for everything else); each
weight format (fp16 / fp8 / fp4) cuts it into as many stages as fit.
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ARCH = os.environ.get("DPS_ARCH", "100a")
DEFAULT_RING_BYTES = {"75": 36864, "80": 131072, "86": 65536, "87": 131072, "89": 65536,
                      "90": 196608, "90a": 196608, "100": 196608, "100a": 196608, "100f": 196608,
                      "103a": 196608, "120": 65536, "120a": 65536, "121a": 65536}
RING_BYTES = os.environ.get("DPS_RING_BYTES", str(DEFAULT_RING_BYTES.get(ARCH, 32768)))
SSTAGES = os.environ.get("DPS_SSTAGES", "6")

HERE = os.path.dirname(os.path.abspath(__file__))

setup(
    name="qwen_dps_C",
    ext_modules=[
        CUDAExtension(
            name="qwen_dps_C",
            sources=["qwen_dps_ops.cpp", "qwen_dps_megakernel.cu"],
            include_dirs=[HERE],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    f"-gencode=arch=compute_{ARCH},code=sm_{ARCH}",
                    f"-DDPS_RING_BYTES={RING_BYTES}",
                    f"-DDPS_SSTAGES={SSTAGES}",
                    "--expt-relaxed-constexpr",
                    "-lineinfo",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
