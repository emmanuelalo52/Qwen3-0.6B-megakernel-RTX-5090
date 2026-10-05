"""
Builds the qwen_dps_C extension (dynamic-persistent Qwen3-0.6B megakernel).

    cd megakernel_dynamic/cuda
    python setup.py build_ext --inplace                  # B200 (sm_100a), the default
    DPS_ARCH=120a python setup.py build_ext --inplace    # RTX 5090
    DPS_ARCH=75   python setup.py build_ext --inplace    # older GPUs: atomic scheduler, emulated TMA
    DPS_TRACE=1   python setup.py build_ext --inplace    # + per-tile timing build (qwen_dps_trace_C)

DPS_ARCH picks one target. The weight ring gets DPS_RING_BYTES of shared memory
(default: what that GPU allows per block, minus ~30 KB for everything else); each
weight format (fp16 / fp8 / fp4) cuts it into as many stages as fit.

DPS_TRACE=1 builds a separate module, qwen_dps_trace_C, that records a timestamp
record per tile (see trace_dps.py). It sits next to qwen_dps_C, so the normal build
stays untouched and can still be benchmarked.
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
TRACE = os.environ.get("DPS_TRACE", "0") == "1"
NAME = "qwen_dps_trace_C" if TRACE else "qwen_dps_C"

HERE = os.path.dirname(os.path.abspath(__file__))

setup(
    name=NAME,
    ext_modules=[
        CUDAExtension(
            name=NAME,
            sources=["qwen_dps_ops.cpp", "qwen_dps_megakernel.cu"],
            include_dirs=[HERE],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": [
                    "-O3",
                    "-std=c++20",
                    f"-gencode=arch=compute_{ARCH},code=sm_{ARCH}",
                    f"-DDPS_RING_BYTES={RING_BYTES}",
                    f"-DDPS_SSTAGES={SSTAGES}",
                    f"-DDPS_TRACE={int(TRACE)}",
                    "--expt-relaxed-constexpr",
                    "-lineinfo",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    # separate object directories, so switching between the two builds stays incremental
    options={"build_ext": {"build_temp": os.path.join("build", "temp-" + NAME)}},
)
