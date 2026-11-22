import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False


def get_array_module():
    """
    Returns CuPy if an NVIDIA GPU is available, otherwise falls back to NumPy.
    Ensures seamless execution across both GPU servers and local CPU laptops.
    """
    return cp


def check_hardware_status():
    """Returns a dictionary detailing the active compute backend."""
    if GPU_AVAILABLE:
        device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8")
        return {"backend": "CUDA GPU (CuPy)", "device": device_name}
    return {"backend": "CPU Fallback (NumPy)", "device": "Standard Host CPU"}


if __name__ == "__main__":
    print(f"Hardware check: {check_hardware_status()}")