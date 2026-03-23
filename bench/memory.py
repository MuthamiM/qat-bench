import psutil
import torch
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    vram_mb = 0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    return {"ram_mb": ram_mb, "vram_mb": vram_mb}

def get_model_size_mb(model_path):
    if not os.path.exists(model_path):
        return 0.0
    return os.path.getsize(model_path) / (1024 * 1024)
