import gc
import torch

class MemoryManager:
    @staticmethod
    def clear_memory():
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    @staticmethod
    def get_memory_usage():
        import psutil
        memory = psutil.virtual_memory()
        used_gb = memory.used // (1024 ** 3)
        total_gb = memory.total // (1024 ** 3)
        return f"Memory: {memory.percent:.1f}% used ({used_gb}GB / {total_gb}GB)"
