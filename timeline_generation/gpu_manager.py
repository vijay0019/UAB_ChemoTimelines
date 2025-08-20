import threading
import time
from typing import Dict, Optional, Tuple
import random

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None


class GPUManager:
    """Thread-safe GPU allocation manager with intelligent load balancing."""

    def __init__(self, cache_duration: float = 0.1):
        self._lock = threading.RLock()
        self._gpu_assignments: Dict[str, int] = {}  # thread_id -> gpu_id mapping
        self._gpu_usage_count: Dict[int, int] = {}  # gpu_id -> usage count
        self._last_utilization_check = 0
        self._utilization_cache: Optional[Tuple[list, list]] = None
        self._cache_duration = cache_duration
        self._device_count = 1
        self._nvml_initialized = False

        self._initialize_nvml()

    def _initialize_nvml(self):
        """Initialize NVML if available."""
        if not NVML_AVAILABLE:
            print("Warning: pynvml not available, using single GPU mode")
            return

        try:
            pynvml.nvmlInit()
            self._device_count = pynvml.nvmlDeviceGetCount()
            self._nvml_initialized = True
            print(f"GPU Manager initialized with {self._device_count} GPU(s)")
        except Exception as e:
            print(f"Warning: Failed to initialize NVML: {e}, falling back to single GPU")
            self._device_count = 1
            self._nvml_initialized = False

    def _get_gpu_utilization(self) -> Tuple[list, list]:
        """Get GPU utilization with caching."""
        current_time = time.time()

        with self._lock:
            # Return cached data if still valid
            if (self._utilization_cache is not None and
                current_time - self._last_utilization_check < self._cache_duration):
                return self._utilization_cache

            if self._nvml_initialized and self._device_count > 1:
                try:
                    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self._device_count)]
                    utils = [pynvml.nvmlDeviceGetUtilizationRates(handle) for handle in handles]
                    mems = [pynvml.nvmlDeviceGetMemoryInfo(handle) for handle in handles]

                    self._utilization_cache = (utils, mems)
                    self._last_utilization_check = current_time
                    return utils, mems
                except Exception as e:
                    print(f"Warning: Failed to get GPU utilization: {e}")

            # Fallback: return dummy data for single GPU
            dummy_util = type('obj', (object,), {'gpu': 0})()
            dummy_mem = type('obj', (object,), {'used': 0, 'total': 1})()
            self._utilization_cache = ([dummy_util], [dummy_mem])
            self._last_utilization_check = current_time
            return self._utilization_cache

    def get_least_utilized_gpu(self) -> int:
        """Return the index of the least utilized GPU (ignores sticky assignments)."""
        if self._device_count == 1:
            return 0

        try:
            utils, mems = self._get_gpu_utilization()
            return min(
                range(self._device_count),
                key=lambda i: (utils[i].gpu, mems[i].used, random.random())
            )
        except Exception as e:
            print(f"Warning: Error in NVML-based GPU selection: {e}, using round-robin")
            return len(self._gpu_assignments) % self._device_count

    def get_optimal_gpu(self, thread_id: str) -> int:
        """Get optimal GPU for thread with sticky assignment and load balancing."""
        with self._lock:
            # Check if thread already has a good GPU assignment
            if thread_id in self._gpu_assignments:
                assigned_gpu = self._gpu_assignments[thread_id]
                if self._is_gpu_still_optimal(assigned_gpu):
                    return assigned_gpu

            # Otherwise, find best GPU
            best_gpu = self._find_best_gpu()

            # Update assignments
            old_gpu = self._gpu_assignments.get(thread_id)
            if old_gpu is not None and old_gpu in self._gpu_usage_count:
                self._gpu_usage_count[old_gpu] = max(0, self._gpu_usage_count[old_gpu] - 1)

            self._gpu_assignments[thread_id] = best_gpu
            self._gpu_usage_count[best_gpu] = self._gpu_usage_count.get(best_gpu, 0) + 1

            return best_gpu

    def _is_gpu_still_optimal(self, gpu_id: int) -> bool:
        """Check if currently assigned GPU is still a good choice."""
        if self._device_count == 1:
            return True

        try:
            utils, mems = self._get_gpu_utilization()
            if gpu_id >= len(utils):
                return False

            gpu_util = utils[gpu_id].gpu
            memory_util = mems[gpu_id].used / mems[gpu_id].total if mems[gpu_id].total > 0 else 0

            return gpu_util < 80 and memory_util < 0.9
        except Exception:
            return True  # Fallback to keeping current assignment

    def _find_best_gpu(self) -> int:
        """Find the best GPU based on utilization and current assignments."""
        if self._device_count == 1:
            return 0

        try:
            utils, mems = self._get_gpu_utilization()

            gpu_scores = []
            for i in range(self._device_count):
                gpu_util = utils[i].gpu if i < len(utils) else 0
                memory_util = mems[i].used / mems[i].total if i < len(mems) and mems[i].total > 0 else 0
                usage_count = self._gpu_usage_count.get(i, 0)


-                    gpu_util * 0.4 +           # GPU utilization weight
-                    memory_util * 100 * 0.4 +  # Memory utilization weight (normalized to 0-100)
-                    usage_count * 10 * 0.2 +   # Current assignment weight
-                    random.random() * 0.1      # Small random factor for tie-breaking


                score = (
                    gpu_util * 0.4 +             # GPU utilization weight
                    memory_util * 100 * 0.4 +    # Memory utilization weight (normalized to 0-100)
                    usage_count * 10 * 0.2 +     # Current assignment weight
                    random.random() * 0.1        # Small random factor for tie-breaking
                )
                gpu_scores.append((score, i))

            gpu_scores.sort()
            return gpu_scores[0][1]

        except Exception as e:
            print(f"Warning: Error in GPU scoring: {e}, using round-robin")
            return len(self._gpu_assignments) % self._device_count

    def release_gpu(self, thread_id: str):
        """Release GPU assignment for thread."""
        with self._lock:
            if thread_id in self._gpu_assignments:
                gpu_id = self._gpu_assignments[thread_id]
                if gpu_id in self._gpu_usage_count:
                    self._gpu_usage_count[gpu_id] = max(0, self._gpu_usage_count[gpu_id] - 1)
                del self._gpu_assignments[thread_id]

    def get_device_count(self) -> int:
        """Get number of available GPUs."""
        return self._device_count

    def get_stats(self) -> Dict:
        """Get current GPU manager statistics."""
        with self._lock:
            return {
                'device_count': self._device_count,
                'active_assignments': len(self._gpu_assignments),
                'gpu_usage_count': dict(self._gpu_usage_count),
                'nvml_available': self._nvml_initialized
            }


# Global GPU manager instance
_gpu_manager = None
_gpu_manager_lock = threading.Lock()


def get_gpu_manager() -> GPUManager:
    """Get the global GPU manager instance (singleton)."""
    global _gpu_manager
    if _gpu_manager is None:
        with _gpu_manager_lock:
            if _gpu_manager is None:
                _gpu_manager = GPUManager()
    return _gpu_manager

