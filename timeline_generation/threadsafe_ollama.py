import threading
import os
from typing import Tuple, List
import dspy

from gpu_manager import get_gpu_manager


class ThreadSafeOllamaLM(dspy.LM):
    """Thread-safe Ollama Language Model with intelligent GPU allocation."""
    
    def __init__(self, ports: Tuple[int, ...] = None, **kwargs):
        self.lms = []
        self.kwargs = kwargs
        self._thread_local = threading.local()
        
        # Get ports from environment or use defaults
        if ports is None:
            ports_str = os.getenv('OLLAMA_PORTS', '11435,11436,11437,11438')
            try:
                ports = tuple(int(p.strip()) for p in ports_str.split(',') if p.strip())
            except ValueError:
                print(f"Warning: Invalid OLLAMA_PORTS format '{ports_str}', using default")
                ports = (11435, 11436, 11437, 11438)
        
        # Initialize language models for each port
        for port in ports:
            try:
                lm = dspy.LM(base_url=f"http://127.0.0.1:{port}", **kwargs)
                self.lms.append(lm)
            except Exception as e:
                print(f"Warning: Failed to initialize LM on port {port}: {e}")
        
        if not self.lms:
            raise RuntimeError("No language models could be initialized")
        
        # Store model name (removing from kwargs to avoid passing it to LM)
        self.model = kwargs.pop("model", "unknown")
        
        # Get GPU manager
        self.gpu_manager = get_gpu_manager()
        
        print(f"ThreadSafeOllamaLM initialized with {len(self.lms)} LM(s) and {self.gpu_manager.get_device_count()} GPU(s)")
    
    def _get_thread_id(self) -> str:
        """Get unique identifier for current thread."""
        return f"thread_{threading.get_ident()}"
    
    def _get_optimal_lm_index(self) -> int:
        """Get optimal language model index for current thread."""
        thread_id = self._get_thread_id()
        
        # Use GPU manager to get optimal GPU, then map to LM
        optimal_gpu = self.gpu_manager.get_optimal_gpu(thread_id)
        
        # Map GPU to LM index (round-robin if more LMs than GPUs)
        lm_index = optimal_gpu % len(self.lms)
        
        return lm_index
    
    def __call__(self, **kwargs):
        """Make a call to the optimal language model."""
        lm_idx = self._get_optimal_lm_index()
        return self.lms[lm_idx](**kwargs)
    
    def generate(self, **kwargs):
        """Generate using the optimal language model."""
        lm_idx = self._get_optimal_lm_index()
        return self.lms[lm_idx].generate(**kwargs)
    
    def __del__(self):
        """Cleanup: release GPU assignment when instance is destroyed."""
        try:
            thread_id = self._get_thread_id()
            self.gpu_manager.release_gpu(thread_id)
        except Exception:
            pass  # Ignore cleanup errors


def create_threadsafe_models(model_name: str, context_window: int, 
                           temperature_range: Tuple[float, float], 
                           num_models: int, **extra_kwargs) -> List[ThreadSafeOllamaLM]:
    """Create multiple ThreadSafeOllamaLM instances with different temperatures."""
    
    min_temp, max_temp = temperature_range
    models = []
    
    # Get ports from environment
    ports_str = os.getenv('OLLAMA_PORTS', '11435,11436,11437,11438')
    try:
        available_ports = tuple(int(p.strip()) for p in ports_str.split(',') if p.strip())
    except ValueError:
        available_ports = (11435, 11436, 11437, 11438)
    
    for i in range(num_models):
        # Calculate temperature for this model
        if num_models > 1:
            temperature = min_temp + (max_temp - min_temp) * i / (num_models - 1)
        else:
            temperature = min_temp
        
        # Create model kwargs
        model_kwargs = {
            "model": model_name,
            "max_tokens": context_window,
            "num_ctx": context_window,
            "temperature": temperature,
            "seed": i,
            **extra_kwargs
        }
        
        try:
            model = ThreadSafeOllamaLM(ports=available_ports, **model_kwargs)
            models.append(model)
        except Exception as e:
            print(f"Warning: Failed to create model {i} with temperature {temperature}: {e}")
    
    if not models:
        raise RuntimeError("Failed to create any ThreadSafe models")
    
    return models