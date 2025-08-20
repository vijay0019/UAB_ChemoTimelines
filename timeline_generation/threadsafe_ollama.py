import threading
import os
from typing import Tuple, List
import random
import time
import socket
import requests

import dspy
import pynvml

from gpu_manager import get_gpu_manager


def detect_running_ollama_ports(candidate_ports: Tuple[int, ...], timeout: float = 2.0) -> Tuple[int, ...]:
    """Detect which Ollama ports are actually running and accessible."""
    running_ports = []
    
    for port in candidate_ports:
        try:
            # First check if port is listening
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            
            if result == 0:  # Port is open
                # Verify it's actually Ollama by checking API endpoint
                try:
                    response = requests.get(f"http://127.0.0.1:{port}/api/tags", timeout=timeout)
                    if response.status_code == 200:
                        running_ports.append(port)
                        print(f"Detected running Ollama instance on port {port}")
                    else:
                        print(f"Port {port} is open but not responding as Ollama (status: {response.status_code})")
                except requests.exceptions.RequestException as e:
                    print(f"Port {port} is open but failed Ollama API check: {e}")
            else:
                print(f"Port {port} is not accessible")
                
        except Exception as e:
            print(f"Failed to check port {port}: {e}")
    
    return tuple(running_ports)


class ThreadSafeOllamaLM(dspy.LM):
    """Thread-safe Ollama Language Model with intelligent GPU allocation."""

    def __init__(self, ports: Tuple[int, ...] = None, **kwargs):
        self.lms = []
        self.kwargs = kwargs
        self._thread_local = threading.local()

        # Get candidate ports from environment or use defaults
        if ports is None:
            ports_str = os.getenv('OLLAMA_PORTS', '11434,11435,11436,11437,11438')
            try:
                candidate_ports = tuple(int(p.strip()) for p in ports_str.split(',') if p.strip())
            except ValueError:
                print(f"Warning: Invalid OLLAMA_PORTS format '{ports_str}', using default")
                candidate_ports = (11434, 11435, 11436, 11437, 11438)
        else:
            candidate_ports = ports

        # Detect which ports are actually running Ollama
        print(f"Checking candidate Ollama ports: {candidate_ports}")
        running_ports = detect_running_ollama_ports(candidate_ports)
        
        if not running_ports:
            print(f"No running Ollama instances found on candidate ports: {candidate_ports}")
            print("Attempting fallback to common ports: 11434, 8080, 8000")
            fallback_ports = (11434, 8080, 8000)
            running_ports = detect_running_ollama_ports(fallback_ports)
        
        if not running_ports:
            raise RuntimeError(f"No accessible Ollama instances found on any port. Checked: {candidate_ports + (11434, 8080, 8000)}")
        
        print(f"Using Ollama ports: {running_ports}")

        # Initialize language models for each running port
        for port in running_ports:
            try:
                lm = dspy.LM(base_url=f"http://127.0.0.1:{port}", **kwargs)
                self.lms.append(lm)
                print(f"Successfully initialized LM on port {port}")
            except Exception as e:
                print(f"Warning: Failed to initialize LM on verified port {port}: {e}")

        if not self.lms:
            raise RuntimeError(f"Failed to initialize any language models on verified ports: {running_ports}")

        # Store model name (removing from kwargs to avoid passing it to LM)
        self.model = kwargs.pop("model", "unknown")

        # Initialize NVML if available
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError:
            print("Warning: NVIDIA GPU not available, falling back to round-robin scheduling")
            self.device_count = len(self.lms)

        # Get GPU manager (external wrapper, can use NVML internally)
        self.gpu_manager = get_gpu_manager()

        print(f"ThreadSafeOllamaLM initialized with {len(self.lms)} LM(s) and {self.device_count} GPU(s)")

    def _get_thread_id(self) -> str:
        """Get unique identifier for current thread."""
        return f"thread_{threading.get_ident()}"

    def _get_nvml_utilization(self):
        """Query NVML for GPU utilization and memory usage."""
        utils, mems = [], []
        for i in range(self.device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                utils.append(pynvml.nvmlDeviceGetUtilizationRates(handle))
                mems.append(pynvml.nvmlDeviceGetMemoryInfo(handle))
            except pynvml.NVMLError:
                utils.append(type("util", (), {"gpu": 100}))  # pretend maxed
                mems.append(type("mem", (), {"used": float("inf")}))
        return utils, mems

    def _get_optimal_lm_index(self) -> int:
        """Get optimal language model index for current thread."""
        thread_id = self._get_thread_id()

        if self.device_count > 1:
            # Try NVML-aware GPU scheduling
            try:
                utils, mems = self._get_nvml_utilization()
                # Choose least utilized GPU
                gpu_idx = min(
                    range(self.device_count),
                    key=lambda i: (utils[i].gpu, mems[i].used, random.random())
                )
                return gpu_idx % len(self.lms)
            except Exception:
                pass

        # Fallback: use GPU manager (thread-based round-robin or similar)
        optimal_gpu = self.gpu_manager.get_optimal_gpu(thread_id)
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


def create_threadsafe_models(model_name: str,
                             context_window: int,
                             temperature_range: Tuple[float, float],
                             num_models: int,
                             **extra_kwargs) -> List[ThreadSafeOllamaLM]:
    """Create multiple ThreadSafeOllamaLM instances with different temperatures."""

    min_temp, max_temp = temperature_range
    models = []

    # Get candidate ports from environment
    ports_str = os.getenv('OLLAMA_PORTS', '11434,11435,11436,11437,11438')
    try:
        candidate_ports = tuple(int(p.strip()) for p in ports_str.split(',') if p.strip())
    except ValueError:
        candidate_ports = (11434, 11435, 11436, 11437, 11438)
    
    # Detect running Ollama instances (only once for efficiency)
    available_ports = detect_running_ollama_ports(candidate_ports)
    
    if not available_ports:
        print("No running Ollama instances found, trying fallback ports...")
        fallback_ports = (11434, 8080, 8000)
        available_ports = detect_running_ollama_ports(fallback_ports)
    
    if not available_ports:
        raise RuntimeError(f"No accessible Ollama instances found. Checked: {candidate_ports + (11434, 8080, 8000)}")
    
    print(f"create_threadsafe_models: Using Ollama ports: {available_ports}")

    for i in range(num_models):
        # Calculate temperature for this model
        if num_models > 1:
            temperature = min_temp + (max_temp - min_temp) * i / (num_models - 1)
        else:
            temperature = min_temp

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

