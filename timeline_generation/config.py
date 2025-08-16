"""
Configuration management for ChemoTimelines project.
Centralizes all configuration settings with environment variable support.
"""
import os
from typing import Tuple, List


def get_bool_env(key: str, default: bool) -> bool:
    """Get boolean value from environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_int_env(key: str, default: int) -> int:
    """Get integer value from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_float_env(key: str, default: float) -> float:
    """Get float value from environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_list_env(key: str, default: List[int]) -> List[int]:
    """Get list of integers from environment variable (comma-separated)."""
    value = os.getenv(key)
    if not value:
        return default
    try:
        return [int(x.strip()) for x in value.split(',') if x.strip()]
    except ValueError:
        return default


class Config:
    """Centralized configuration for ChemoTimelines."""
    
    # Model Configuration
    MODEL = os.getenv('CHEMO_MODEL', 'ollama/phi4:latest')
    CONTEXT_WINDOW = get_int_env('CHEMO_CONTEXT_WINDOW', 16384)
    
    # Temperature Configuration
    MIN_TEMPERATURE = get_float_env('CHEMO_MIN_TEMPERATURE', 0.2)
    MAX_TEMPERATURE = get_float_env('CHEMO_MAX_TEMPERATURE', 1.0)
    MAX_RETRIES = get_int_env('CHEMO_MAX_RETRIES', 8)
    
    # Ollama Configuration
    OLLAMA_PORTS = get_list_env('OLLAMA_PORTS', [11435, 11436, 11437, 11438])
    
    # Repeat Penalty Configuration (for task2-style models)
    DEFAULT_REPEAT_PENALTY = get_float_env('CHEMO_DEFAULT_REPEAT_PENALTY', 1.1)
    DEFAULT_REPEAT_LAST_N = get_int_env('CHEMO_DEFAULT_REPEAT_LAST_N', 64)
    LOW_REP_REPEAT_PENALTY = get_float_env('CHEMO_LOW_REP_REPEAT_PENALTY', 1.25)
    LOW_REP_REPEAT_LAST_N = get_int_env('CHEMO_LOW_REP_REPEAT_LAST_N', 160)
    
    # Threading Configuration
    NUM_THREADS = get_int_env('CHEMO_NUM_THREADS', 1)  # Default to 1 for safety
    ENABLE_THREADING = get_bool_env('CHEMO_ENABLE_THREADING', True)
    
    # GPU Configuration
    GPU_CACHE_DURATION = get_float_env('CHEMO_GPU_CACHE_DURATION', 0.1)  # 100ms
    ENABLE_GPU_MONITORING = get_bool_env('CHEMO_ENABLE_GPU_MONITORING', True)
    
    # Performance Configuration
    TOKEN_CACHE_SIZE = get_int_env('CHEMO_TOKEN_CACHE_SIZE', 1000)
    ENABLE_TOKEN_CACHE = get_bool_env('CHEMO_ENABLE_TOKEN_CACHE', True)
    
    @classmethod
    def get_ollama_ports_tuple(cls) -> Tuple[int, ...]:
        """Get Ollama ports as tuple for backward compatibility."""
        return tuple(cls.OLLAMA_PORTS)
    
    @classmethod
    def print_config(cls):
        """Print current configuration values."""
        print("=== ChemoTimelines Configuration ===")
        print(f"Model: {cls.MODEL}")
        print(f"Context Window: {cls.CONTEXT_WINDOW}")
        print(f"Temperature Range: {cls.MIN_TEMPERATURE} - {cls.MAX_TEMPERATURE}")
        print(f"Max Retries: {cls.MAX_RETRIES}")
        print(f"Ollama Ports: {cls.OLLAMA_PORTS}")
        print(f"Threading Enabled: {cls.ENABLE_THREADING}")
        print(f"Num Threads: {cls.NUM_THREADS}")
        print(f"GPU Monitoring: {cls.ENABLE_GPU_MONITORING}")
        print("====================================")


# Backward compatibility - expose common constants at module level
MODEL = Config.MODEL
CONTEXT_WINDOW = Config.CONTEXT_WINDOW
MIN_TEMPERATURE = Config.MIN_TEMPERATURE
MAX_TEMPERATURE = Config.MAX_TEMPERATURE
MAX_RETRIES = Config.MAX_RETRIES