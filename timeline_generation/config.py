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
    
    # Prompt Optimization Configuration
    ENABLE_PROMPT_OPTIMIZATION = get_bool_env('CHEMO_ENABLE_PROMPT_OPTIMIZATION', False)
    PROMPT_OPTIMIZER = os.getenv('CHEMO_PROMPT_OPTIMIZER', 'simba').lower()
    
    # SIMBA Optimizer Configuration
    SIMBA_BSIZE = get_int_env('CHEMO_SIMBA_BSIZE', 10)
    SIMBA_NUM_CANDIDATES = get_int_env('CHEMO_SIMBA_NUM_CANDIDATES', 4)
    SIMBA_MAX_STEPS = get_int_env('CHEMO_SIMBA_MAX_STEPS', 4)
    SIMBA_NUM_THREADS = get_int_env('CHEMO_SIMBA_NUM_THREADS', 1)
    
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
        print(f"Prompt Optimization: {cls.ENABLE_PROMPT_OPTIMIZATION}")
        if cls.ENABLE_PROMPT_OPTIMIZATION:
            print(f"  Optimizer: {cls.PROMPT_OPTIMIZER}")
            print(f"  SIMBA Config: bsize={cls.SIMBA_BSIZE}, candidates={cls.SIMBA_NUM_CANDIDATES}, steps={cls.SIMBA_MAX_STEPS}, threads={cls.SIMBA_NUM_THREADS}")
        else:
            print("  ⚠️  WARNING: Prompt optimization is DISABLED")
            print("     This may result in lower accuracy!")
        print("====================================")
    
    @classmethod
    def warn_if_optimization_disabled(cls):
        """Print a warning if prompt optimization is disabled."""
        if not cls.ENABLE_PROMPT_OPTIMIZATION:
            print()
            print("🚨 IMPORTANT NOTICE 🚨")
            print("Prompt optimization is currently DISABLED!")
            print("This will likely result in lower model accuracy.")
            print()
            print("To enable prompt optimization, run:")
            print("  export CHEMO_ENABLE_PROMPT_OPTIMIZATION=true")
            print("  export CHEMO_PROMPT_OPTIMIZER=simba")
            print()
            print("Or set them in your environment before running the script.")
            print("="*50)
            print()


# Backward compatibility - expose common constants at module level
MODEL = Config.MODEL
CONTEXT_WINDOW = Config.CONTEXT_WINDOW
MIN_TEMPERATURE = Config.MIN_TEMPERATURE
MAX_TEMPERATURE = Config.MAX_TEMPERATURE
MAX_RETRIES = Config.MAX_RETRIES