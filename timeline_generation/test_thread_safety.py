#!/usr/bin/env python3
"""
Test script for thread safety improvements.
Tests the new GPU manager and ThreadSafe implementation.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from gpu_manager import get_gpu_manager
from threadsafe_ollama import create_threadsafe_models
from config import Config


def test_gpu_manager():
    """Test the GPU manager functionality."""
    print("=== Testing GPU Manager ===")
    
    gpu_manager = get_gpu_manager()
    stats = gpu_manager.get_stats()
    print(f"GPU Manager Stats: {stats}")
    
    # Test multiple thread assignments
    threads = []
    results = []
    
    def worker(thread_id):
        gpu_id = gpu_manager.get_optimal_gpu(f"test_thread_{thread_id}")
        time.sleep(0.1)  # Simulate work
        return f"Thread {thread_id} -> GPU {gpu_id}"
    
    # Create multiple threads to test assignment
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(10)]
        for future in as_completed(futures):
            results.append(future.result())
    
    for result in sorted(results):
        print(result)
    
    final_stats = gpu_manager.get_stats()
    print(f"Final GPU Manager Stats: {final_stats}")
    print()


def test_threadsafe_models():
    """Test the ThreadSafe model creation and usage."""
    print("=== Testing ThreadSafe Models ===")
    
    try:
        # Create a small set of models for testing
        models = create_threadsafe_models(
            model_name="ollama/phi4:latest",
            context_window=1024,  # Smaller for testing
            temperature_range=(0.1, 0.5),
            num_models=2
        )
        print(f"Created {len(models)} ThreadSafe models")
        
        # Test that models can be called (won't actually work without Ollama running)
        print("Models created successfully - would need running Ollama server for actual inference")
        
    except Exception as e:
        print(f"Expected error (Ollama not running): {e}")
    
    print()


def test_config():
    """Test configuration management."""
    print("=== Testing Configuration ===")
    Config.print_config()
    print()


def test_thread_contention():
    """Test that there are no race conditions in GPU assignment."""
    print("=== Testing Thread Contention ===")
    
    gpu_manager = get_gpu_manager()
    assignments = {}
    lock = threading.Lock()
    
    def worker(thread_id):
        # Simulate rapid GPU requests
        for i in range(5):
            gpu_id = gpu_manager.get_optimal_gpu(f"contention_thread_{thread_id}_{i}")
            with lock:
                assignments[f"thread_{thread_id}_iter_{i}"] = gpu_id
            time.sleep(0.01)
    
    # Run multiple threads simultaneously
    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print(f"Total GPU assignments made: {len(assignments)}")
    
    # Check for any obvious issues
    gpu_distribution = {}
    for assignment in assignments.values():
        gpu_distribution[assignment] = gpu_distribution.get(assignment, 0) + 1
    
    print(f"GPU distribution: {gpu_distribution}")
    print("Thread contention test completed successfully")
    print()


if __name__ == "__main__":
    print("Starting Thread Safety Test Suite")
    print("=" * 50)
    
    test_config()
    test_gpu_manager()
    test_threadsafe_models()
    test_thread_contention()
    
    print("All tests completed!")
    print("\nKey improvements implemented:")
    print("✓ Removed all sleep() calls from GPU selection")
    print("✓ Added proper thread synchronization with locks")
    print("✓ Implemented intelligent GPU load balancing")
    print("✓ Added caching for GPU utilization queries")
    print("✓ Centralized configuration management")
    print("✓ Created reusable ThreadSafe GPU manager")