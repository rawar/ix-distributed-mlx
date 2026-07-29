#!/usr/bin/env python3
"""
Simple All-Reduce Example with MLX Distributed

This is a minimal example demonstrating RDMA benefits in distributed computing.
Each node contributes a value, and all nodes receive the sum.

RDMA Advantage:
- Direct memory access between nodes
- Lower latency for collective operations
- Minimal CPU overhead

Usage:
    # Local testing
    mlx.launch -n 2 simple_allreduce.py
    
    # With RDMA (JACCL)
    mlx.launch --backend jaccl --hostfile jaccl-config.json simple_allreduce.py
    
    # Without RDMA (Ring)
    mlx.launch --hosts ip1,ip2 simple_allreduce.py
"""

import mlx.core as mx
import mlx.core.distributed as dist
import time


def main():
    # Initialize distributed context
    dist.init()
    
    rank = dist.rank()
    world_size = dist.world_size()
    
    print(f"[Rank {rank}] Hello from rank {rank}/{world_size}", flush=True)
    
    # Each rank contributes its rank number
    local_value = mx.array([rank], dtype=mx.float32)
    
    print(f"[Rank {rank}] Local value: {float(local_value[0])}", flush=True)
    
    # Warm up
    _ = dist.all_sum(local_value)
    mx.eval(_)
    
    # Benchmark all_reduce operation
    iterations = 100
    times = []
    
    for i in range(iterations):
        start = time.perf_counter()
        result = dist.all_sum(local_value)
        mx.eval(result)
        end = time.perf_counter()
        times.append(end - start)
    
    # Remove warmup iteration
    times = times[1:]
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # Expected result: sum of all ranks (0 + 1 + ... + (world_size-1))
    expected_sum = (world_size * (world_size - 1)) // 2
    actual_sum = float(result[0])
    
    # Print results
    print(f"\n[Rank {rank}] Results:", flush=True)
    print(f"  Expected sum: {expected_sum}", flush=True)
    print(f"  Actual sum: {actual_sum}", flush=True)
    print(f"  Correct: {'✓' if abs(actual_sum - expected_sum) < 1e-6 else '✗'}", flush=True)
    print(f"\n[Rank {rank}] Performance ({iterations-1} iterations):", flush=True)
    print(f"  Average time: {avg_time*1000:.3f} ms", flush=True)
    print(f"  Min time: {min_time*1000:.3f} ms", flush=True)
    print(f"  Max time: {max_time*1000:.3f} ms", flush=True)
    
    if rank == 0:
        print("\n" + "="*60)
        print("RDMA PERFORMANCE COMPARISON")
        print("="*60)
        print("Expected performance:")
        print("  - JACCL (RDMA):  ~0.5-2 ms per all_reduce")
        print("  - Ring (TCP/IP): ~2-10 ms per all_reduce")
        print("  - Speedup:       2-5x with RDMA")
        print("="*60)


if __name__ == "__main__":
    main()
