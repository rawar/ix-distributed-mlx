#!/usr/bin/env python3
"""
Distributed Matrix Multiplication with MLX - RDMA Performance Demo

This example demonstrates the advantages of RDMA (Remote Direct Memory Access)
in a fully-connected mesh cluster using Apple's MLX.distributed framework.

The example performs distributed matrix multiplication where:
1. Each node computes a portion of the result matrix
2. Results are gathered using all_gather (benefits from RDMA)
3. Performance metrics show communication overhead

RDMA Advantages Demonstrated:
- Zero-copy data transfer between nodes
- Lower latency for collective operations
- Higher bandwidth utilization in mesh topology
- Reduced CPU overhead during communication

Usage:
    # Local testing (2 processes)
    mlx.launch -n 2 distributed_matmul.py

    # Multi-node with RDMA (JACCL backend over Thunderbolt)
    mlx.distributed_config --backend jaccl \
        --hosts node1,node2 --over thunderbolt \
        --auto-setup --output jaccl-config.json
    mlx.launch --backend jaccl --hostfile jaccl-config.json distributed_matmul.py

    # Multi-node without RDMA (Ring backend over Ethernet)
    mlx.launch --hosts ip1,ip2 distributed_matmul.py
"""

import mlx.core as mx
import mlx.core.distributed as dist
import time
from typing import Tuple


def print_rank(msg: str, rank: int):
    """Print message with rank prefix"""
    print(f"[Rank {rank}] {msg}", flush=True)


def create_test_matrices(size: int, rank: int) -> Tuple[mx.array, mx.array]:
    """
    Create test matrices for multiplication.
    Each rank gets different data to ensure computation is distributed.
    """
    # Seed based on rank for reproducibility but different data per rank
    mx.random.seed(42 + rank)
    
    A = mx.random.uniform(shape=(size, size))
    B = mx.random.uniform(shape=(size, size))
    
    return A, B


def distributed_matmul(A: mx.array, B: mx.array, rank: int, world_size: int) -> mx.array:
    """
    Perform distributed matrix multiplication.
    
    Strategy:
    1. Each rank computes a portion of rows (A_local @ B)
    2. Use all_gather to collect results from all ranks
    3. Concatenate to form final result
    
    This demonstrates RDMA benefits during the all_gather operation.
    """
    rows_per_rank = A.shape[0] // world_size
    start_row = rank * rows_per_rank
    end_row = start_row + rows_per_rank if rank < world_size - 1 else A.shape[0]
    
    # Each rank computes its portion
    A_local = A[start_row:end_row, :]
    local_result = A_local @ B
    
    # Synchronize before timing communication
    mx.eval(local_result)
    
    return local_result


def benchmark_communication(data: mx.array, iterations: int = 10) -> float:
    """
    Benchmark all_gather communication performance.
    This is where RDMA shows its advantages.
    """
    times = []
    
    for _ in range(iterations):
        # Ensure data is evaluated before timing
        mx.eval(data)
        
        start = time.perf_counter()
        gathered = dist.all_gather(data)
        mx.eval(gathered)  # Force evaluation
        end = time.perf_counter()
        
        times.append(end - start)
    
    # Remove first iteration (warmup)
    times = times[1:]
    avg_time = sum(times) / len(times)
    
    return avg_time


def main():
    # Initialize distributed context
    dist.init()
    
    rank = dist.rank()
    world_size = dist.world_size()
    
    print_rank(f"Initialized! World size: {world_size}", rank)
    
    # Configuration
    matrix_size = 2048  # Size of matrices
    benchmark_iters = 10
    
    # Create test matrices
    print_rank(f"Creating {matrix_size}x{matrix_size} matrices...", rank)
    A, B = create_test_matrices(matrix_size, rank)
    
    # Warm up
    print_rank("Warming up...", rank)
    _ = A @ B
    mx.eval(_)
    
    # Benchmark local computation
    print_rank("Benchmarking local computation...", rank)
    start = time.perf_counter()
    local_result = A @ B
    mx.eval(local_result)
    local_compute_time = time.perf_counter() - start
    
    # Perform distributed computation
    print_rank("Starting distributed matrix multiplication...", rank)
    
    start_total = time.perf_counter()
    
    # Compute local portion
    start_compute = time.perf_counter()
    local_result = distributed_matmul(A, B, rank, world_size)
    compute_time = time.perf_counter() - start_compute
    
    # Gather results (RDMA advantage here!)
    print_rank("Gathering results from all ranks...", rank)
    start_comm = time.perf_counter()
    gathered_results = dist.all_gather(local_result)
    mx.eval(gathered_results)
    comm_time = time.perf_counter() - start_comm
    
    total_time = time.perf_counter() - start_total
    
    # Benchmark pure communication overhead
    print_rank("Benchmarking communication performance...", rank)
    avg_comm_time = benchmark_communication(local_result, benchmark_iters)
    
    # Calculate data transfer size
    data_size_mb = (local_result.size * local_result.itemsize) / (1024 * 1024)
    bandwidth_mbps = data_size_mb / avg_comm_time if avg_comm_time > 0 else 0
    
    # Print results (only rank 0 for clarity)
    if rank == 0:
        print("\n" + "="*70)
        print("DISTRIBUTED MATRIX MULTIPLICATION RESULTS")
        print("="*70)
        print(f"Configuration:")
        print(f"  - World size: {world_size} nodes")
        print(f"  - Matrix size: {matrix_size}x{matrix_size}")
        print(f"  - Data per rank: {data_size_mb:.2f} MB")
        print(f"\nPerformance Metrics:")
        print(f"  - Local compute time: {local_compute_time*1000:.2f} ms")
        print(f"  - Distributed compute time: {compute_time*1000:.2f} ms")
        print(f"  - Communication time: {comm_time*1000:.2f} ms")
        print(f"  - Total distributed time: {total_time*1000:.2f} ms")
        print(f"\nCommunication Performance (avg over {benchmark_iters-1} iterations):")
        print(f"  - Average all_gather time: {avg_comm_time*1000:.2f} ms")
        print(f"  - Effective bandwidth: {bandwidth_mbps:.2f} MB/s")
        print(f"  - Communication overhead: {(comm_time/total_time)*100:.1f}%")
        print(f"\nRDMA Benefits:")
        print(f"  - With RDMA (JACCL): Lower latency, higher bandwidth")
        print(f"  - Without RDMA (Ring): Higher latency, TCP/IP overhead")
        print(f"  - Speedup potential: 2-5x for communication-heavy workloads")
        print("="*70)
        
        # Verify correctness (optional)
        print("\nVerifying correctness...")
        expected = A @ B
        actual = mx.concatenate(gathered_results, axis=0)
        max_diff = mx.max(mx.abs(expected - actual))
        print(f"Maximum difference: {float(max_diff):.2e}")
        print("✓ Results verified!" if float(max_diff) < 1e-4 else "✗ Verification failed!")
    
    # Synchronize before exit
    dist.all_sum(mx.array([rank]))


if __name__ == "__main__":
    main()
