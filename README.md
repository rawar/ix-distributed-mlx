# ix-distributed-mlx

Distributed computing examples using **MLX** — Apple's array framework for machine learning on Apple Silicon — with a focus on the [`mx.distributed`](https://ml-explore.github.io/mlx/build/html/usage/distributed.html) communication primitives and RDMA (Remote Direct Memory Access) performance.

## Examples

This repository contains three self-contained examples that progressively build from basic communication to compute-intensive distributed workloads:

| Example | What it demonstrates |
|---|---|
| [`simple_allreduce.py`](#1-simple-all-reduce-simple_allreducepy) | Minimal all-reduce benchmark — the "hello world" of distributed computing |
| [`distributed_matmul.py`](#2-distributed-matrix-multiplication-distributed_matmulpy) | Distributed matrix multiplication with communication benchmarking |
| [`mpi-mandelbrot.py`](#3-distributed-mandelbrot-mpi-mandelbrotpy) | Classic Mandelbrot set renderer split across processes |

---

### 1. Simple All-Reduce — `simple_allreduce.py`

A minimal example demonstrating RDMA benefits in distributed computing. Each node contributes a value, and all nodes receive the sum.

**RDMA Advantage:**
- Direct memory access between nodes
- Lower latency for collective operations
- Minimal CPU overhead

**Usage:**
```bash
# Local testing
mlx.launch -n 2 simple_allreduce.py

# With RDMA (JACCL)
mlx.launch --backend jaccl --hostfile jaccl-config.json simple_allreduce.py

# Without RDMA (Ring)
mlx.launch --hosts ip1,ip2 simple_allreduce.py
```

---

### 2. Distributed Matrix Multiplication — `distributed_matmul.py`

Demonstrates the advantages of RDMA in a fully-connected mesh cluster using MLX's distributed framework. Each node computes a portion of the result matrix, results are gathered using `all_gather`, and performance metrics show communication overhead.

**RDMA Advantages Demonstrated:**
- Zero-copy data transfer between nodes
- Lower latency for collective operations
- Higher bandwidth utilization in mesh topology
- Reduced CPU overhead during communication

**Usage:**
```bash
# Local testing (2 processes)
mlx.launch -n 2 distributed_matmul.py

# Multi-node with RDMA (JACCL backend over Thunderbolt)
mlx.distributed_config --backend jaccl \
  --hosts node1,node2 --over thunderbolt \
  --auto-setup --output jaccl-config.json
mlx.launch --backend jaccl --hostfile jaccl-config.json distributed_matmul.py

# Multi-node without RDMA (Ring backend over Ethernet)
mlx.launch --hosts ip1,ip2 distributed_matmul.py
```

---

### 3. Distributed Mandelbrot — `mpi-mandelbrot.py`

Distributes the image rows of a classic Mandelbrot set renderer — implemented with vectorised MLX array operations — evenly across all available processes.

**How it works:**
1. Each process initialises the distributed group via `mx.distributed.init()`.
2. The complex-plane grid is built with `mx.linspace`.
3. Each process extracts its row slice and computes the Mandelbrot recurrence (`z ← z² + c`) using vectorised MLX operations on the GPU.
4. Results are combined with `mx.distributed.all_gather()`.
5. Rank 0 reports the stats and optionally saves the result as a NumPy `.npy` file.

**Usage:**
```bash
# Single process (no distributed communication)
python mpi-mandelbrot.py

# Multiple processes on localhost
mlx.launch --backend mpi -n 4 -- python mpi-mandelbrot.py
mlx.launch --backend ring -n 4 -- python mpi-mandelbrot.py
mpirun -np 4 python mpi-mandelbrot.py

# Multiple hosts
mlx.launch --hostfile hostfile.json -- python mpi-mandelbrot.py
```

All `mx.distributed` operations are no-ops when the group size is 1 — the program works identically in single-process mode.

### How the Distributed Mandelbrot Works

```
┌───────────────────────────────────────────────┐
│                  Full Image                    │
│  ┌───────┬───────┬───────┬───────┬───────┐   │
│  │Rank 0 │Rank 1 │Rank 2 │Rank 3 │Rank 4 │   │
│  │ chunk │ chunk │ chunk │ chunk │ chunk │   │
│  └───────┴───────┴───────┴───────┴───────┘   │
│        ▲                                      │
│        │            all_gather                 │
│  ┌───────┐  ┌───────┐  ┌───────┐             │
│  │Rank 0 │  │Rank 1 │  │Rank 2 │  ...         │
│  │compute│  │compute│  │compute│              │
│  └───────┘  └───────┘  └───────┘             │
└───────────────────────────────────────────────┘
```

## Requirements

- macOS (Apple Silicon) or a Linux machine with CUDA GPUs
- Python 3.10+
- `mlx` (installed via `pip install mlx`)

For MPI backend specifically:
- OpenMPI (`brew install openmpi` on macOS)

## References

- [MLX Distributed Communication docs](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
- [MLX Launching Distributed Programs](https://ml-explore.github.io/mlx/build/html/usage/launching_distributed.html)
- [MLX Distributed API Reference](https://ml-explore.github.io/mlx/build/html/python/distributed.html)
- [WWDC26: Explore distributed inference and training with MLX](https://developer.apple.com/videos/play/wwdc2026/233)