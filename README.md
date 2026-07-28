# ix-distributed-mlx

Distributed Mandelbrot set computation using **MLX** — Apple's array framework for machine learning on Apple Silicon.

## Overview

This repository demonstrates how to use MLX's distributed communication primitives ([`mx.distributed`](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)) to split a compute-intensive workload across multiple processes.  The workload is a classic Mandelbrot set renderer implemented with vectorised MLX array operations.

## The Program

`mpi-mandelbrot.py` distributes the image rows evenly across all available processes:

1. Each process initialises the distributed group via `mx.distributed.init()`.
2. The complex-plane grid is built with `mx.linspace`.
3. Each process extracts its row slice and computes the Mandelbrot recurrence (`z ← z² + c`) using vectorised MLX operations on the GPU.
4. Results are combined with `mx.distributed.all_gather()`.
5. Rank 0 reports the stats and optionally saves the result as a NumPy `.npy` file.

## Requirements

- macOS (Apple Silicon) or a Linux machine with CUDA GPUs
- Python 3.10+
- `mlx` (installed via `pip install mlx`)

For MPI backend specifically:
- OpenMPI (`brew install openmpi` on macOS)

## Usage

### Single process (no distributed communication)

```bash
python mpi-mandelbrot.py
```

All `mx.distributed` operations are no-ops when the group size is 1 — the program works identically.

### Multiple processes on localhost

```bash
# MLX launcher with MPI backend
mlx.launch --backend mpi -n 4 -- python mpi-mandelbrot.py

# MLX launcher with ring backend (TCP, no MPI needed)
mlx.launch --backend ring -n 4 -- python mpi-mandelbrot.py

# Direct mpirun
mpirun -np 4 python mpi-mandelbrot.py
```

### Multiple hosts

Create a JSON hostfile (see [MLX docs](https://ml-explore.github.io/mlx/build/html/usage/launching_distributed.html)) and launch:

```bash
mlx.launch --hostfile hostfile.json -- python mpi-mandelbrot.py
```

## How the Distributed Mandelbrot Works

```
┌───────────────────────────────────────────────┐
│                  Full Image                    │
│  ┌───────┬───────┬───────┬───────┬───────┐    │
│  │Rank 0 │Rank 1 │Rank 2 │Rank 3 │Rank 4 │    │
│  │chunk  │chunk  │chunk  │chunk  │chunk  │    │
│  └───────┴───────┴───────┴───────┴───────┘    │
│                    ▲                           │
│                    │ all_gather                │
│  ┌───────┐  ┌───────┐  ┌───────┐              │
│  │Rank 0 │  │Rank 1 │  │Rank 2 │  ...         │
│  │compute│  │compute│  │compute│              │
│  └───────┘  └───────┘  └───────┘              │
└───────────────────────────────────────────────┘
```

## References

- [MLX Distributed Communication docs](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
- [MLX Launching Distributed Programs](https://ml-explore.github.io/mlx/build/html/usage/launching_distributed.html)
- [MLX Distributed API Reference](https://ml-explore.github.io/mlx/build/html/python/distributed.html)
- [WWDC26: Explore distributed inference and training with MLX](https://developer.apple.com/videos/play/wwdc2026/233)