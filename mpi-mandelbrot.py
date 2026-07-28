#!/usr/bin/env python3
"""
mpi-mandelbrot.py — Distributed Mandelbrot Set Computation with MLX
====================================================================

This program computes the Mandelbrot set across multiple processes using
MLX's distributed communication primitives.  The image rows are split
evenly among all available processes; each process computes its chunk
independently, then the results are gathered with `all_gather`.

Backends
--------
MLX supports several distributed backends — MPI, ring (TCP), JACCL
(RDMA over Thunderbolt), and NCCL (CUDA).  The program is backend-agnostic;
select the backend when launching.

Usage
-----
    # Single process (no-op distributed — useful for testing):
    python mpi-mandelbrot.py

    # With MLX launcher (MPI backend, 4 processes):
    mlx.launch --backend mpi -n 4 -- python mpi-mandelbrot.py

    # With mpirun directly:
    mpirun -np 4 python mpi-mandelbrot.py

    # With MLX ring backend (localhost test):
    mlx.launch --backend ring -n 4 -- python mpi-mandelbrot.py

    # Across remote hosts:
    mlx.launch --hosts host1,host2,host3,host4 -- python mpi-mandelbrot.py

Notes
-----
- When run with a single process (or if no backend is available), all
  `mx.distributed` operations are no-ops.  This lets you develop and debug
  locally without MPI installed.
- The image height is truncated to a multiple of the world size so that
  every process receives the same number of rows (required by `all_gather`).
"""

from __future__ import annotations

import time
import sys

import mlx.core as mx

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

WIDTH: int = 1920          # Image width  (pixels / x-axis samples)
HEIGHT: int = 1080         # Image height (pixels / y-axis samples)
MAX_ITER: int = 256        # Maximum iterations per point
ESCAPE_RADIUS: float = 20.0  # Divergence threshold (|z| > radius → escaped)

# Complex-plane bounding box
X_MIN, X_MAX = -2.5, 1.0
Y_MIN, Y_MAX = -1.25, 1.25


# ═══════════════════════════════════════════════════════════════════
# Core computation
# ═══════════════════════════════════════════════════════════════════

def compute_mandelbrot(
    x_vals: mx.array,
    y_chunk: mx.array,
    max_iter: int,
    escape_radius: float,
) -> mx.array:
    """Compute the Mandelbrot set for a rectangular region of the complex plane.

    Vectorised over the full grid — every pixel is evaluated in parallel
    on the GPU (or CPU) without any per-pixel Python loop.

    Args:
        x_vals: 1-D array of real-axis coordinates; length = image width.
        y_chunk: 1-D array of imaginary-axis coordinates for this rank's
            subset of rows.
        max_iter: Maximum number of iterations per point.
        escape_radius: Divergence threshold.

    Returns:
        2-D array of iteration counts with shape ``(len(y_chunk), len(x_vals))``.
    """
    # Build the complex grid:  c = x + i y  for every pixel in the chunk.
    # y_chunk is reshaped to a column vector so broadcasting yields a 2-D grid.
    # MLX handles the Python-complex → MLX-complex promotion automatically.
    c = x_vals + 1j * y_chunk.reshape(-1, 1)

    z = mx.zeros(c.shape, dtype=mx.complex64)
    counts = mx.zeros(c.shape, dtype=mx.int32)

    # Iterate the Mandelbrot recurrence  z ← z² + c.
    # Points that have escaped (|z| ≥ radius) are masked out so their
    # count stops increasing (they keep their final iteration value).
    for _ in range(max_iter):
        z = z * z + c
        still_bounded = mx.abs(z) < escape_radius
        counts = counts + still_bounded

    return counts


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    # ── 1. Initialise the distributed group ───────────────────────
    world = mx.distributed.init()
    rank = world.rank()
    size = world.size()

    # ── 2. Build coordinate grids ─────────────────────────────────
    # The height must be divisible by *size* because all_gather
    # requires every rank to contribute an identically-shaped array.
    adj_height = (HEIGHT // size) * size
    if adj_height < HEIGHT and rank == 0:
        print(
            f"Note: truncated height {HEIGHT} → {adj_height} "
            f"for uniform work distribution across {size} process(es).",
            file=sys.stderr,
        )

    x_vals = mx.linspace(X_MIN, X_MAX, WIDTH)
    y_vals = mx.linspace(Y_MIN, Y_MAX, adj_height)
    rows_per_rank = adj_height // size

    # ── 3. Each rank works on its own row slice ───────────────────
    start_row = rank * rows_per_rank
    y_chunk = y_vals[start_row : start_row + rows_per_rank]

    t0 = time.perf_counter()
    local_counts = compute_mandelbrot(x_vals, y_chunk, MAX_ITER, ESCAPE_RADIUS)
    mx.eval(local_counts)  # force execution so we can time it
    compute_time = time.perf_counter() - t0

    # ── 4. Gather every rank's chunk into the full image ──────────
    t1 = time.perf_counter()
    full_counts = mx.distributed.all_gather(local_counts, group=world)
    mx.eval(full_counts)
    gather_time = time.perf_counter() - t1

    # ── 5. Report summary ─────────────────────────────────────────
    if rank == 0:
        print(f"\n{'=' * 52}")
        print(f"  Distributed Mandelbrot — MLX")
        print(f"{'=' * 52}")
        print(f"  World size          : {size}")
        print(f"  Image dimensions    : {full_counts.shape[1]} × {full_counts.shape[0]}")
        print(f"  Max iterations      : {MAX_ITER}")
        print(f"  Rows per process    : {rows_per_rank}")
        print(f"  Compute time        : {compute_time:.3f} s")
        print(f"  Gather time         : {gather_time:.3f} s")
        print(f"  Total time          : {compute_time + gather_time:.3f} s")
        print(f"  Min iterations      : {mx.min(full_counts).item()}")
        print(f"  Max iterations      : {mx.max(full_counts).item()}")
        print(f"{'=' * 52}\n")

        # ── 6. Save result as a NumPy file for later visualisation ─
        # Convert to a NumPy-compatible array and save as .npy
        try:
            import numpy as np

            np_counts = np.array(full_counts)
            np.save("mandelbrot_result.npy", np_counts)
            print(f"  Saved result → mandelbrot_result.npy  ({np_counts.shape})")
        except ImportError:
            print("  [NumPy not available — skipping .npy export]")
        print()


if __name__ == "__main__":
    main()