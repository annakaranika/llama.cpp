"""
GPU/CPU Performance Benchmarking Module for LLM Inference

This module provides tools to measure and estimate the effective computational performance
(in GFLOPs) of hardware devices for Large Language Model (LLM) inference workloads.

The module performs matrix multiplication benchmarks across various sizes commonly used
in LLM operations and applies efficiency factors to estimate realistic inference performance.
This is useful for hardware profiling, performance optimization, and resource allocation
in LLM deployment scenarios.

Key Features:
- Matrix multiplication benchmarking with configurable sizes and repetitions
- LLM-specific performance estimation with realistic efficiency factors
- Support for both raw matmul and inference-adjusted workload measurements
- Robust timing measurements using median values from multiple runs
- Command-line interface for easy benchmarking

Typical Usage:
    # Benchmark device for LLM inference
    python gflops_measure.py --workload inference

    # Raw matrix multiplication benchmark
    python gflops_measure.py --workload matmul

The output provides estimated GFLOPs values that can be used to configure
device performance parameters in LLM inference systems.
"""

import time

import argparse
import numpy as np


def measure_matmul_gflops(sizes=None, warmup=3, runs=5):
    """
    Measure matrix multiplication GFLOPs for different sizes.
    LLM inference uses various matrix sizes, so test a range.
    """
    if sizes is None:
        sizes = [512, 1024, 2048, 4096]
    results = {}

    for N in sizes:  # pylint: disable=invalid-name
        print(f"Testing {N}x{N} matrices...")

        # Generate matrices (using float32 like most LLM inference)
        A = np.random.rand(N, N).astype(np.float32)  # pylint: disable=invalid-name
        B = np.random.rand(N, N).astype(np.float32)  # pylint: disable=invalid-name

        # Warmup runs to get CPU/cache warmed up
        for _ in range(warmup):
            _ = A @ B

        # Timing runs
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            C = A @ B  # pylint: disable=invalid-name
            # Prevent optimization away
            checksum = np.sum(C)  # pylint: disable=unused-variable
            end = time.perf_counter()
            times.append(end - start)

        # Use median time (more robust than mean)
        median_time = np.median(times)
        ops = 2 * N**3  # 2*N^3 FLOPs for N×N @ N×N matmul
        gflops = ops / median_time / 1e9

        results[N] = {
            "gflops": gflops,
            "time_ms": median_time * 1000,
            "all_times": times,
        }

        print(f"  {N}×{N}: {gflops:.2f} GFLOPS ({median_time*1000:.1f}ms)")

    return results


def estimate_llm_inference_gflops(matmul_results):
    """
    Estimate effective GFLOPs for LLM inference based on matmul benchmarks.

    LLM inference characteristics:
    - Mixed matrix sizes (attention projections, FFN, etc.)
    - Memory bandwidth limited (not compute limited)
    - Lower arithmetic intensity than pure matmul
    """

    # Weighted average based on typical LLM operation distribution
    size_weights = {
        512: 0.1,  # Smaller embeddings, some attention computations
        1024: 0.3,  # Common hidden dimensions
        2048: 0.4,  # Typical d_model for many models
        4096: 0.2,  # Large FFN dimensions
    }

    weighted_gflops = 0.0
    total_weight = 0.0

    for size, weight in size_weights.items():
        if size in matmul_results:
            weighted_gflops += matmul_results[size]["gflops"] * weight
            total_weight += weight

    if total_weight == 0:
        return None

    average_matmul_gflops = weighted_gflops / total_weight

    # Apply LLM-specific efficiency factors
    llm_efficiency_factors = {
        "memory_bandwidth_limit": 0.6,  # Memory bandwidth is often the bottleneck
        "mixed_operations": 0.8,  # Not all ops are matmul (softmax, elementwise, etc.)
        "attention_overhead": 0.85,  # Attention has some non-matmul overhead
        "cache_effects": 0.9,  # Real inference has less optimal cache usage
        "quantization_speedup": 1.0,  # Assume FP32 for now, adjust if using quantization
    }

    # Apply efficiency factors
    llm_gflops = average_matmul_gflops
    for factor_name, factor_value in llm_efficiency_factors.items():
        llm_gflops *= factor_value
        print(f"After {factor_name}: {llm_gflops:.2f} GFLOPS")

    return {
        "raw_matmul_gflops": average_matmul_gflops,
        "estimated_llm_gflops": llm_gflops,
        "efficiency_ratio": llm_gflops / average_matmul_gflops,
    }


def benchmark_device(workload="inference"):
    """Complete device benchmarking for LLM inference estimation."""
    print(f"Workload: {workload}")
    print("=" * 50)

    if workload == "inference":
        # Test different matrix sizes
        matmul_results = measure_matmul_gflops(
            sizes=[512, 1024, 2048, 4096], warmup=3, runs=5
        )
    else:
        matmul_results = measure_matmul_gflops(sizes=[2000], warmup=0, runs=1)

    print("\nMatrix Multiplication Results:")
    for size, result in matmul_results.items():
        print(f"  {size}×{size}: {result['gflops']:.2f} GFLOPS")

    print("\nEstimating LLM Inference Performance:")
    print("-" * 30)
    llm_estimate = estimate_llm_inference_gflops(matmul_results)

    if llm_estimate:
        print(f"Raw MatMul Average: {llm_estimate['raw_matmul_gflops']:.2f} GFLOPS")
        print(
            f"Estimated LLM Performance: {llm_estimate['estimated_llm_gflops']:.2f} GFLOPS"
        )
        print(f"LLM Efficiency Ratio: {llm_estimate['efficiency_ratio']:.2f}")

        return llm_estimate["estimated_llm_gflops"]

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark device performance for LLM inference"
    )
    parser.add_argument(
        "--workload",
        choices=["matmul", "inference"],
        default="inference",
        help="Choose workload type: "
        "'matmul' for raw matmul, 'inference' for LLM-adjusted performance",
    )
    args = parser.parse_args()

    # Benchmark this device
    effective_gflops = benchmark_device(args.workload)
    if effective_gflops:
        print("\nUse this value in your Device config:")
        print(f"gflops={effective_gflops:.1f}")
