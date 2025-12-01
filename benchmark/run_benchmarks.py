#!/usr/bin/env python3
"""
OMP Benchmark Runner and Plotter

Compiles and runs all implementations across different N values and distributions,
then generates plots comparing performance.

Usage:
    python benchmark/run_benchmarks.py [--skip-compile] [--skip-run] [--plot-only]
"""

import os
import sys
import subprocess
import argparse
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import re
import threading
import queue

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plotting will be skipped.")

# Benchmark configuration
N_VALUES = [100, 1000, 10000, 100000, 500000, 1000000, 2000000, 4000000, 8000000, 16000000, 32000000, 64000000, 128000000, 256000000, 512000000, 1024000000]
K_VALUES = [50, 100, 500]
DISTRIBUTIONS = ["uniform", "zipf", "heavy_elephant", "heavy_mice", "bimodal"]

# Implementation configurations
import platform

# Detect platform for OpenMP flags
IS_MACOS = platform.system() == "Darwin"

# OpenMP flags differ by platform
if IS_MACOS:
    # macOS with Homebrew libomp: brew install libomp
    OMP_FLAGS = "-Xpreprocessor -fopenmp"
    OMP_LIBS = "-lomp"
else:
    # Linux with GCC
    OMP_FLAGS = "-fopenmp"
    OMP_LIBS = ""

IMPLEMENTATIONS = {
    "cpu": {
        "compiler": "gcc",
        "flags": "-O3 -march=native -std=c11 -mcmodel=medium",
        "sources": "benchmark/benchmark.c CPU/switch.c CPU/server.c",
        "libs": "-lm",
        "defines": "",
    },
    "cpu_omp": {
        "compiler": "gcc",
        "flags": f"-mcmodel=medium",
        "sources": "benchmark/benchmark.c CPU/switch.c CPU/server.c",
        "libs": f"-lm {OMP_LIBS}",
        "defines": "-DUSE_OPENMP",
    },
    "gpu": {
        "compiler": "nvcc",
        "flags": "-O3 --use_fast_math -std=c++14 -Xcompiler -mcmodel=medium",
        "arch": "",
        "sources": "benchmark/benchmark.c CPU/server.c CPU/switch.c GPU/server_gpu.cu",
        "libs": "",
        "defines": "-DUSE_CUDA",
    },
    "gpu_opt": {
        "compiler": "nvcc",
        "flags": "-O3 --use_fast_math -std=c++14 -Xcompiler -mcmodel=medium",
        "arch": "",  # Let nvcc auto-detect GPU architecture
        "sources": "benchmark/benchmark.c CPU/server.c CPU/switch.c GPU_Optimized/server_gpu_opt.cu",
        "libs": "-lcublas",  # cuBLAS for parallel triangular solves
        "defines": "-DUSE_CUDA",
    },
}

@dataclass
class BenchmarkResult:
    distribution: str
    n: int
    k: int
    impl: str
    time_ms: float
    stddev_ms: float
    throughput: float

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent

def compile_implementation(impl_name: str, n: int, project_root: Path) -> Tuple[bool, str]:
    """Compile a single implementation with given N value."""
    impl = IMPLEMENTATIONS[impl_name]
    build_dir = project_root / "build"
    build_dir.mkdir(exist_ok=True)
    
    output = build_dir / f"{impl_name}-bench-n{n}"
    
    if impl["compiler"] == "gcc":
        cmd = (
            f"{impl['compiler']} {impl['flags']} {impl['defines']} -DN={n} "
            f"{impl['sources']} -o {output} {impl['libs']}"
        )
    else:  # nvcc
        cmd = (
            f"{impl['compiler']} {impl['flags']} {impl['defines']} -DN={n} "
            f"{impl.get('arch', '')} {impl['sources']} -o {output} {impl['libs']}"
        )
    
    print(f"  Compilation command: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=project_root,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"  Compilation stderr: {result.stderr}")
            return False, f"Compilation failed: {result.stderr}"
        if result.stdout:
            print(f"  Compilation stdout: {result.stdout}")
        print(f"  ✓ Compilation successful: {output}")
        return True, str(output)
    except Exception as e:
        print(f"  Exception during compilation: {e}")
        return False, str(e)

def run_benchmark(executable: str, impl_name: str, k: int, dist: str = "all") -> List[BenchmarkResult]:
    """Run a single benchmark and parse results."""
    cmd = f"{executable} --k {k} --impl {impl_name}"
    if dist != "all":
        cmd += f" --dist {dist}"
    
    print(f"    Executing: {cmd}")
    print()  # Add blank line before streaming output
    
    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
        )
        
        # Queues to collect output
        stdout_queue = queue.Queue()
        stderr_queue = queue.Queue()
        
        def read_stdout():
            for line in iter(process.stdout.readline, ''):
                stdout_queue.put(line.rstrip())
            process.stdout.close()
        
        def read_stderr():
            for line in iter(process.stderr.readline, ''):
                stderr_queue.put(line.rstrip())
            process.stderr.close()
        
        # Start threads to read output
        stdout_thread = threading.Thread(target=read_stdout)
        stderr_thread = threading.Thread(target=read_stderr)
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        # Collect and print output in real-time
        stdout_lines = []
        stderr_lines = []
        
        while process.poll() is None or not stdout_queue.empty() or not stderr_queue.empty():
            # Process stderr (benchmark progress)
            while not stderr_queue.empty():
                try:
                    line = stderr_queue.get_nowait()
                    if line:
                        stderr_lines.append(line)
                        print(f"    {line}", flush=True)
                except queue.Empty:
                    break
            
            # Process stdout (RESULT lines and other output)
            while not stdout_queue.empty():
                try:
                    line = stdout_queue.get_nowait()
                    if line:
                        stdout_lines.append(line)
                        if not line.startswith("RESULT,"):
                            print(f"    {line}", flush=True)
                except queue.Empty:
                    break
        
        # Wait for threads to finish
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        
        # Wait for process to complete
        return_code = process.wait(timeout=3600)
        
        # Parse RESULT lines
        results = []
        for line in stdout_lines:
            if line.startswith("RESULT,"):
                parts = line.split(",")
                if len(parts) >= 8:
                    results.append(BenchmarkResult(
                        distribution=parts[1],
                        n=int(parts[2]),
                        k=int(parts[3]),
                        impl=parts[4],
                        time_ms=float(parts[5]),
                        stddev_ms=float(parts[6]),
                        throughput=float(parts[7])
                    ))
                    print(f"    ✓ {parts[1]}: {parts[5]}ms (±{parts[6]}ms), {float(parts[7])/1e6:.2f}M flows/sec", flush=True)
        
        print()  # Add blank line after output
        return results
    except subprocess.TimeoutExpired:
        print(f"    ✗ Timeout running {executable}")
        return []
    except Exception as e:
        print(f"    ✗ Error running {executable}: {e}")
        return []

def save_results(results: List[BenchmarkResult], output_file: Path):
    """Save results to CSV file."""
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["distribution", "n", "k", "impl", "time_ms", "stddev_ms", "throughput"])
        for r in results:
            writer.writerow([r.distribution, r.n, r.k, r.impl, r.time_ms, r.stddev_ms, r.throughput])

def load_results(input_file: Path) -> List[BenchmarkResult]:
    """Load results from CSV file."""
    results = []
    with open(input_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(BenchmarkResult(
                distribution=row["distribution"],
                n=int(row["n"]),
                k=int(row["k"]),
                impl=row["impl"],
                time_ms=float(row["time_ms"]),
                stddev_ms=float(row["stddev_ms"]),
                throughput=float(row["throughput"])
            ))
    return results

def plot_time_vs_n(results: List[BenchmarkResult], output_dir: Path, k: int = 100):
    """Plot execution time vs N for each distribution."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = {"cpu": "#2ecc71", "cpu_omp": "#3498db", "gpu": "#e74c3c", "gpu_opt": "#9b59b6"}
    markers = {"cpu": "o", "cpu_omp": "s", "gpu": "^", "gpu_opt": "D"}
    
    for idx, dist in enumerate(DISTRIBUTIONS):
        ax = axes[idx]
        
        for impl in IMPLEMENTATIONS.keys():
            data = [r for r in results if r.distribution == dist and r.impl == impl and r.k == k]
            if not data:
                continue
            
            data.sort(key=lambda x: x.n)
            ns = [r.n for r in data]
            times = [r.time_ms for r in data]
            stds = [r.stddev_ms for r in data]
            
            ax.errorbar(ns, times, yerr=stds, label=impl, 
                       color=colors.get(impl, "gray"),
                       marker=markers.get(impl, "o"),
                       capsize=3, linewidth=2, markersize=6)
        
        ax.set_xlabel("Number of Flows (N)")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"Distribution: {dist}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplot
    axes[-1].axis("off")
    
    plt.suptitle(f"Execution Time vs N (K={k})", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "time_vs_n.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_throughput_bars(results: List[BenchmarkResult], output_dir: Path, n: int = 100000, k: int = 100):
    """Plot throughput comparison as grouped bar chart."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    impls = list(IMPLEMENTATIONS.keys())
    x = np.arange(len(DISTRIBUTIONS))
    width = 0.2
    
    colors = {"cpu": "#2ecc71", "cpu_omp": "#3498db", "gpu": "#e74c3c", "gpu_opt": "#9b59b6"}
    
    for i, impl in enumerate(impls):
        throughputs = []
        for dist in DISTRIBUTIONS:
            data = [r for r in results if r.distribution == dist and r.impl == impl 
                   and r.n == n and r.k == k]
            if data:
                throughputs.append(data[0].throughput / 1e6)  # Convert to millions
            else:
                throughputs.append(0)
        
        ax.bar(x + i * width, throughputs, width, label=impl, color=colors.get(impl, "gray"))
    
    ax.set_xlabel("Distribution")
    ax.set_ylabel("Throughput (M flows/sec)")
    ax.set_title(f"Throughput Comparison (N={n:,}, K={k})")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(DISTRIBUTIONS, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_dir / "throughput_bars.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_speedup_heatmap(results: List[BenchmarkResult], output_dir: Path, k: int = 100):
    """Plot speedup heatmap (GPU-opt vs CPU)."""
    if not HAS_MATPLOTLIB:
        return
    
    # Get unique N values that have data
    available_ns = sorted(set(r.n for r in results))
    
    speedup_data = np.zeros((len(DISTRIBUTIONS), len(available_ns)))
    
    for i, dist in enumerate(DISTRIBUTIONS):
        for j, n in enumerate(available_ns):
            cpu_data = [r for r in results if r.distribution == dist and r.impl == "cpu" 
                       and r.n == n and r.k == k]
            gpu_opt_data = [r for r in results if r.distribution == dist and r.impl == "gpu" 
                          and r.n == n and r.k == k]
            
            if cpu_data and gpu_opt_data:
                speedup_data[i, j] = cpu_data[0].time_ms / gpu_opt_data[0].time_ms
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(speedup_data, cmap="RdYlGn", aspect="auto")
    
    ax.set_xticks(np.arange(len(available_ns)))
    ax.set_yticks(np.arange(len(DISTRIBUTIONS)))
    ax.set_xticklabels([f"{n:,}" for n in available_ns], rotation=45, ha="right")
    ax.set_yticklabels(DISTRIBUTIONS)
    
    # Add text annotations
    for i in range(len(DISTRIBUTIONS)):
        for j in range(len(available_ns)):
            val = speedup_data[i, j]
            if val > 0:
                text = ax.text(j, i, f"{val:.1f}x", ha="center", va="center",
                              color="white" if val > 5 else "black")
    
    ax.set_xlabel("Number of Flows (N)")
    ax.set_ylabel("Distribution")
    ax.set_title(f"GPU vs CPU (K={k})")
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Speedup (x)")
    
    plt.tight_layout()
    plt.savefig(output_dir / "speedup_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_stress_test(results: List[BenchmarkResult], output_dir: Path, k: int = 100):
    """Plot results for high N values (stress test)."""
    if not HAS_MATPLOTLIB:
        return
    
    stress_ns = [n for n in N_VALUES if n >= 500000]
    stress_results = [r for r in results if r.n in stress_ns and r.k == k]
    
    if not stress_results:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {"cpu": "#2ecc71", "cpu_omp": "#3498db", "gpu": "#e74c3c", "gpu_opt": "#9b59b6"}
    
    # Time plot
    for impl in IMPLEMENTATIONS.keys():
        data = [r for r in stress_results if r.impl == impl and r.distribution == "zipf"]
        if not data:
            continue
        data.sort(key=lambda x: x.n)
        ns = [r.n for r in data]
        times = [r.time_ms for r in data]
        ax1.plot(ns, times, "o-", label=impl, color=colors.get(impl, "gray"), linewidth=2, markersize=8)
    
    ax1.set_xlabel("Number of Flows (N)")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Stress Test: Time (Zipf Distribution)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
    
    # Throughput plot
    for impl in IMPLEMENTATIONS.keys():
        data = [r for r in stress_results if r.impl == impl and r.distribution == "zipf"]
        if not data:
            continue
        data.sort(key=lambda x: x.n)
        ns = [r.n for r in data]
        tps = [r.throughput / 1e6 for r in data]  # Convert to millions
        ax2.plot(ns, tps, "o-", label=impl, color=colors.get(impl, "gray"), linewidth=2, markersize=8)
    
    ax2.set_xlabel("Number of Flows (N)")
    ax2.set_ylabel("Throughput (M flows/sec)")
    ax2.set_title("Stress Test: Throughput (Zipf Distribution)")
    ax2.set_xscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
    
    plt.tight_layout()
    plt.savefig(output_dir / "stress_test.png", dpi=150, bbox_inches="tight")
    plt.close()

def generate_report(results: List[BenchmarkResult], output_dir: Path):
    """Generate markdown report with summary statistics."""
    report_lines = [
        "# OMP Benchmark Report",
        "",
        "## Configuration",
        f"- N values: {', '.join(f'{n:,}' for n in N_VALUES)}",
        f"- K values: {', '.join(str(k) for k in K_VALUES)}",
        f"- Distributions: {', '.join(DISTRIBUTIONS)}",
        f"- Implementations: {', '.join(IMPLEMENTATIONS.keys())}",
        "",
        "## Summary",
        "",
    ]
    
    # Find best speedups
    k = 100
    for n in [10000, 100000, 1000000]:
        cpu_zipf = [r for r in results if r.distribution == "zipf" and r.impl == "cpu" 
                   and r.n == n and r.k == k]
        gpu_opt_zipf = [r for r in results if r.distribution == "zipf" and r.impl == "gpu_opt" 
                       and r.n == n and r.k == k]
        
        if cpu_zipf and gpu_opt_zipf:
            speedup = cpu_zipf[0].time_ms / gpu_opt_zipf[0].time_ms
            report_lines.append(f"- GPU-Opt speedup at N={n:,} (Zipf): **{speedup:.1f}x**")
    
    report_lines.extend([
        "",
        "## Plots",
        "",
        "- [Time vs N](time_vs_n.png)",
        "- [Throughput Comparison](throughput_bars.png)",
        "- [Speedup Heatmap](speedup_heatmap.png)",
        "- [Stress Test Results](stress_test.png)",
        "- [Distribution Shapes](distributions.png)",
        "",
        "## Raw Data",
        "",
        "See [raw_data.csv](raw_data.csv) for all benchmark results.",
    ])
    
    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(report_lines))

def main():
    parser = argparse.ArgumentParser(description="Run OMP benchmarks")
    parser.add_argument("--skip-compile", action="store_true", help="Skip compilation")
    parser.add_argument("--skip-run", action="store_true", help="Skip running benchmarks")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots from existing data")
    parser.add_argument("--impls", type=str, default="all", help="Comma-separated list of implementations to run")
    parser.add_argument("--ns", type=str, default="all", help="Comma-separated list of N values to test")
    parser.add_argument("--ks", type=str, default="100", help="Comma-separated list of K values to test")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  OMP BENCHMARK RUNNER")
    print("="*70)
    
    project_root = get_project_root()
    output_dir = project_root / "benchmark" / "results"
    output_dir.mkdir(exist_ok=True)
    
    print(f"📁 Project root: {project_root}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🖥️  Platform: {'macOS' if IS_MACOS else 'Linux'}")
    
    # Parse implementation list
    if args.impls == "all":
        impls_to_run = list(IMPLEMENTATIONS.keys())
    else:
        impls_to_run = [x.strip() for x in args.impls.split(",")]
    
    # Parse N values
    if args.ns == "all":
        ns_to_run = N_VALUES
    else:
        ns_to_run = [int(x.strip()) for x in args.ns.split(",")]
    
    # Parse K values
    ks_to_run = [int(x.strip()) for x in args.ks.split(",")]
    
    all_results = []
    
    if args.plot_only:
        # Load existing results
        print("=" * 70)
        print("PLOT-ONLY MODE: Loading existing results")
        print("=" * 70)
        csv_file = output_dir / "raw_data.csv"
        if csv_file.exists():
            all_results = load_results(csv_file)
            print(f"✓ Loaded {len(all_results)} results from {csv_file}")
        else:
            print(f"✗ Error: {csv_file} not found")
            return 1
    else:
        print("=" * 70)
        print("BENCHMARK EXECUTION")
        print("=" * 70)
        print(f"Implementations to test: {', '.join(impls_to_run)}")
        print(f"N values: {', '.join(f'{n:,}' for n in ns_to_run)}")
        print(f"K values: {', '.join(str(k) for k in ks_to_run)}")
        print(f"Distributions: {', '.join(DISTRIBUTIONS)}")
        print("=" * 70)
        
        # Compile and run benchmarks
        total_tasks = len(impls_to_run) * len(ns_to_run) * len(ks_to_run)
        current_task = 0
        
        for impl in impls_to_run:
            if impl not in IMPLEMENTATIONS:
                print(f"⚠ Warning: Unknown implementation '{impl}', skipping")
                continue
            
            # Check if compiler is available
            compiler = IMPLEMENTATIONS[impl]["compiler"]
            print(f"\n▶ Checking {compiler} availability...")
            try:
                version_result = subprocess.run([compiler, "--version"], capture_output=True, check=True, text=True)
                print(f"  ✓ {compiler} found: {version_result.stdout.split()[0] if version_result.stdout else 'version unknown'}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"  ✗ {compiler} not found, skipping {impl}")
                continue
            
            for n in ns_to_run:
                print(f"\n{'='*70}")
                print(f"▶ Processing {impl.upper()} with N={n:,}")
                print(f"{'='*70}")
                
                if not args.skip_compile:
                    print(f"📦 Compiling {impl} for N={n:,}...")
                    success, result = compile_implementation(impl, n, project_root)
                    if not success:
                        print(f"  ✗ Compilation failed: {result}")
                        continue
                    executable = result
                else:
                    executable = str(project_root / "build" / f"{impl}-bench-n{n}")
                    print(f"⏭  Skipping compilation, using existing executable: {executable}")
                
                if not args.skip_run:
                    for k in ks_to_run:
                        current_task += 1
                        print(f"\n  🚀 Running benchmarks with K={k} [{current_task}/{total_tasks}]...")
                        results = run_benchmark(executable, impl, k)
                        all_results.extend(results)
                        print(f"  ✓ Collected {len(results)} results")
                else:
                    print(f"  ⏭  Skipping benchmark runs")
        
        # Save results
        print(f"\n{'='*70}")
        if all_results:
            output_file = output_dir / "raw_data.csv"
            save_results(all_results, output_file)
            print(f"💾 Saved {len(all_results)} total results to {output_file}")
        else:
            print("⚠ No results to save")
        print(f"{'='*70}")
    
    # Generate plots
    if all_results and HAS_MATPLOTLIB:
        print(f"\n{'='*70}")
        print("📊 GENERATING PLOTS")
        print(f"{'='*70}")
        
        print("  📈 Generating time vs N plot...")
        plot_time_vs_n(all_results, output_dir)
        print(f"    ✓ Saved: {output_dir / 'time_vs_n.png'}")
        
        print("  📊 Generating throughput bar chart...")
        plot_throughput_bars(all_results, output_dir)
        print(f"    ✓ Saved: {output_dir / 'throughput_bars.png'}")
        
        print("  🔥 Generating speedup heatmap...")
        plot_speedup_heatmap(all_results, output_dir)
        print(f"    ✓ Saved: {output_dir / 'speedup_heatmap.png'}")
        
        print("  💪 Generating stress test plot...")
        plot_stress_test(all_results, output_dir)
        print(f"    ✓ Saved: {output_dir / 'stress_test.png'}")
        
        print("  📝 Generating report...")
        generate_report(all_results, output_dir)
        print(f"    ✓ Saved: {output_dir / 'report.md'}")
        
        print(f"\n✅ All plots saved to {output_dir}")
        print(f"{'='*70}")
    elif all_results and not HAS_MATPLOTLIB:
        print("\n⚠ matplotlib not installed, skipping plot generation")
    elif not all_results:
        print("\n⚠ No results available for plotting")
    
    print("\n🎉 Benchmark run complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

