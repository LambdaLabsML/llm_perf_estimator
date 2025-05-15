#!/usr/bin/env python3
import argparse
import subprocess
import sys
import re
from pathlib import Path

import pandas as pd

def df_to_markdown(df, index_name="GPU"):
    """
    Convert a DataFrame with index to a Markdown table without external dependencies.
    """
    # Prepare header
    columns = [index_name] + list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    
    # Prepare rows
    rows = []
    for idx, row in df.iterrows():
        cells = [str(idx)] + [str(row[col]) for col in df.columns]
        rows.append("| " + " | ".join(cells) + " |")
    
    return "\n".join([header, separator] + rows)

def parse_output(output: str):
    """
    Parse llm_perf_estimator.py stdout for metrics.
    Returns a dict with keys: 'prefill', 'decode', 'aggregate_decode', 'latency'.
    """
    metrics = {}
    for line in output.splitlines():
        if line.startswith("Prefill throughput"):
            m = re.search(r":\s*([\d\.]+)\s*tokens/sec", line)
            if m: metrics["prefill"] = float(m.group(1))
        elif line.startswith("Decode throughput"):
            m = re.search(r":\s*([\d\.]+)\s*tokens/sec", line)
            if m: metrics["decode"] = float(m.group(1))
        elif line.startswith("Latency per query"):
            m = re.search(r":\s*([\d\.]+)\s*sec", line)
            if m: metrics["latency"] = float(m.group(1))
    return metrics

def main():
    parser = argparse.ArgumentParser(
        description="Grid search over GPUs and model sizes using llm_perf_estimator.py"
    )
    parser.add_argument("--precision", default="FP8",
                        choices=["FP16","FP8","FP4"],
                        help="Precision for inference.")
    parser.add_argument("--input_tokens", type=int, default=512,
                        help="Number of input prompt tokens.")
    parser.add_argument("--output_tokens", type=int, default=512,
                        help="Number of output tokens to generate.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (concurrent sequences).")
    parser.add_argument("--cache_ratio", type=float, default=0.05,
                        help="KV-cache size as ratio of model size.")
    parser.add_argument("--mfu", type=float, default=0.15,
                        help="Compute FLOPs utilization (e.g. 0.15 for 15%).")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to scale bandwidth & FLOPs.")
    parser.add_argument("--output", default="llm_perf_grid.md",
                        help="Markdown output filename.")
    args = parser.parse_args()

    # Grid definitions
    gpu_types = ["H100", "H200", "B200"]
    model_sizes = [8, 13, 32, 70, 405, 671]

    # Prepare empty dataframes
    df_prefill = pd.DataFrame(index=gpu_types, columns=[f"{s}B" for s in model_sizes])
    df_decode = df_prefill.copy()
    df_agg = df_prefill.copy()
    df_latency = df_prefill.copy()

    # Loop and collect metrics
    for gpu in gpu_types:
        for size in model_sizes:
            cmd = [
                sys.executable, "llm_perf_estimator.py",
                "--model_size_b", str(size),
                "--precision", args.precision,
                "--gpu_type", gpu,
                "--num_gpus", str(args.num_gpus),
                "--input_tokens", str(args.input_tokens),
                "--output_tokens", str(args.output_tokens),
                "--batch_size", str(args.batch_size),
                "--cache_ratio", str(args.cache_ratio),
                "--mfu", str(args.mfu)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            metrics = parse_output(result.stdout)

            df_prefill.at[gpu, f"{size}B"] = metrics.get("prefill", float("nan"))
            df_decode.at[gpu,  f"{size}B"] = metrics.get("decode", float("nan"))
            df_latency.at[gpu, f"{size}B"] = metrics.get("latency", float("nan"))

    # Build markdown content
    md_lines = []
    # Command snippet
    cmd_args = [
        f"--precision {args.precision}",
        f"--input_tokens {args.input_tokens}",
        f"--output_tokens {args.output_tokens}",
        f"--batch_size {args.batch_size}",
        f"--cache_ratio {args.cache_ratio}",
        f"--mfu {args.mfu}",
        f"--num_gpus {args.num_gpus}",
        f"--output {args.output}"
    ]
    md_lines.append("```bash")
    md_lines.append(f"python {Path(__file__).name} " + " ".join(cmd_args))
    md_lines.append("```")
    md_lines.append("")

    # Tables
    md_lines.append("## Prefill Throughput (tokens/sec)")
    md_lines.append(df_to_markdown(df_prefill))
    md_lines.append("")

    md_lines.append("## Decode Throughput (tokens/sec)")
    md_lines.append(df_to_markdown(df_decode))
    md_lines.append("")
    
    md_lines.append("## Latency per Query (sec)")
    md_lines.append(df_to_markdown(df_latency))
    md_lines.append("")

    # Write to file
    with open(args.output, "w") as f:
        f.write("\n".join(md_lines))

    print(f"Markdown results saved to {args.output}")

if __name__ == "__main__":
    main()

