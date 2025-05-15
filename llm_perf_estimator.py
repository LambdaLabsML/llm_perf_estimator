import argparse

# GPU memory bandwidth in bytes per second
GPU_BANDWIDTH = {
    "H100": 3.35e12,  # 3.35 TB/s
    "H200": 4.8e12,   # 4.8 TB/s
    "B200": 8.0e12    # 8.0 TB/s
}

# GPU peak compute performance in FLOPS (approximate for FP16)
GPU_PEAK_FLOPS = {
    "H100": 2e15,  # 2 PFLOPS
    "H200": 3e15,  # 3 PFLOPS
    "B200": 5e15   # 5 PFLOPS
}

# Bytes per parameter for each precision
PRECISION_BYTES = {
    "FP16": 2.0,
    "FP8": 1.0,
    "FP4": 0.5
}

# Approximate FLOPS per parameter per token for transformer forward pass
FLOPS_PER_PARAM = 6.0

def estimate_performance(model_size_b, precision, gpu_type,
                         input_tokens, output_tokens, qps,
                         batch_size, cache_ratio, mfu, num_gpus):
    """
    Estimate LLM inference performance with compute-bound prefill 
    (using FLOPs utilization) and memory-bound decode metrics,
    scaling by number of GPUs.
    """
    # Convert model size to bytes and count params
    param_bytes = PRECISION_BYTES[precision]
    model_bytes = model_size_b * 1e9 * param_bytes
    model_params = model_size_b * 1e9  # total parameters
    
    # KV cache total size per sequence (bytes)
    cache_bytes = cache_ratio * model_bytes
    
    # GPU specs scaled by number of GPUs
    mem_bw_total = GPU_BANDWIDTH[gpu_type] * num_gpus
    peak_flops_total = GPU_PEAK_FLOPS[gpu_type] * num_gpus
    
    # Compute-bound prefill throughput (tokens/sec) scaled by MFU (FLOPs utilization)
    flops_per_token = FLOPS_PER_PARAM * model_params
    effective_flops = peak_flops_total * mfu
    tokens_per_sec_prefill = effective_flops / flops_per_token
    
    # Prefill time per batch (sec)
    time_prefill_s = batch_size * input_tokens / tokens_per_sec_prefill
    
    # Memory-bound decode throughput (tokens/sec) across GPUs
    traffic_step_decode = model_bytes + batch_size * cache_bytes
    tokens_per_sec_decode = batch_size * mem_bw_total / traffic_step_decode
    
    # Decode time per batch (sec)
    time_decode_s = batch_size * output_tokens / tokens_per_sec_decode
    
    # Total time per batch (sec)
    latency_per_query_s = time_prefill_s + time_decode_s
    
    
    return {
        "tokens_per_sec_prefill": tokens_per_sec_prefill,
        "tokens_per_sec_decode": tokens_per_sec_decode,
        "time_prefill_s": time_prefill_s,
        "time_decode_s": time_decode_s,
        "latency_per_query_s": latency_per_query_s,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Estimate LLM inference performance with MFU as FLOPs utilization and multi-GPU support."
    )
    parser.add_argument("--model_size_b", type=float, required=True,
                        help="Model size in billions of parameters.")
    parser.add_argument("--precision", choices=PRECISION_BYTES.keys(), required=True,
                        help="Precision: FP16, FP8, or FP4.")
    parser.add_argument("--gpu_type", choices=GPU_BANDWIDTH.keys(), required=True,
                        help="GPU type: H100, H200, or B200.")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs in the setup.")
    parser.add_argument("--input_tokens", type=int, required=True,
                        help="Number of input (prompt) tokens.")
    parser.add_argument("--output_tokens", type=int, required=True,
                        help="Number of output tokens.")
    parser.add_argument("--qps", type=float, default=1.0,
                        help="Desired queries per second (QPS).")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (# of concurrent sequences).")
    parser.add_argument("--cache_ratio", type=float, default=0.05,
                        help="KV-cache total size as ratio of model size.")
    parser.add_argument("--mfu", type=float, default=0.15,
                        help="Model FLOPs utilization fraction for prefill stage (e.g., 0.15).")

    args = parser.parse_args()
    res = estimate_performance(
        args.model_size_b, args.precision, args.gpu_type,
        args.input_tokens, args.output_tokens, args.qps,
        args.batch_size, args.cache_ratio, args.mfu,
        args.num_gpus
    )
    
    print(f"Prefill throughput (compute-bound, {args.mfu*100:.0f}% FLOPs util) across {args.num_gpus} GPUs: {res['tokens_per_sec_prefill']:.2f} tokens/sec")
    print(f"Decode throughput (memory-bound) across {args.num_gpus} GPUs: {res['tokens_per_sec_decode']:.2f} tokens/sec")
    print(f"Prefill time:                      {res['time_prefill_s']:.3f} sec")
    print(f"Decode time:                       {res['time_decode_s']:.3f} sec")
    print(f"Latency per query:                 {res['latency_per_query_s']:.3f} sec")

if __name__ == "__main__":
    main()
