# LLM Performance Estimator

This project provides tools to estimate and benchmark the theoretical inference performance of large language models (LLMs) on high-end NVIDIA GPUs. It consists of two scripts:

* **`llm_perf_estimator.py`**: Estimates per-query and per-token metrics for a single LLM configuration.
* **`grid_search_runner.py`**: Automates grid search across multiple GPUs and model sizes, producing comprehensive Markdown reports.

---

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)

   * [Running `llm_perf_estimator.py`](#running-llm_perf_estimatorpy)
   * [Running `grid_search_runner.py`](#running-grid_search_runnerpy)
3. [How `llm_perf_estimator.py` Works](#how-llm_perf_estimatorpy-works)

   * [Inference Phases](#inference-phases)
   * [Metrics of Interest](#metrics-of-interest)
   * [Script Parameters](#script-parameters)
   * [Hardcoded Heuristics](#hardcoded-heuristics)
   * [Performance Formulas](#performance-formulas)

---

## Installation

1. Ensure you have **Python 3.8+** installed.
2. Install required dependencies:

   ```bash
   pip install pandas
   ```
3. Place both scripts (`llm_perf_estimator.py` and `grid_search_runner.py`) in the same directory.

---

## Usage

### Running `llm_perf_estimator.py`

Estimate performance for a single model/GPU configuration:

```bash
python llm_perf_estimator.py \
  --model_size_b 70 \
  --precision FP8 \
  --gpu_type H200 \
  --num_gpus 1 \
  --input_tokens 512 \
  --output_tokens 512 \
  --batch_size 8 \
  --cache_ratio 0.05 \
  --mfu 0.15
```

This prints:

* **Prefill throughput** (tokens/sec, compute-bound with FLOPs utilization)
* **Decode throughput** (tokens/sec, memory-bound)
* **Prefill & Decode times** (sec)
* **Latency per query** (sec)
* 
### Running `grid_search_runner.py`

Perform a grid search over GPU types and model sizes, saving results to Markdown:

```bash
python grid_search_runner.py \
  --precision FP8 \
  --input_tokens 512 \
  --output_tokens 512 \
  --batch_size 1 \
  --cache_ratio 0.05 \
  --mfu 0.15 \
  --num_gpus 1 \
  --output llm_perf_grid.md
```

The generated `llm_perf_grid.md` will include:

1. The command used (in a code block)
2. Four tables for:

   * Prefill Throughput
   * Decode Throughput
   * Aggregate Decode Throughput
   * Latency per Query

---

## How `llm_perf_estimator.py` Works

### Inference Phases

The script models two distinct phases of autoregressive LLM inference:

1. **Prefill (prompt encoding):**  Processes the entire input prompt in one batched forward pass. This phase is **compute-bound**, since the GPU performs large matrix‐matrix operations (high arithmetic intensity).
2. **Decode (token generation):**  Generates new tokens one at a time (or in micro‐batches). This phase is **memory-bound**, driven by streaming weights and accessing the growing KV‐cache from GPU memory.

### Metrics of Interest

* **Prefill throughput** <br> Tokens/sec during prompt encoding.
* **Decode throughput** <br> Tokens/sec during generation.
* **Latency per query** <br> Seconds to process one query (input + output).

### Script Parameters

| Argument          | Description                                                     |
| ----------------- | --------------------------------------------------------------- |
| `--model_size_b`  | Model size in billions of parameters (e.g., `70` for 70B).      |
| `--precision`     | Numeric precision: `FP16`, `FP8`, or `FP4`.                     |
| `--gpu_type`      | GPU hardware: `H100`, `H200`, or `B200`.                        |
| `--num_gpus`      | Number of GPUs to scale bandwidth & compute.                    |
| `--input_tokens`  | Number of prompt tokens per query.                              |
| `--output_tokens` | Number of tokens to generate per query.                         |
| `--batch_size`    | Concurrent sequences in one batch.                              |
| `--cache_ratio`   | KV-cache total size as a fraction of model size (e.g., `0.05`). |
| `--mfu`           | FLOPs utilization fraction for prefill (e.g., `0.15` for 15%).  |

### Hardcoded Heuristics

* **`FLOPS_PER_PARAM = 6.0`** <br> Approximate FLOPs per parameter per token for a forward pass.
* **`GPU_BANDWIDTH`** <br> H100: 3.35 TB/s, H200: 4.8 TB/s, B200: 8 TB/s.
* **`GPU_PEAK_FLOPS`** <br> H100: \~2 PFLOPS (FP16), H200: \~3 PFLOPS, B200: \~5 PFLOPS.
* **`PRECISION_BYTES`** <br> FP16: 2 B/param, FP8: 1 B/param, FP4: 0.5 B/param.

### Performance 

#### 1xGPU bs=16
```bash
python grid_search_runner.py --precision FP8 --input_tokens 512 --output_tokens 512 --batch_size 16 --cache_ratio 0.05 --mfu 0.15 --num_gpus 1 --output llm_perf_grid_bs16_1x.md
```

#### Prefill Throughput (tokens/sec)
| GPU | 8B | 13B | 32B | 70B | 405B | 671B |
| --- | --- | --- | --- | --- | --- | --- |
| H100 | 50000.0 | 30769.23 | 12500.0 | 5714.29 | 987.65 | 596.13 |
| H200 | 75000.0 | 46153.85 | 18750.0 | 8571.43 | 1481.48 | 894.19 |
| B200 | 125000.0 | 76923.08 | 31250.0 | 14285.71 | 2469.14 | 1490.31 |

#### Decode Throughput (tokens/sec)
| GPU | 8B | 13B | 32B | 70B | 405B | 671B |
| --- | --- | --- | --- | --- | --- | --- |
| H100 | 29777.78 | 18324.79 | 7444.44 | 3403.17 | 588.2 | 355.03 |
| H200 | 42666.67 | 26256.41 | 10666.67 | 4876.19 | 842.8 | 508.69 |
| B200 | 71111.11 | 43760.68 | 17777.78 | 8126.98 | 1404.66 | 847.82 |

#### Latency per Query (sec)
| GPU | 8B | 13B | 32B | 70B | 405B | 671B |
| --- | --- | --- | --- | --- | --- | --- |
| H100 | 0.439 | 0.713 | 1.756 | 3.841 | 22.222 | 36.816 |
| H200 | 0.301 | 0.489 | 1.205 | 2.636 | 15.25 | 25.265 |
| B200 | 0.181 | 0.294 | 0.723 | 1.581 | 9.15 | 15.159 |


#### 8xGPU bs=16
```bash
python grid_search_runner.py --precision FP8 --input_tokens 512 --output_tokens 512 --batch_size 16 --cache_ratio 0.05 --mfu 0.15 --num_gpus 8 --output llm_perf_grid_bs16_8x.md
```

#### Prefill Throughput (tokens/sec)
| GPU | 8B | 13B | 32B | 70B | 405B | 671B |
| --- | --- | --- | --- | --- | --- | --- |
| H100 | 50000.0 | 30769.23 | 12500.0 | 5714.29 | 987.65 | 596.13 |
| H200 | 75000.0 | 46153.85 | 18750.0 | 8571.43 | 1481.48 | 894.19 |
| B200 | 125000.0 | 76923.08 | 31250.0 | 14285.71 | 2469.14 | 1490.31 |

#### Decode Throughput (tokens/sec)
| GPU | 8B | 13B | 32B | 70B | 405B | 671B |
| --- | --- | --- | --- | --- | --- | --- |
| H100 | 29777.78 | 18324.79 | 7444.44 | 3403.17 | 588.2 | 355.03 |
| H200 | 42666.67 | 26256.41 | 10666.67 | 4876.19 | 842.8 | 508.69 |
| B200 | 71111.11 | 43760.68 | 17777.78 | 8126.98 | 1404.66 | 847.82 |

#### Latency per Query (sec)
| GPU | 8B | 13B | 32B | 70B | 405B | 671B |
| --- | --- | --- | --- | --- | --- | --- |
| H100 | 0.439 | 0.713 | 1.756 | 3.841 | 22.222 | 36.816 |
| H200 | 0.301 | 0.489 | 1.205 | 2.636 | 15.25 | 25.265 |
| B200 | 0.181 | 0.294 | 0.723 | 1.581 | 9.15 | 15.159 |

