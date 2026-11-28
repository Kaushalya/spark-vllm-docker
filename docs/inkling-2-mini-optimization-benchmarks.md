# Inkling 2 Mini Optimization Benchmarks on DGX Spark

- **Date:** 2026-11-28
- **Hardware:** Single NVIDIA DGX Spark (GB10, SM 12.1, 48 SMs, 119 GiB unified memory, ~207.7 GB/s measured copy bandwidth, NVIDIA driver 580.178.04)
- **Pinned image:** `vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e` (`vLLM 0.1.dev25855+g7cb5dae22.cu130`, PyTorch 2.14.0, CUDA 13.0)
- **Target checkpoint:** `thinkingmachines/Inkling-2-Mini` (`c9d028302cd0f035fdf8e7e91f089a4ce94d2195`)
- **Draft checkpoint:** `incoai/Inkling-2-Mini-DFlash2` (`4f9088ea66c21f2b99ed657d1941c306b03e84d2`, 7 probabilistic draft tokens)
- **Shared settings:** `--tensor-parallel-size 1`, `--max-model-len 262144`, `--gpu-memory-utilization 0.8`, `--moe-backend humming`, `--reasoning-parser inkling`, `--tool-call-parser inkling`, `--enable-auto-tool-choice`, plus `mods/inkling2-dflash2`
- **Companion HTML report:** [docs/inkling-2-mini-spark-findings.html](inkling-2-mini-spark-findings.html)

---

## 1. End-to-End Controlled Benchmark Sweep (17 Configurations)

All 17 configurations (`baseline_humming_dflash2` plus Phase 1 `exp1`–`exp7` and Phase 2 profile-guided `exp8`–`exp16`) were benchmarked on the same host with the same frozen workload:

- **Coding (`Coding`):** 4 tasks (`python_ttl_lru`, `typescript_async_map`, `rust_log_aggregator`, `postgres_event_analytics`) × 2 measured runs after per-task warmup, 512 output tokens, `temperature=0`, `seed=42`, `ignore_eos=true`, `reasoning_effort="none"`. **Mean Step** = `mean_accepted_len / mean_coding_decode_tps`.
- **Sampled (`Sampled`):** 3 technical prompts, 512 output tokens, `temperature=0.6`, `ignore_eos=true`, `reasoning_effort="high"`.
- **Concurrency (`4-req` / `8-req`):** 4 and 8 concurrent requests, 256 output tokens each, `temperature=0.6`, `reasoning_effort="none"`.
- **Prefill (`PF 32K`):** Warmed prefill on fresh unique-nonce ~32.5K-token prompts (2 measured repeats after 1 warmup).
- **Long-context decode (`Dec 41K`):** ~41K-token cold prompt followed by 256 output tokens at `temperature=0`.

### Phase 1 Sweep (`baseline` + `exp1`–`exp7`)

| Configuration | Coding (tok/s) | Mean Step (ms) | Acc % | TTFT (ms) | Sampled (tok/s) | 4-req (tok/s) | 8-req (tok/s) | PF 32K (tok/s) | Dec 41K (tok/s) | 41K Step | KV Capacity (tok) | Greedy Hash Match vs. Baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline_humming_dflash2` | 55.77 | 59.0 | 34.3% | 217.5 | 50.91 | 107.91 | 178.35 | 2,850.5 | 47.03 | 64.3 ms | 1,450,299 | 4/4 (ref) |
| `exp1_fp8_kv` | 60.26 | 58.2 | 36.8% | 231.2 | 57.50 | 102.93 | 166.18 | 2,666.0 | 46.24 | 61.8 ms | **2,282,491** | 0/4 (det) |
| **`exp2_dflash2_quant` (Lossless recipe)** | **59.04** | **56.1** | 34.6% | **204.0** | **54.22** | **114.70** | 178.16 | 2,869.0 | **51.36** | **61.3 ms** | 1,458,487 | **4/4 exact** |
| `exp3_moe_w4a16_sk` | 60.14 | 61.9 | 39.1% | 231.3 | 51.04 | 112.33 | 165.45 | 2,933.1 | 42.35 | 69.5 ms | 1,464,628 | 0/4 (non-det) |
| `exp4_lossless_stack` | 55.52 | 54.9 | 31.4% | 205.4 | 55.60 | 112.55 | 173.83 | 3,154.2 | 48.69 | 62.8 ms | 1,468,722 | 0/4 (det) |
| `exp5_target_dense_mxfp8` | 70.88 | 50.4 | 37.3% | 192.4 | 60.42 | 121.09 | 177.20 | 3,091.9 | 49.08 | 56.1 ms | 1,468,722 | 0/4 (det) |
| **`exp6_mxfp8_all` (MXFP8 recipe)** | **73.40** | **48.9** | **36.9%** | **192.1** | **68.41** | **126.10** | **191.10** | 3,084.7 | **60.11** | **53.2 ms** | 1,452,005 | 0/4 (det) |
| `exp7_mxfp8_all_b4096` | 73.00 | 48.7 | 36.6% | 193.9 | 68.03 | 126.78 | 181.96 | **3,335.5** | 53.81 | 54.3 ms | 1,106,066 | 0/4 (det, 4/4 vs. `exp6`) |

### Phase 2 Profile-Guided Sweep (`exp8`–`exp16`)

| Configuration | Coding (tok/s) | Mean Step (ms) | Acc % | TTFT (ms) | Sampled (tok/s) | 4-req (tok/s) | 8-req (tok/s) | PF 32K (tok/s) | Dec 41K (tok/s) | 41K Step | KV Capacity (tok) | Quality Verification |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `exp8_lossless_v2` | 55.11 | 54.6 | 31.4% | 202.9 | 58.14 | 101.31 | 150.28 | 3,138.5 | 42.80 | 62.3 ms | 1,462,923 | Math / Tool / 30K+94K 3/3 |
| `exp9_mxfp8_w4a16_tuned` | 68.10 | 48.0 | 33.7% | 197.3 | 66.01 | 127.97 | 178.08 | 2,854.1 | 57.91 | 53.7 ms | 1,456,099 | Math / Tool / 30K+94K 3/3 |
| `exp10_target_nvfp4_dense` | 66.92 | 42.7 | 33.5% | 182.6 | 70.44 | 127.20 | 184.71 | 3,094.1 | 51.05 | 51.9 ms | 1,461,217 | Math / Tool / Py / 30K+94K 3/3 |
| `exp11_max_nvfp4` | **73.60** | **42.5** | 30.6% | **171.0** | 68.68 | 126.13 | 199.16 | 3,108.4 | **61.49** | 49.2 ms | **1,503,863** | Math / Tool / 30K+94K 3/3 |
| `exp12_mxfp8_w4a16_tuned` | 67.84 | 48.8 | 34.6% | 200.2 | 66.69 | 117.62 | 183.82 | 2,865.8 | 58.65 | 56.6 ms | 1,449,617 | **FAIL @ 94K** (looping) |
| `exp13_nvfp4_dense_mxfp8_head` | 67.91 | 43.9 | 35.2% | 181.7 | 66.89 | 120.27 | 195.86 | 3,106.2 | 56.08 | 51.1 ms | 1,456,782 | Math / Tool / Py / 30K+94K 3/3 |
| **`exp14_max_nvfp4_dflash_mxfp8` (NVFP4 recipe)** | 71.96 | **43.2** | 30.3% | 175.3 | **75.97** | **140.30** | **207.94** | **3,136.8** | 58.36 | **48.9 ms** | **1,501,133** | **All PASS** (Math / Tool / Py / 30K+94K 3/3) |
| `exp15_mxfp8_dense_coda_nvfp4` | 73.23 | 47.3 | 35.3% | 183.7 | 63.36 | 137.01 | 204.59 | 3,102.5 | 57.69 | 52.6 ms | 1,488,169 | Math / Tool / 30K+94K 3/3 |
| `exp16_nvfp4_dense_coda_mxfp8_head` | 70.39 | 44.5 | 31.3% | 174.9 | 71.85 | 130.93 | 205.96 | 3,080.0 | 47.62 | 50.0 ms | 1,496,357 | Math / Tool / 30K+94K 3/3 |

### Per-Task Coding Breakdown (Median tok/s)

| Configuration | Python TTL LRU | TypeScript async map | Rust log aggregator | PostgreSQL analytics | Overall Median | Mean Accepted Len |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_humming_dflash2` | 66.83 | 51.65 | 53.49 | 58.58 | 55.77 | 3.40 |
| `exp1_fp8_kv` | 76.98 | 54.39 | 66.13 | 48.58 | 60.26 | 3.58 |
| **`exp2_dflash2_quant` (Lossless)** | 70.52 | 55.57 | 56.73 | 61.17 | **59.04** | 3.42 |
| `exp3_moe_w4a16_sk` | 64.38 | 58.90 | 58.20 | 60.01 | 60.14 | 3.74 |
| `exp4_lossless_stack` | 72.73 | 53.75 | 56.86 | 49.50 | 55.52 | 3.20 |
| `exp5_target_dense_mxfp8` | 80.35 | 71.04 | 64.85 | 70.76 | 70.88 | 3.61 |
| **`exp6_mxfp8_all` (MXFP8)** | 80.98 | 65.57 | 72.95 | 73.65 | **73.40** | **3.58** |
| `exp7_mxfp8_all_b4096` | 80.74 | 65.83 | 73.09 | 72.83 | 73.00 | 3.56 |
| `exp8_lossless_v2` | 72.10 | 53.27 | 57.18 | 51.41 | 55.11 | 3.20 |
| `exp9_mxfp8_w4a16_tuned` | 81.42 | 71.45 | 64.82 | 62.12 | 68.10 | 3.36 |
| `exp10_target_nvfp4_dense` | 116.67 | 62.33 | 66.91 | 67.04 | 66.92 | 3.34 |
| `exp11_max_nvfp4` | 82.45 | 72.94 | 74.35 | 66.50 | **73.60** | 3.14 |
| `exp12_mxfp8_w4a16_tuned` | 81.03 | 69.87 | 65.97 | 63.76 | 67.84 | 3.43 |
| `exp13_nvfp4_dense_mxfp8_head` | 115.78 | 64.80 | 67.07 | 68.17 | 67.91 | 3.46 |
| **`exp14_max_nvfp4_dflash_mxfp8` (NVFP4)** | 81.86 | 70.79 | 72.80 | 63.10 | **71.96** | 3.12 |
| `exp15_mxfp8_dense_coda_nvfp4` | 67.90 | 75.03 | 71.58 | 79.01 | 73.23 | 3.47 |
| `exp16_nvfp4_dense_coda_mxfp8_head` | 78.26 | 68.56 | 72.20 | 67.95 | 70.39 | 3.19 |

---

## 2. 12-Iteration PyTorch/CUDA Kernel Profile (`exp6_mxfp8_all`)

To pinpoint where the remaining decode step time went after Phase 1, we captured a 12-iteration steady-state PyTorch/CUDA Chrome trace (`delay_iterations=10, max_iterations=12`) on `inkling-2-mini-dflash2-mxfp8`:

- **Wall span:** `526.65 ms` (**`43.89 ms/iter`**, range `38.38–48.33 ms/iter` across the 12 profiled steps)
- **GPU utilization:** **98.9% busy** (`43.41 ms/iter` active; only **`0.48 ms/iter`** CPU/idle gaps)
- **PDL overlap separation:** Raw summed kernel time was `69.82 ms/iter` because `_situ_quantize_kernel` (`15.18 ms` raw → `0.10 ms` exclusive) and `_combine_kernel` (`9.09 ms` raw → `0.07 ms` exclusive) launch concurrently via Hopper/Blackwell PDL (`gdc_wait`) and spin-wait on the preceding MoE GEMM. Attributing exclusive completion-delta wall time gives the exact breakdown of the `43.89 ms/iter`:

| Component | Calls/iter | Excl ms/iter | Wall % | Measured Bandwidth / Notes |
| --- | ---: | ---: | ---: | --- |
| Target MoE NVFP4 (`w13` + `w2`, 44 visits) | 88 | **23.00 ms** | **52.4%** | `w13` `15.23 ms` (`346 µs`), `w2` `7.77 ms` (`177 µs`), ~235–247 GB/s |
| Target Dense MXFP8 (`qkvr` + `wo_ud` + `k_rel`, 48 visits) | 152 | **6.17 ms** | **14.1%** | `qkvr` `3.77 ms`, `wo_ud` `1.61 ms`, `k_rel` `0.79 ms` (~218–233 GB/s) |
| Target MoE MXFP8 Coda (`w13` + `w2`, 4 visits, layers 20–23) | 8 | **3.95 ms** | **9.0%** | Only 4 layers: `w13` `660 µs`, `w2` `327 µs` (~210–225 GB/s) |
| DFlash2 Draft Total (backbone MXFP8 + `lm_head` NVFP4 + attn) | 46 | **3.16 ms** | **7.2%** | Backbone `1.90 ms`, `lm_head` `1.14 ms` (`227 GB/s`), attn/selector `0.12 ms` |
| Target `lm_head` MXFP8 (`201216×2304`) | 1 | **2.10 ms** | **4.8%** | `227 GB/s` (saturated for 8-bit weights) |
| Remaining BF16 CUTLASS GEMMs | 163 | **1.77 ms** | **4.0%** | Loop message controllers (`blend`), DFlash2 conv projections, dynamic sconv |
| Target MoE overhead (`_fused_route`, `moe_align`, `_prepare_rows`, `_situ`, `_combine`) | 328 | **1.03 ms** | **2.4%** | `_prepare_rows` is only `0.12 ms/iter`; PDL already hides `_situ` & `_combine` |
| Target Attention + Prologue (48 visits) | 144 | **0.85 ms** | **1.9%** | Short context in trace; grows at 41K |
| Target `attn_res` / `loop_blend` / norms / `sconv` | 313 | **0.93 ms** | **2.1%** | — |
| Other CUDA + GPU idle gaps | — | **0.98 ms** | **2.2%** | `0.50 ms` misc CUDA + `0.48 ms` idle |
| **Total wall time per decode step** | — | **43.89 ms** | **100.0%** | |

Key profile takeaways that drove Phase 2:
1. **MoE preparation overhead is already tiny (`1.03 ms/iter`, 2.4%)** — rewriting Inkling's MoE layout to eliminate `_prepare_rows` (`0.12 ms/iter`) is low-leverage.
2. **88% of decode wall time is pure memory-bound weight streaming already near GB10's physical bandwidth ceiling (~220–247 GB/s).**
3. **The three largest remaining reducible weight streams** were Target Dense MXFP8 (`6.17 ms/iter`), the 4 MXFP8 Coda MoE layers (`3.95 ms/iter`), and Target `lm_head` MXFP8 (`2.10 ms/iter`) — **12.22 ms/iter** of 8-bit weights reducible to 4-bit NVFP4 (~6 ms/step savings).

---

## 3. Experiment Descriptions

| Label | Flags / `mods/inkling2-perf-opt` Env Vars | Summary of Findings |
| --- | --- | --- |
| `baseline_humming_dflash2` | `recipes/inkling-2-mini-dflash2.yaml` defaults | Reference baseline (`55.77` coding, `50.91` sampled, `107.9`/`178.4` 4/8-req, `2,851` 32K prefill, `47.03` @ 41K). |
| `exp1_fp8_kv` | `--kv-cache-dtype fp8` | Expands KV capacity **+57%** (`1.45M → 2.28M` tok, `8.71×` at 262K) and trims 1-req step latency 2–4%, but slows 32K prefill (−6.5%) and 4/8-req concurrency (−5–7%) due to FP8→BF16 dequant in Triton `_load_fp8_kv`. |
| `exp2_dflash2_quant` | `INKLING2_OPT_DFLASH_LMHEAD=nvfp4`<br>`INKLING2_OPT_DFLASH_BACKBONE=mxfp8` | **Best strictly target-lossless config.** Target weights/activations/`lm_head` untouched; **4/4 bitwise-identical greedy hashes** with baseline; +5.9% coding, +6.5% sampled, +6.3% 4-req, +9.2% decode @ 41K. |
| `exp3_moe_w4a16_sk` | `INKLING2_OPT_MOE=w4a16_sk` | BF16 activations raised coding acceptance `34.3% → 39.1%`, but Stream-K BF16 atomics broke greedy determinism and W4A16 regressed 8-req throughput (`165.5` vs. `178.4 tok/s`). |
| `exp4_lossless_stack` | `INKLING2_OPT_MOE=mxfp8_only`, draft quant, sparse CG | Keeps deterministic W4A4 on NVFP4 layers with cached alignment, switches the 4 MXFP8 coda layers to Humming MXFP8 W8A16, and trims graph memory −21%. Lifts 32K prefill **+10.6%** (`3,154 tok/s`). |
| `exp5_target_dense_mxfp8` | `exp4` + `INKLING2_OPT_TARGET_DENSE=mxfp8` | Online block-32 MXFP8 W8A16 on target `qkvr` + `wo_ud` (BF16 target `lm_head`). Coding `70.88 tok/s` (+27.1%), sampled `60.42 tok/s` (+18.7%). |
| `exp6_mxfp8_all` | `exp4` + `INKLING2_OPT_TARGET_DENSE=mxfp8_all` | **Best greedy coding & 41K decode config (MXFP8 recipe).** Also quantizes target `lm_head` to MXFP8 W8A16: **73.40 tok/s** coding (+31.6%, `36.9%` acc), **68.41 tok/s** sampled (+34.4%), **126.1 / 191.1 tok/s** 4/8-req, **60.11 tok/s** @ 41K. |
| `exp7_mxfp8_all_b4096` | `exp6` + `--max-num-batched-tokens 4096` | Peak 32K prefill (**3,335.5 tok/s**, +17.0% vs. baseline) and 4/4 hashes vs. `exp6`, at the cost of KV capacity (`1.45M → 1.11M`) and 8-req (`182.0 tok/s`). |
| `exp8_lossless_v2` | `exp4` + `INKLING2_OPT_DFLASH_BACKBONE=nvfp4` + `INKLING2_OPT_DFLASH_CONV=1` + `INKLING2_OPT_W4A4_TUNE=1` | Quantizing the 5-layer DFlash2 draft backbone to NVFP4 dropped acceptance (`31.4%` coding, `23.8%` @ 41K) and hurt 8-req (`150.3 tok/s`). Proved DFlash2 backbone should stay MXFP8. |
| `exp9_mxfp8_w4a16_tuned` | `mxfp8_all` + `INKLING2_OPT_MOE=w4a16` + `INKLING2_OPT_W4A16_TUNE=1` + `DFLASH_BACKBONE=nvfp4` | GB10-tuned W4A16 matched W4A4 step time (`48.0 ms/step`), lifting 4-req to `128.0 tok/s`, but regressed prefill (`2,854 tok/s`) and 8-req (`178.1 tok/s`). |
| `exp10_target_nvfp4_dense` | `mxfp8_only` + `W4A4_TUNE=1` + `TARGET_DENSE=nvfp4_mxfp8_head` + `DFLASH_BACKBONE=nvfp4` | Cut mean step time to `42.7 ms` (`python_ttl_lru` hit `116.67 tok/s`), lifting sampled decode to `70.44 tok/s`. |
| `exp11_max_nvfp4` | `MOE=coda_nvfp4` + `W4A4_TUNE=1` + `W4A16_TUNE=1` + `TARGET_DENSE=nvfp4_all` + `DFLASH_BACKBONE=nvfp4` | Cut mean step latency to **`42.5 ms/step`** and model footprint to **`17.99 GiB`** (`1.504M` KV tok), hitting `73.60 tok/s` coding, `199.16 tok/s` 8-req, and `61.49 tok/s` @ 41K, though `DFLASH_BACKBONE=nvfp4` capped acceptance (`30.6%`). |
| `exp12_mxfp8_w4a16_tuned` | `exp9` with `DFLASH_BACKBONE=mxfp8` | Restoring MXFP8 DFlash2 did not overtake `exp6` (`67.84` coding, `2,866` prefill) and **failed the 94K needle check** with token repetition. Ruled out. |
| `exp13_nvfp4_dense_mxfp8_head` | `exp10` with `DFLASH_BACKBONE=mxfp8` | Restored coding acceptance to `35.2%` (`43.9 ms/step`, `195.86 tok/s` 8-req) and passed all quality checks including executable Python unittest and 94K 3/3 needles. |
| **`exp14_max_nvfp4_dflash_mxfp8`** | **`MOE=coda_nvfp4` + `W4A4_TUNE=1` + `W4A16_TUNE=1` + `TARGET_DENSE=nvfp4_all` + `DFLASH_BACKBONE=mxfp8`** | **Fastest sampled reasoning, concurrency, step latency & lowest VRAM (NVFP4 recipe).** Cuts mean step time to **`43.2 ms/step`** (−26.8% vs. baseline, −11.7% vs. `exp6`), shrinks model memory to **`18.20 GiB`** (`1.501M` KV tok), sets new records on **sampled decode (`75.97 tok/s`, +49.2%)**, **4-req (`140.30 tok/s`, +30.0%)**, and **8-req (`207.94 tok/s`, +16.6% — first >200 tok/s config)**, and passed every quality check. |
| `exp15_mxfp8_dense_coda_nvfp4` | `exp6` + `MOE=coda_nvfp4` + `W4A4_TUNE=1` + `W4A16_TUNE=1` | Isolates NVFP4 on only the 4 coda MoE layers (`47.3 ms/step`, `73.23` coding at `35.3%` acc, `137.0 / 204.6 tok/s` 4/8-req). |
| `exp16_nvfp4_dense_coda_mxfp8_head` | `exp14` with `TARGET_DENSE=nvfp4_mxfp8_head` (MXFP8 target `lm_head`) | Did not recover greedy acceptance (`31.3%`) and trailed `exp14` across all metrics (`70.39` coding, `71.85` sampled, `130.9 / 206.0` 4/8-req). |

---

## 4. Standalone GB10 Kernel Microbenchmarks

Measured directly on GB10 inside `vllm/vllm-openai:inkling-2-mini`:

### MoE Layer (128 experts, top-8, H=2304, I=1152, single layer)

| Tokens (`M`) | NVFP4 Triton | NVFP4 Humming W4A4 | NVFP4 Humming W4A16 (SK) | MXFP8 Triton | MXFP8 Humming W8A16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1.055 ms | 0.989 ms | **0.899 ms** | 1.837 ms | **1.710 ms** |
| 32 | 2.315 ms | 2.161 ms | **2.100 ms** | 4.204 ms | **4.003 ms** |
| 64 | 2.807 ms | 2.623 ms | **2.580 ms** | 4.615 ms | **4.510 ms** |
| 512 | 3.759 ms | 3.266 ms | **3.097 ms** | 5.670 ms | **5.194 ms** |
| 2048 | 8.537 ms | **4.783 ms** | 5.217 ms | 14.621 ms | **7.531 ms** |

### Phase 2 GB10 Tile Tuning & Cold-DRAM CUDA-Graph Checks

- **MoE `w13` (`2304×2304`, 24 active experts, 71.7 MB):** Humming `Sm121Heuristics` W4A4 (`block=(16,128,256)`) = `291.1 µs` (`246.2 GB/s`); GB10-tuned `block=(16,256,128), warp=(16,64,128), s=3` = **`283.5 µs` (`252.8 GB/s`)**. W4A16 default (`block=(8,128,64)`) = `324.8 µs` (`220.6 GB/s`); GB10-tuned `block=(8,256,64), warp=(8,64,64), s=4` = **`299.5 µs` (`239.3 GB/s`)**.
- **Cold-DRAM CUDA-Graph Dense `6656×2304` (`M=8`, 16 distinct layers = 138–253 MB >> L2):**
  - `MXFP8 (use_stream_k=True)`: `77.4 µs` (`204.2 GB/s`)
  - `MXFP8 (use_stream_k=False)`: **`71.9 µs` (`219.8 GB/s`)** (+7.6% faster without Stream-K)
  - `NVFP4 (default heur, sk=False)`: `42.4 µs` (`203.3 GB/s`)
  - `NVFP4 (8×128×64, warp 8×32×64, s=4, c=2, sk=False)`: **`40.8 µs` (`211.5 GB/s`, 1.76× faster than MXFP8)**
- **Tokenizer vocab check (ruled out naive sparse draft vocab):** Inkling 2 Mini's BPE vocabulary is not frequency-ordered (`' if'` is at ID `150003`, `' CONST'` near `199000`), so a naive prefix slice would drop common code tokens.

### Output Head (`N=200064`, `K=2304`) & 5-Layer DFlash2 Backbone (Eager)

| Op / Batch (`M`) | cuBLAS BF16 | `sm12x_bf16_gemv` | Humming MXFP8 W8A16 | Humming NVFP4 W4A16 |
| --- | ---: | ---: | ---: | ---: |
| `lm_head` `M=7` (1-req draft) | 4.063 ms | 3.916 ms | 2.109 ms | **1.174 ms** |
| `lm_head` `M=8` (1-req target) | 3.993 ms | 3.904 ms | 2.064 ms | **1.162 ms** |
| `lm_head` `M=28` (4-req draft) | 4.542 ms | — | 2.094 ms | **1.220 ms** |
| `lm_head` `M=56` (8-req draft) | 4.674 ms | — | 2.210 ms | **1.377 ms** |
| 5-layer DFlash2 `M=8` | 2.913 ms | 3.089 ms | **1.398 ms** | 1.805 ms *(eager SK jitter)* |
| 5-layer DFlash2 `M=16` | 2.780 ms | 3.213 ms | 1.480 ms | **0.795 ms** |
| 5-layer DFlash2 `M=32` | 3.204 ms | — | 1.228 ms | **0.474 ms** |

---

## 5. Quality & Functional Verification

| Profile | Math (`17*23+19=410`) | Tool Calling (`get_weather`) | 3-Needle @ 30K | 3-Needle @ 94K | Notes |
| --- | --- | --- | --- | --- | --- |
| `exp2` (`lossless`) | PASS | PASS | 3/3 | 3/3 | **4/4 bitwise-identical greedy hashes** vs. baseline (`exp8` lossless also confirmed `max_tokens=3072` truncation explains the earlier Python length limit) |
| `exp6` (`mxfp8`) | PASS (`410`) | PASS | 3/3 (`13.3 s`) | 3/3 (`42.4 s`) | Highest greedy coding acceptance (`36.9%`, above baseline `34.3%`) |
| `exp14` (`nvfp4`) | PASS (`410`) | PASS | 3/3 (`14.0 s`) | 3/3 (`41.2 s`) | Also passed executable Python unittest check (`PASS=True`); ~6-point lower greedy coding acceptance (`30.3%`) due to 4-bit target quantization, with strong sampled reasoning acceptance (`30.9%`) |
| `exp12` (`w4a16_tuned`) | PASS | PASS | 3/3 | **FAIL (0/3)** | Degenerated into repetition at 94K context; excluded from recipes |

---

## 6. Packaged Recipes (Three-Tier Pareto Frontier)

All three winning tiers are packaged in `recipes/` using [mods/inkling2-perf-opt](../mods/inkling2-perf-opt/README.md):

```bash
# 1. Strictly target-lossless (exp2_dflash2_quant — 4/4 exact baseline hashes)
./run-recipe.sh inkling-2-mini-dflash2-lossless --solo --setup

# 2. Best greedy coding & 41K long-context decode (exp6_mxfp8_all — 73.40 coding, 60.11 @ 41K)
./run-recipe.sh inkling-2-mini-dflash2-mxfp8 --solo --setup

# 3. Fastest sampled reasoning, 4/8-req concurrency & lowest VRAM (exp14_max_nvfp4_dflash_mxfp8 — 75.97 sampled, 140.3 / 207.9 4/8-req, 43.2 ms/step, 18.2 GiB)
./run-recipe.sh inkling-2-mini-dflash2-nvfp4 --solo --setup
```

---

## 7. Remaining Ceiling on Single GB10

With `exp14` streaming 4-bit weights across all 48 MoE visits, dense projections, and `lm_head` at **~212–253 GB/s** (**98.9% GPU utilization**), single-GPU decode on GB10 is now at the LPDDR5X physical bandwidth wall (`~43 ms/step`). Further single-Spark speedups beyond this point require either:
1. **Higher speculative acceptance** (e.g., a DFlash2 draft model tuned for quantized targets or tree/multi-branch verification),
2. **SM121 Triton attention & native FP8 KV kernel optimization** (`attention_sm90.py` / `_load_fp8_kv`) for ultra-long contexts (`>100K` tokens) where KV reads overtake weight streaming, or
3. **On-disk caching of repacked Humming weights** to reduce cold startup time (~110 s model load).
