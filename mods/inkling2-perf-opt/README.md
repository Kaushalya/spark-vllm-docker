# Inkling 2 Mini SM12x performance optimizations

This opt-in runtime mod applies to the vLLM image pinned by the
[Inkling 2 Mini DFlash2 recipes](../../recipes/inkling-2-mini-dflash2.yaml)
(`vllm 0.1.dev25855+g7cb5dae22.cu130`) and layers cleanly on top of
`mods/inkling2-dflash2`.

Each optimization is enabled via container environment variables so a single
patch supports strictly target-lossless, MXFP8-dense, and full NVFP4
configurations:

| Variable | Values | Effect |
| --- | --- | --- |
| `INKLING2_OPT_DFLASH_LMHEAD` | `""`, `mxfp8`, `nvfp4` | Online-quantizes a draft-only copy of `lm_head` via Humming W4A16/W8A16 (`use_stream_k=False`). Target verification still uses the untouched BF16 `lm_head` (100% lossless to target outputs). |
| `INKLING2_OPT_DFLASH_BACKBONE` | `""`, `mxfp8`, `nvfp4` | Online-quantizes the 5-layer DFlash2 draft backbone (`qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`, `fc`, and fused context KV) via Humming (`use_stream_k=False`). Recommended: `mxfp8` (100% lossless to target outputs while preserving full draft acceptance). |
| `INKLING2_OPT_DFLASH_CONV` | `""`, `"1"` | Also quantizes DFlash2 grouped-conv `kernel_projection` layers when `INKLING2_OPT_DFLASH_BACKBONE` is set. |
| `INKLING2_OPT_MOE` | `""`, `mxfp8_only`, `coda_nvfp4`, `w4a16`, `w4a16_sk`, `w4a16_coda_nvfp4` | `mxfp8_only` keeps NVFP4 layers on deterministic Humming W4A4 (caching `moe_align_block_size` between `w13` and `w2`) and switches the 4 MXFP8 coda layers (layers 20–23) from Triton to Humming MXFP8 W8A16. `coda_nvfp4` additionally dequantizes and re-quantizes those 4 coda MoE layers from MXFP8 to NVFP4 W4A16 at load time. `w4a16` / `w4a16_sk` run all 24 layers through `HummingExperts` W{4,8}A16. |
| `INKLING2_OPT_W4A4_TUNE` | `""`, `"1"` | Applies GB10-tuned `block=(16, 256, 128), warp=(16, 64, 128)` tile for Humming W4A4 `w13` (`(4, 1024]` range). |
| `INKLING2_OPT_W4A16_TUNE` | `""`, `"1"` | Applies GB10-tuned `block=(8, 256, 64), warp=(8, 64, 64), stages=4` tile for Humming NVFP4 W4A16 MoE (`(1, 512]` range). |
| `INKLING2_OPT_TARGET_DENSE` | `""`, `mxfp8`, `mxfp8_all`, `nvfp4`, `nvfp4_mxfp8_head`, `nvfp4_all` | Online quantization of Inkling 2 Mini's dense BF16 attention projections (`qkvr`, `wo_ud`, skipping unused fused/split copies). `mxfp8_all` / `nvfp4_mxfp8_head` quantize the target `lm_head` to MXFP8 W8A16; `nvfp4_all` quantizes dense projections and target `lm_head` to NVFP4 W4A16 with GB10 `M<=8` tile tuning. |

Reapplying the mod to the same fresh container is safe and idempotent.
See [Inkling 2 Mini on DGX Spark](../../docs/inkling-2-mini-spark-findings.html)
and [Optimization Benchmarks](../../docs/inkling-2-mini-optimization-benchmarks.md)
for controlled single-Spark measurements.
