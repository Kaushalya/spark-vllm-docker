# Visualizing vLLM and SGLang metrics with Prometheus and Grafana

vLLM and SGLang expose Prometheus metrics from their OpenAI-compatible servers
at `/metrics` when metrics are enabled. This setup runs Prometheus and Grafana
alongside either server, providing a web UI for request latency, throughput,
cache usage, and speculative-decoding performance.

The repository includes a ready-to-run monitoring stack:

- `docker-compose.metrics.yaml` runs Prometheus and Grafana with persistent
  named volumes. The images are pinned to Prometheus 3.13.2 and Grafana 13.1.3
  so deployments do not change unexpectedly when upstream `latest` tags move.
- `prometheus.yaml` scrapes vLLM on port 8000 and SGLang on port 30000 from the
  Docker host every five seconds. It is valid for one target to be down while
  the other backend is in use.
- `prometheus-rules.yaml` records one-minute averages for prompt, generation,
  and total token throughput for both servers, plus vLLM speculative-decoding
  counters.
- `grafana/provisioning/` configures Prometheus as Grafana's default data source
  and loads dashboards from disk.
- `grafana/dashboards/vllm-throughput.json` defines the default **vLLM
  Throughput** dashboard.
- `grafana/dashboards/vllm-spec-decode.json` defines the **vLLM Speculative
  Decoding** dashboard.
- `grafana/dashboards/sglang-dspark.json` defines the **SGLang Speculative
  Decoding** dashboard, including SGLang's native speculative-decoding gauges.

The default configuration assumes vLLM is listening on port `8000` on the
Docker host. Confirm the endpoint first:

```bash
curl http://127.0.0.1:8000/metrics
```

For SGLang, metrics must be enabled on the server; add `--enable-metrics` to
`sglang serve` if your build does not expose them by default. The setting
cannot be changed on an already-running server. With the standard SGLang port,
verify:

```bash
curl http://127.0.0.1:30000/metrics
```

`--enable-mfu-metrics` is optional and is not required by the provisioned
dashboard. It can add model-flops-utilization metrics when that additional
instrumentation is desired.

## Pinned SGLang build

The SGLang metrics in this repository were developed and verified against a
specific nightly build, pinned here for reproducibility:

| Property | Value |
| --- | --- |
| Upstream image tag | `lmsysorg/sglang:nightly-dev-20260814-c4271c3f` |
| Local image tag used on this host | `lmsysorg/sglang:qwen38-27b` |
| SGLang build commit | `c4271c3fe1262fc2adbd162c33b25de5255251c5` |
| Build | <https://github.com/sgl-project/sglang/actions/runs/31757139994> |
| Image ID | `sha256:0076dffa60b76b7bf033c04d05e0cc69d46f2b8cd60aa2468827782afe9bc38f` |
| FlashInfer | `0.6.18.dev20260807` |

This is a nightly development build without a release version, so pin the
image by the upstream tag or image ID above. The server used to verify the
dashboard was launched with the Qwen3.8-27B DSpark configuration:

```bash
docker run --rm -it --name sglang-dspark \
  --gpus all -p 30000:30000 \
  -v "${HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  lmsysorg/sglang:nightly-dev-20260814-c4271c3f \
  sglang serve --trust-remote-code \
    --model-path RadixArk/Qwen3.8-27B-NVFP4 \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark \
    --speculative-draft-attention-backend flashinfer \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.95 \
    --attention-backend flashinfer \
    --chunked-prefill-size 8192 \
    --disable-prefill-cuda-graph \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --mamba-radix-cache-strategy extra_buffer \
    --mamba-ssm-dtype float32 \
    --mamba-full-memory-ratio 6.626953125 \
    --host 0.0.0.0 --port 30000 \
    --enable-metrics
```

## Start Prometheus and Grafana

From the repository root, start both services:

```bash
docker compose -f docker-compose.metrics.yaml up -d
```

Run the same command after changing the Compose file or provisioning files.
Compose recreates affected containers when their configuration changes; a plain
`docker compose restart` does not apply new mounts or environment settings.

If either server uses a different host port, update its target in
`prometheus.yaml` before starting the stack. You can inspect service state and
logs with:

```bash
docker compose -f docker-compose.metrics.yaml ps
docker compose -f docker-compose.metrics.yaml logs prometheus grafana
```

Open the services in a browser, replacing `<spark-ip>` with the LAN address
of the head node:

- Grafana: `http://<spark-ip>:3000`
- Prometheus: `http://<spark-ip>:9090`

The initial Grafana login is `admin` / `admin`. Grafana will ask you to change
the password after the first login. The provisioned **vLLM Throughput**
dashboard remains the home dashboard. The vLLM and SGLang dashboards appear in
the **LLM Inference** folder.

## Provisioned data source and dashboard

The Compose stack provisions Prometheus as Grafana's default data source at
startup. No manual data-source setup is required. Check
`http://<spark-ip>:9090/targets`; the target for the active backend should be
`UP`.

The dashboard uses these recording rules from `prometheus-rules.yaml`:

| Recorded metric | Meaning |
| --- | --- |
| `vllm:generation_tokens_per_second:rate1m` | Output tokens per second |
| `vllm:prompt_tokens_per_second:rate1m` | Input/prompt tokens per second |
| `vllm:total_tokens_per_second:rate1m` | Combined input and output tokens per second |

The SGLang dashboard uses the equivalent recording rules:

| Recorded metric | Meaning |
| --- | --- |
| `sglang:generation_tokens_per_second:rate1m` | Output tokens per second |
| `sglang:prompt_tokens_per_second:rate1m` | Input/prompt tokens per second |
| `sglang:total_tokens_per_second:rate1m` | Combined input and output tokens per second |

The same dashboard also queries current vLLM metrics directly for latency and
KV-cache pressure:

| Panel | Source metric | Display |
| --- | --- | --- |
| Time to first token | `vllm:time_to_first_token_seconds` | P50 and P95 over five minutes |
| Inter-token latency | `vllm:inter_token_latency_seconds` | P50 over five minutes |
| End-to-end request latency | `vllm:e2e_request_latency_seconds` | P50 and P95 over five minutes |
| KV-cache usage | `vllm:kv_cache_usage_perc` | Maximum utilization across engines |

Latency metrics are Prometheus histograms, so the dashboard calculates rolling
percentiles with `histogram_quantile`. The KV-cache metric is a fraction from
0 to 1 and is displayed as a percentage. Latency panels show no value until
the five-minute window contains completed observations.

Inspect loaded rules at `http://<spark-ip>:9090/rules`, or enter one of the
recorded metric names in the Prometheus query UI. Prometheus needs at least two
samples before a rate appears, and the panels remain at zero while vLLM is
idle. The dashboard refreshes every five seconds and displays the most recent
one-minute averages plus a 15-minute history.

The provisioned dashboard can be edited in Grafana. For a durable,
version-controlled change, export the updated dashboard JSON and replace
`grafana/dashboards/vllm-throughput.json`; later provisioning-file changes can
overwrite a dashboard saved only in Grafana's database.

The upstream vLLM repository provides an example dashboard that can be
imported into Grafana:

<https://docs.vllm.ai/en/latest/examples/observability/prometheus_grafana/>

## SGLang speculative-decoding panels

The **SGLang Speculative Decoding** dashboard queries the SGLang exporter
directly for request state, cache pressure, latency histograms, and
speculative-decoding gauges. It applies to native MTP/NEXTN, DSpark, EAGLE,
and other SGLang speculative methods that expose these metrics. The pinned
SGLang build exposes metric names with the `sglang:` prefix, including the
colon. If you run a different SGLang build, confirm the names against the
server's `/metrics` output.

The generic speculative-decoding panels use:

- `sglang:spec_accept_rate` — accepted draft tokens divided by proposed draft
  tokens for the most recently reported batch.
- `sglang:spec_accept_length` — mean accepted drafts plus the target/bonus token
  per verification forward pass.
- `sglang:spec_block_accept_length` — uncapped full-block accepted length. It
  is exact only when DSpark cap-accept mode is active and is otherwise zero or
  unavailable.
- `sglang:spec_cap_length` — confidence-scheduled verification window including
  the bonus slot. It is used by DSpark and remains zero when no cap is
  scheduled.

These are gauges rather than cumulative counters. Query and graph them
directly; do not apply `rate()` to them. The dashboard also graphs the
`sglang:prompt_tokens_total` and `sglang:generation_tokens_total` counters
through the one-minute recording rules, and calculates latency percentiles
from these histograms:

- `sglang:time_to_first_token_seconds`
- `sglang:inter_token_latency_seconds`
- `sglang:e2e_request_latency_seconds`

SGLang metrics are created when the server starts, but throughput and latency
panels require request traffic and at least two Prometheus scrapes before they
show meaningful values. The pinned SGLang build updates the batch-level
speculative-decoding gauges every 40 decode steps by default, so a very short completion
can leave those gauges at zero even though token and verification counters
increase.

## Speculative decoding and MTP panels

The provisioned **vLLM Speculative Decoding** dashboard
(`grafana/dashboards/vllm-spec-decode.json`) covers these metrics out of the
box. It appears in the **LLM Inference** folder and queries the `vllm:spec_decode_*`
recording rules from the `vllm-spec-decode` group in `prometheus-rules.yaml`:

- **Draft acceptance rate** — fraction of draft tokens accepted
  (`vllm:spec_decode_acceptance_rate:rate1m`)
- **Mean accepted length** — tokens emitted per verification step, including
  the bonus token (`vllm:spec_decode_mean_accepted_length:rate1m`); with N
  speculative tokens it ranges from 1.0 to N+1
- **Draft tokens / second** (`vllm:spec_decode_draft_tokens_per_second:rate1m`)
- **Draft vs accepted tokens / second** — the gap between the two series is
  wasted draft compute
- **Acceptance rate by draft position** — per-position quality of the MTP
  cascade (`vllm:spec_decode_acceptance_rate_by_pos:rate1m`)

The stat panels query the recorded metrics as instant values. The ratio
rules (acceptance rate, mean accepted length, per-position) evaluate to no
value once traffic stops, so those panels show **No value** shortly after
the last request, when the last recorded sample leaves Prometheus'
five-minute lookback window; this avoids a misleading 100% acceptance rate
at idle. The token-rate rules have no such guard and keep recording zero
while vLLM is running, so the draft and accepted token panels show 0 at
idle.

If you add or replace dashboard files on disk, restart the Grafana container
for the file provider to reload them:

```bash
docker compose -f docker-compose.metrics.yaml restart grafana
```

If you prefer to build your own panels directly on the raw counters (for
example with a longer rate window), the following PromQL queries are
equivalent to the recording rules:

### Draft-token acceptance rate

```promql
(
  100 *
  sum(rate(vllm:spec_decode_num_accepted_tokens_total[5m]))
  /
  sum(rate(vllm:spec_decode_num_draft_tokens_total[5m]))
)
and on()
(
  sum(rate(vllm:spec_decode_num_draft_tokens_total[5m])) > 0
)
```

Use a Gauge or Time series visualization with the unit set to percent.

### Mean acceptance length

This convention includes the target/bonus token emitted by a verification
step:

```promql
(
  1 +
  sum(rate(vllm:spec_decode_num_accepted_tokens_total[5m]))
  /
  sum(rate(vllm:spec_decode_num_drafts_total[5m]))
)
and on()
(
  sum(rate(vllm:spec_decode_num_drafts_total[5m])) > 0
)
```

### Draft and accepted tokens per second

Draft tokens proposed:

```promql
sum(rate(vllm:spec_decode_num_draft_tokens_total[5m]))
```

Draft tokens accepted:

```promql
sum(rate(vllm:spec_decode_num_accepted_tokens_total[5m]))
```

### Acceptance rate by draft position

```promql
(
  100 *
  sum by (position) (
    rate(vllm:spec_decode_num_accepted_tokens_per_pos_total[5m])
  )
  /
  scalar(
    sum(rate(vllm:spec_decode_num_drafts_total[5m]))
  )
)
and on()
(
  sum(rate(vllm:spec_decode_num_drafts_total[5m]))
  > 0
)
```

The provisioned dashboard renders this metric as a Bar gauge with one bar
per draft position. If you build your own panel, use a Bar gauge or Time
series visualization and set the legend to `Position {{position}}`.

## Generate test traffic

Prometheus counters change only while requests are being processed. Generate
a sufficiently long response while viewing the dashboard, for example:

```bash
uvx llama-benchy \
  --base-url http://127.0.0.1:8000/v1 \
  --model google/gemma-4-26B-A4B-it \
  --pp 2048 \
  --tg 512
```

If a new panel initially shows no data, wait for at least two Prometheus
scrapes. You can temporarily change `[5m]` to `[30s]` while testing.

Acceptance rate alone is not the speculative-decoding speedup. Compare the
same llama-benchy or `vllm bench serve` workload with speculative decoding
enabled and disabled to measure the actual change in tokens per second and
latency.

## Stop the monitoring services

```bash
docker compose -f docker-compose.metrics.yaml down
```

The named volumes preserve Prometheus history and Grafana configuration. Add
`-v` only when you intentionally want to delete that stored monitoring data.
