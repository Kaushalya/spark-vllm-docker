#!/usr/bin/env python3

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_model_configs", ROOT / "benchmark-model-configs.py"
)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class SpecMetricsTests(unittest.TestCase):
    def test_parses_and_diffs_prometheus_counters(self):
        before = BENCH.parse_spec_metrics(
            """
vllm:spec_decode_num_drafts_total 10
vllm:spec_decode_num_draft_tokens_total 40
vllm:spec_decode_num_accepted_tokens_total 20
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 9
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 6
"""
        )
        after = BENCH.parse_spec_metrics(
            """
vllm:spec_decode_num_drafts_total 20
vllm:spec_decode_num_draft_tokens_total 80
vllm:spec_decode_num_accepted_tokens_total 50
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 19
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 14
"""
        )

        delta = BENCH.spec_metric_delta(before, after)

        self.assertIsNotNone(delta)
        self.assertAlmostEqual(delta["acceptance_percent"], 75.0)
        self.assertAlmostEqual(delta["mean_accepted_length"], 4.0)
        self.assertEqual(
            delta["acceptance_by_position_percent"], {"0": 100.0, "1": 80.0}
        )

    def test_missing_metrics_returns_none(self):
        empty = BENCH.parse_spec_metrics("# no speculative metrics\n")
        self.assertIsNone(BENCH.spec_metric_delta(empty, empty))


class SummaryTests(unittest.TestCase):
    def test_summarizes_stable_outputs(self):
        runs = [
            {
                "decode_tokens_per_second": rate,
                "ttft_seconds": ttft,
                "total_seconds": total,
                "completion_tokens": 512,
                "response_sha256": "same",
            }
            for rate, ttft, total in ((18.0, 0.5, 29.0), (20.0, 0.4, 27.0), (19.0, 0.6, 28.0))
        ]

        summary = BENCH.summarize_runs(runs)

        self.assertEqual(summary["median_decode_tokens_per_second"], 19.0)
        self.assertEqual(summary["median_ttft_seconds"], 0.5)
        self.assertTrue(summary["outputs_stable_within_config"])
        self.assertEqual(summary["response_sha256"], "same")

    def test_marks_varying_outputs(self):
        runs = [
            {
                "decode_tokens_per_second": 20.0,
                "ttft_seconds": 0.5,
                "total_seconds": 27.0,
                "completion_tokens": 512,
                "response_sha256": digest,
            }
            for digest in ("first", "second")
        ]

        summary = BENCH.summarize_runs(runs)

        self.assertFalse(summary["outputs_stable_within_config"])
        self.assertIsNone(summary["response_sha256"])


class StreamSchemaTests(unittest.TestCase):
    def test_reads_current_reasoning_field(self):
        self.assertEqual(
            BENCH.delta_text_pieces({"reasoning": "think", "content": "answer"}),
            ["think", "answer"],
        )

    def test_reads_legacy_reasoning_content_field(self):
        self.assertEqual(
            BENCH.delta_text_pieces({"reasoning_content": "think", "content": None}),
            ["think"],
        )


class ServerAddressTests(unittest.TestCase):
    def test_port_overrides_base_url_port(self):
        self.assertEqual(
            BENCH.resolve_base_url("http://127.0.0.1:8000/v1", 30000),
            "http://127.0.0.1:30000/v1",
        )

    def test_port_override_preserves_ipv6_host_and_url_components(self):
        self.assertEqual(
            BENCH.resolve_base_url("https://[::1]:8000/v1/?mode=test#fragment", 30000),
            "https://[::1]:30000/v1/?mode=test#fragment",
        )

    def test_omitted_port_preserves_existing_base_url(self):
        self.assertEqual(
            BENCH.resolve_base_url("http://server.example:9000/v1/"),
            "http://server.example:9000/v1",
        )

    def test_rejects_credentials_before_replacing_port(self):
        with self.assertRaises(BENCH.BenchmarkError):
            BENCH.resolve_base_url("http://user:secret@localhost:8000/v1", 30000)

    def test_parser_accepts_server_port(self):
        args = BENCH.build_parser().parse_args(
            ["run", "--label", "sglang", "--port", "30000"]
        )

        BENCH.validate_args(args)

        self.assertEqual(args.port, 30000)

    def test_rejects_out_of_range_port(self):
        args = BENCH.build_parser().parse_args(
            ["run", "--label", "invalid", "--port", "65536"]
        )

        with self.assertRaises(BENCH.BenchmarkError):
            BENCH.validate_args(args)


if __name__ == "__main__":
    unittest.main()
