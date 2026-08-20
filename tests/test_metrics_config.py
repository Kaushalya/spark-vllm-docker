#!/usr/bin/env python3

import json
import pathlib
import unittest

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


class MetricsComposeTests(unittest.TestCase):
    def test_monitoring_images_are_version_pinned(self):
        compose = yaml.safe_load(
            (ROOT / "docker-compose.metrics.yaml").read_text(encoding="utf-8")
        )

        self.assertEqual(
            compose["services"]["prometheus"]["image"],
            "prom/prometheus:v3.13.2",
        )
        self.assertEqual(
            compose["services"]["grafana"]["image"],
            "grafana/grafana:13.1.3",
        )

    def test_prometheus_scrapes_vllm_and_sglang(self):
        prometheus = yaml.safe_load(
            (ROOT / "prometheus.yaml").read_text(encoding="utf-8")
        )
        jobs = {
            job["job_name"]: job["static_configs"][0]["targets"]
            for job in prometheus["scrape_configs"]
        }

        self.assertEqual(jobs["vllm"], ["host.docker.internal:8000"])
        self.assertEqual(jobs["sglang"], ["host.docker.internal:30000"])


class VLLMMetricsDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dashboard = json.loads(
            (ROOT / "grafana/dashboards/vllm-throughput.json").read_text(
                encoding="utf-8"
            )
        )

    def test_panel_ids_are_unique_and_fit_the_grid(self):
        panels = self.dashboard["panels"]
        ids = [panel["id"] for panel in panels]

        self.assertEqual(len(ids), len(set(ids)))
        for panel in panels:
            grid = panel["gridPos"]
            self.assertGreater(grid["w"], 0)
            self.assertLessEqual(grid["x"] + grid["w"], 24)

    def test_all_panels_use_the_provisioned_prometheus_datasource(self):
        for panel in self.dashboard["panels"]:
            self.assertEqual(
                panel["datasource"],
                {"type": "prometheus", "uid": "prometheus"},
            )

    def test_latency_and_kv_cache_metrics_are_queried(self):
        expressions = {
            target["expr"]
            for panel in self.dashboard["panels"]
            for target in panel.get("targets", [])
        }
        rendered = "\n".join(expressions)

        for metric in (
            "vllm:time_to_first_token_seconds_bucket",
            "vllm:inter_token_latency_seconds_bucket",
            "vllm:e2e_request_latency_seconds_bucket",
            "vllm:kv_cache_usage_perc",
        ):
            self.assertIn(metric, rendered)

    def test_speculative_decoding_metrics_are_queried(self):
        expressions = {
            target["expr"]
            for panel in self.dashboard["panels"]
            for target in panel.get("targets", [])
        }
        rendered = "\n".join(expressions)

        for metric in (
            "vllm:spec_decode_acceptance_rate:rate1m",
            "vllm:spec_decode_mean_accepted_length:rate1m",
            "vllm:spec_decode_draft_tokens_per_second:rate1m",
            "vllm:spec_decode_accepted_tokens_per_second:rate1m",
            "vllm:spec_decode_acceptance_rate_by_pos:rate1m",
        ):
            self.assertIn(metric, rendered)

    def test_dashboard_combines_throughput_and_speculative_decoding(self):
        self.assertEqual(self.dashboard["title"], "vLLM Metrics")
        self.assertIn("speculative-decoding", self.dashboard["tags"])
        self.assertFalse(
            (ROOT / "grafana/dashboards/vllm-spec-decode.json").exists()
        )


class SGLangDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dashboard = json.loads(
            (ROOT / "grafana/dashboards/sglang-dspark.json").read_text(
                encoding="utf-8"
            )
        )

    def test_panel_ids_are_unique_and_fit_the_grid(self):
        panels = self.dashboard["panels"]
        ids = [panel["id"] for panel in panels]

        self.assertEqual(len(ids), len(set(ids)))
        for panel in panels:
            grid = panel["gridPos"]
            self.assertGreater(grid["w"], 0)
            self.assertLessEqual(grid["x"] + grid["w"], 24)

    def test_panels_use_the_provisioned_prometheus_datasource(self):
        for panel in self.dashboard["panels"]:
            self.assertEqual(
                panel["datasource"],
                {"type": "prometheus", "uid": "prometheus"},
            )

    def test_sglang_speculative_metrics_are_queried(self):
        expressions = {
            target["expr"]
            for panel in self.dashboard["panels"]
            for target in panel.get("targets", [])
        }
        rendered = "\n".join(expressions)

        for metric in (
            "sglang:generation_tokens_per_second:rate1m",
            "sglang:time_to_first_token_seconds_bucket",
            "sglang:spec_accept_rate",
            "sglang:spec_accept_length",
            "sglang:spec_block_accept_length",
            "sglang:spec_cap_length",
            "sglang:num_running_reqs",
        ):
            self.assertIn(metric, rendered)

    def test_dashboard_is_method_agnostic(self):
        self.assertEqual(self.dashboard["title"], "SGLang Speculative Decoding")
        self.assertIn("speculative-decoding", self.dashboard["tags"])
        self.assertNotIn("DSpark", self.dashboard["title"])

        panel_titles = {panel["title"] for panel in self.dashboard["panels"]}
        self.assertIn("Speculative acceptance rate", panel_titles)
        self.assertIn("Mean accepted length", panel_titles)


if __name__ == "__main__":
    unittest.main()
